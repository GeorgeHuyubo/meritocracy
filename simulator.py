"""Meritocracy 无界面模拟器，用来跑平衡。

    python3 /meritocracy/simulator.py --players 6 --games 10000
    python3 /meritocracy/simulator.py --players 4 --games 2000 --strategy greedy --seed 42 --json

和 Web 服务器共用同一套 game.py / rules.py，所以这里跑出来的数值就是真实对局数值。
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config, DEFAULT_CONFIG
from game import Game
from models import Card, DemotionKind, PlayerState, PromotionKind

# --------------------------------------------------------------------------
# 策略
# --------------------------------------------------------------------------

Strategy = Callable[[Game, PlayerState, list[Card], list[int], random.Random], tuple[Card, int | None]]


def _multi(single):
    """把"挑一张"包装成"挑 PICKS_PER_ROUND 张"。"""

    def wrapped(game, me, hand, others, rng):
        """hand 是 DealtCard 列表；老策略只关心牌型。"""
        remaining, picks = list(hand), []
        for _ in range(game.cfg.picks_per_round):
            if not remaining:
                break
            card, target = single(game, me, [d.card for d in remaining], others, rng)
            if card is None:
                break
            idx = next((i for i, d in enumerate(remaining) if d.card is card), None)
            if idx is None:
                break
            remaining.pop(idx)
            picks.append({"action": card.value, "target": target})
        return picks

    return wrapped


def strategy_random(
    game: Game, me: PlayerState, hand: list[Card], others: list[int], rng: random.Random
) -> tuple[Card, int | None]:
    card = rng.choice(hand)
    target = rng.choice(others) if card.needs_target and others else None
    if card.needs_target and target is None:
        # 理论上不会发生（至少 2 人），兜底换一张不需要目标的牌
        fallback = [c for c in hand if not c.needs_target]
        return (fallback[0] if fallback else Card.WORK), None
    return card, target


def strategy_greedy(
    game: Game, me: PlayerState, hand: list[Card], others: list[int], rng: random.Random
) -> tuple[Card, int | None]:
    """一个很朴素的启发式：奔着最近的晋升门槛去，顺手打压领先者。

    只是给平衡测试多一个参照系，不代表"最优解"。
    """
    cfg = game.cfg
    money_cost = cfg.money_cost(me.rank)
    merit_cost = cfg.merit_cost(me.rank)
    merit_gap = (merit_cost - me.merit) if merit_cost is not None else 10**9
    money_gap = (money_cost - me.money) if money_cost is not None else 10**9

    # 够门槛就先把晋升卡打出去（晋升现在必须用卡）
    if merit_gap <= 0:
        for c in (Card.PROMOTE_MERIT, Card.PROMOTE_ANY):
            if c in hand:
                return c, None
    if money_gap <= 0:
        for c in (Card.PROMOTE_MONEY, Card.PROMOTE_ANY):
            if c in hand:
                return c, None

    if merit_gap <= 0 or money_gap <= 0:
        # 已经够了但没摸到对应的晋升卡，这轮拿来干扰别人
        leader = max(
            (game.players[pid] for pid in others),
            key=lambda p: (p.rank, p.merit),
            default=None,
        )
        if leader is not None:
            if Card.ATTACK in hand:
                return Card.ATTACK, leader.id
            if Card.REPORT in hand:
                return Card.REPORT, leader.id

    if merit_gap <= money_gap and Card.WORK in hand:
        return Card.WORK, None
    if Card.CORRUPT in hand:
        return Card.CORRUPT, None
    if Card.WORK in hand:
        return Card.WORK, None
    return strategy_random(game, me, hand, others, rng)


STRATEGIES: dict[str, Strategy] = {
    "random": _multi(strategy_random),
    "greedy": _multi(strategy_greedy),
}

# --------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------


@dataclass
class Stats:
    games: int = 0
    rounds_total: int = 0
    end_round_hist: Counter = field(default_factory=Counter)
    president_wins: int = 0  # 有人当上国家主席而结束的对局数
    timeout_games: int = 0  # 打满 MAX_ROUNDS 才结束的对局数
    draws: int = 0  # 终局三项全同的平局

    final_money: list[int] = field(default_factory=list)
    final_merit: list[int] = field(default_factory=list)
    final_rank: list[int] = field(default_factory=list)

    rank_rounds: Counter = field(default_factory=Counter)  # rank -> 玩家-轮 数
    player_rounds: int = 0

    card_counts: Counter = field(default_factory=Counter)
    reports_received: int = 0
    reports_effective: int = 0
    demotions_minor: int = 0
    demotions_major: int = 0
    attacks_landed: int = 0
    merit_promotions_blocked: int = 0
    special_blocks: int = 0
    money_confiscated: int = 0
    money_to_reporters: int = 0

    promotions: Counter = field(default_factory=Counter)
    event_counts: Counter = field(default_factory=Counter)

    win_by_seat: Counter = field(default_factory=Counter)


def play_game(
    n_players: int, rng: random.Random, cfg: Config, strategy: Strategy, stats: Stats
) -> None:
    game = Game(game_id="sim", cfg=cfg, rng=rng)
    for i in range(n_players):
        game.add_player(f"玩家{i + 1}")
    game.start_game()

    while not game.is_over:
        for pid in sorted(game.players):
            me = game.players[pid]
            hand = game.hands[pid]
            others = [o for o in game.players if o != pid]
            picks = strategy(game, me, hand, others, rng) or []
            game.select_actions(pid, picks)
            if picks:
                game.lock_action(pid)

        game.reveal_event()
        outcome = game.resolve()

        stats.event_counts[outcome.event.name] += 1
        for pid, o in outcome.outcomes.items():
            for c in o.cards:
                stats.card_counts[c.value] += 1
            if o.reported:
                stats.reports_received += 1
            if o.report_effective:
                stats.reports_effective += 1
            if o.demotion is DemotionKind.MINOR:
                stats.demotions_minor += 1
            elif o.demotion is DemotionKind.MAJOR:
                stats.demotions_major += 1
            if o.attacked:
                stats.attacks_landed += 1
            stats.money_confiscated += o.money_confiscated
            stats.money_to_reporters += o.money_from_reports
            if o.merit_promotion_blocked:
                stats.merit_promotions_blocked += 1
            if o.promotion_blocked_by_attack_report:
                stats.special_blocks += 1
            if o.promotion is not PromotionKind.NONE:
                stats.promotions[o.promotion.value] += 1

        for p in game.ordered_players():
            stats.rank_rounds[p.rank] += 1
            stats.player_rounds += 1

        if not game.is_over:
            game.advance_round()

    stats.games += 1
    stats.rounds_total += game.round_number
    stats.end_round_hist[game.round_number] += 1
    if any(p.rank >= cfg.president_rank for p in game.players.values()):
        stats.president_wins += 1
    else:
        stats.timeout_games += 1
        if len(game.winners) > 1:
            stats.draws += 1
    for winner_id in game.winners:
        stats.win_by_seat[winner_id] += 1
    for p in game.ordered_players():
        stats.final_money.append(p.money)
        stats.final_merit.append(p.merit)
        stats.final_rank.append(p.rank)


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------


def build_report(stats: Stats, cfg: Config, n_players: int, elapsed: float) -> dict[str, Any]:
    g = max(1, stats.games)
    ppg = max(1, len(stats.final_money) // g)

    return {
        "games": stats.games,
        "players": n_players,
        "elapsed_seconds": round(elapsed, 2),
        "avg_rounds": round(stats.rounds_total / g, 3),
        "end_round_distribution": {
            str(r): stats.end_round_hist.get(r, 0) for r in range(1, cfg.max_rounds + 1)
        },
        "president_win_rate": round(stats.president_wins / g, 4),
        "full_length_rate": round(stats.timeout_games / g, 4),
        "draw_rate": round(stats.draws / g, 4),
        "final_money_avg": round(statistics.fmean(stats.final_money), 3) if stats.final_money else 0,
        "final_merit_avg": round(statistics.fmean(stats.final_merit), 3) if stats.final_merit else 0,
        "final_rank_avg": round(statistics.fmean(stats.final_rank), 3) if stats.final_rank else 0,
        "final_rank_distribution": {
            cfg.rank_name(r): sum(1 for x in stats.final_rank if x == r)
            for r in range(len(cfg.rank_names))
        },
        "avg_rounds_per_rank": {
            cfg.rank_name(r): round(stats.rank_rounds.get(r, 0) / (g * ppg), 3)
            for r in range(len(cfg.rank_names))
        },
        "card_counts": dict(stats.card_counts),
        "card_per_game": {k: round(v / g, 3) for k, v in stats.card_counts.items()},
        "reports_received": stats.reports_received,
        "reports_effective": stats.reports_effective,
        "demotions_minor": stats.demotions_minor,
        "demotions_major": stats.demotions_major,
        "attacks_landed": stats.attacks_landed,
        "merit_promotions_blocked": stats.merit_promotions_blocked,
        "attack_report_special_blocks": stats.special_blocks,
        "money_confiscated": stats.money_confiscated,
        "money_to_reporters": stats.money_to_reporters,
        "money_confiscated_per_game": round(stats.money_confiscated / g, 3),
        "money_to_reporters_per_game": round(stats.money_to_reporters / g, 3),
        "promotions": dict(stats.promotions),
        "event_counts": dict(stats.event_counts),
        "win_by_seat": {str(k): v for k, v in sorted(stats.win_by_seat.items())},
    }


def print_report(rep: dict[str, Any], cfg: Config) -> None:
    line = "-" * 56
    print(line)
    print(f"Meritocracy 模拟结果  {rep['players']} 人 x {rep['games']} 局"
          f"  （{rep['elapsed_seconds']}s）")
    print(line)
    print(f"平均游戏轮数          : {rep['avg_rounds']}")
    print(f"国家主席胜率          : {rep['president_win_rate']:.2%}")
    print(f"打满 {cfg.max_rounds} 轮的比例      : {rep['full_length_rate']:.2%}")
    print(f"终局平局比例          : {rep['draw_rate']:.2%}")
    print()
    print("第几轮结束的分布:")
    total = max(1, rep["games"])
    for r, n in rep["end_round_distribution"].items():
        bar = "#" * int(40 * n / total)
        print(f"  第 {r:>2} 轮 : {n:>7}  {n / total:6.2%} {bar}")
    print()
    print("终局玩家指标（全部玩家平均）:")
    print(f"  金钱 {rep['final_money_avg']}   政绩 {rep['final_merit_avg']}   官职 {rep['final_rank_avg']}")
    print("  终局官职分布:")
    for name, n in rep["final_rank_distribution"].items():
        print(f"    {name:<8}: {n}")
    print()
    print("每个官职平均停留轮数（每名玩家）:")
    for name, v in rep["avg_rounds_per_rank"].items():
        print(f"  {name:<8}: {v}")
    print()
    print("行动使用次数:")
    label = {"WORK": "WORK   埋头工作", "CORRUPT": "CORRUPT 中饱私囊",
             "REPORT": "REPORT  匿名举报", "ATTACK": "ATTACK  政治攻击"}
    for key in ("WORK", "CORRUPT", "REPORT", "ATTACK"):
        print(f"  {label[key]:<18}: {rep['card_counts'].get(key, 0):>9}"
              f"  （每局 {rep['card_per_game'].get(key, 0)}）")
    print()
    print("举报与攻击:")
    print(f"  被举报次数          : {rep['reports_received']}")
    print(f"  举报查实次数        : {rep['reports_effective']}")
    print(f"  降一级次数          : {rep['demotions_minor']}")
    print(f"  打回基层次数        : {rep['demotions_major']}")
    print(f"  攻击命中次数        : {rep['attacks_landed']}")
    print(f"  政绩晋升被阻次数    : {rep['merit_promotions_blocked']}")
    print(f"  攻击+举报双杀次数   : {rep['attack_report_special_blocks']}")
    print(f"  没收赃款总额        : {rep['money_confiscated']}"
          f"  （每局 {rep['money_confiscated_per_game']}）")
    print(f"  举报人分赃总额      : {rep['money_to_reporters']}"
          f"  （每局 {rep['money_to_reporters_per_game']}，差额为充公）")
    print()
    print("晋升方式:")
    for k, v in sorted(rep["promotions"].items()):
        print(f"  {k:<8}: {v}")
    print()
    print("各事件出现次数:")
    for k, v in sorted(rep["event_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:<16}: {v}")
    print()
    print("按座位统计的获胜次数（检查先手优势）:")
    for k, v in rep["win_by_seat"].items():
        print(f"  玩家{k}: {v}")
    print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Meritocracy 模拟器")
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="random")
    parser.add_argument("--seed", type=int, default=None, help="固定随机种子，便于复现")
    parser.add_argument("--json", action="store_true", help="输出 JSON 而不是文本报告")
    args = parser.parse_args(argv)

    cfg = DEFAULT_CONFIG
    if not (cfg.min_players <= args.players <= cfg.max_players):
        parser.error(f"players 必须在 {cfg.min_players}–{cfg.max_players} 之间")

    rng = random.Random(args.seed)
    strategy = STRATEGIES[args.strategy]
    stats = Stats()

    start = time.time()
    for _ in range(args.games):
        play_game(args.players, rng, cfg, strategy, stats)
    elapsed = time.time() - start

    report = build_report(stats, cfg, args.players, elapsed)
    report["strategy"] = args.strategy
    report["seed"] = args.seed

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
