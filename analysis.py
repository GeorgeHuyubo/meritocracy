"""Meritocracy 平衡性分析。

回答四个问题：

    1. 一局里领先者换多少次手？反转发生在什么时候？
    2. 一旦爬上去了还拦得住吗？（官职转移矩阵 / 领先锁定曲线）
    3. 四种行动各自值多少胜率？
    4. 哪些成型策略胜率异常高或低？（策略对抗赛）

用法：

    python3 analysis.py                          # 默认 6 人 20000 局
    python3 analysis.py --players 4 --games 5000
    python3 analysis.py --section leader         # 只看某一节
    python3 analysis.py --json

所有策略都只能看公开信息（别人的 rank / merit / tenure）和自己的私密状态，
和真人玩家看到的一样多——不会偷看别人的金钱或手牌。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ai
from config import Config, DEFAULT_CONFIG
from game import Game
from models import Card, DemotionKind, Origin, PlayerState, PromotionKind

Strategy = Callable[..., tuple[Card, int | None]]

# --------------------------------------------------------------------------
# 策略池（只读公开信息）
# --------------------------------------------------------------------------


def _others(game: Game, me_id: int) -> list[int]:
    return [pid for pid in game.players if pid != me_id]


def _public_leader(game: Game, exclude: int) -> PlayerState | None:
    """公开信息下的"最风光的人"：官职高、其次政绩高。"""
    pool = [game.players[pid] for pid in _others(game, exclude)]
    if not pool:
        return None
    return max(pool, key=lambda p: (p.rank, p.merit, -p.id))


def multi(single):
    """把"一次挑一张"的老策略包装成"一次挑 N 张"：反复调用并从手牌里扣掉已选的。"""

    def wrapped(game, me, hand, others, rng):
        """hand 现在是 DealtCard 列表；老策略只关心牌型，这里帮它们脱壳。"""
        remaining = list(hand)
        picks = []
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
            if card.is_promotion:
                # 一轮最多升一级，第二张晋升卡必然浪费。
                # 老策略是"无状态"的，第二次调用还会看到自己够门槛、
                # 于是再掏一张晋升卡出来——白扔一个回合。
                remaining = [d for d in remaining if not d.card.is_promotion]
        return picks

    return wrapped


def _fallback(hand: list[Card], *prefer: Card) -> Card | None:
    for card in prefer:
        if card in hand:
            return card
    return None


def s_random(game, me, hand, others, rng):
    card = rng.choice(hand)
    if card.needs_target:
        return card, rng.choice(others)
    return card, None


def _cash_in(game, me, hand):
    """够门槛就把晋升卡打出去——所有非随机策略都该会这一手。"""
    cfg = game.cfg
    tc, mc = cfg.merit_cost(me.rank), cfg.money_cost(me.rank)
    if tc is not None and me.merit >= tc:
        card = _fallback(hand, Card.PROMOTE_MERIT, Card.PROMOTE_ANY)
        if card:
            return card
    if mc is not None and me.money >= mc:
        card = _fallback(hand, Card.PROMOTE_MONEY, Card.PROMOTE_ANY)
        if card:
            return card
    return None


def s_worker(game, me, hand, others, rng):
    """只想干活，实在没牌才贪。够门槛就兑现。"""
    cashed = _cash_in(game, me, hand)
    if cashed:
        return cashed, None
    card = _fallback(hand, Card.WORK, Card.GRAFT, Card.CORRUPT)
    if card:
        return card, None
    return s_random(game, me, hand, others, rng)


def s_corrupt(game, me, hand, others, rng):
    """只想捞钱。够门槛就兑现。"""
    cashed = _cash_in(game, me, hand)
    if cashed:
        return cashed, None
    card = _fallback(hand, Card.CORRUPT, Card.GRAFT, Card.WORK)
    if card:
        return card, None
    return s_random(game, me, hand, others, rng)


def s_reporter(game, me, hand, others, rng):
    """有举报就举报领先者，否则干活。够门槛照样先兑现。"""
    cashed = _cash_in(game, me, hand)
    if cashed:
        return cashed, None
    if Card.REPORT in hand:
        leader = _public_leader(game, me.id)
        if leader is not None:
            return Card.REPORT, leader.id
    return s_worker(game, me, hand, others, rng)


def _closest_to_merit_line(game: Game, me_id: int, rng=None) -> PlayerState | None:
    """公开信息下最像"正在攒政绩"的人：离政绩晋升线最近的那个。

    在前两名里随机挑，避免所有攻击者都挤在同一个目标上把收益摊薄——
    真人也不会集体撞车。
    """
    pool = [game.players[pid] for pid in _others(game, me_id)]
    if not pool:
        return None
    def gap(p):
        tc = game.cfg.merit_cost(p.rank)
        return 10**6 if tc is None else max(0, tc - p.merit)
    ranked = sorted(pool, key=lambda p: (gap(p), -p.rank))
    if rng is not None and len(ranked) > 1:
        return rng.choice(ranked[:2])
    return ranked[0]


def s_attacker(game, me, hand, others, rng):
    """有攻击就打"最像在攒政绩的人"，否则干活。政绩是公开的，这是个可靠的读牌。"""
    cashed = _cash_in(game, me, hand)
    if cashed:
        return cashed, None
    if Card.ATTACK in hand:
        target = _closest_to_merit_line(game, me.id, rng)
        if target is not None:
            return Card.ATTACK, target.id
    return s_worker(game, me, hand, others, rng)


def s_saboteur(game, me, hand, others, rng):
    """专职搅局：能攻击就攻击，能举报就举报，都没有才干活。"""
    cashed = _cash_in(game, me, hand)
    if cashed:
        return cashed, None
    leader = _public_leader(game, me.id)
    if leader is not None:
        if Card.ATTACK in hand:
            return Card.ATTACK, leader.id
        if Card.REPORT in hand:
            return Card.REPORT, leader.id
    return s_worker(game, me, hand, others, rng)


def s_climber(game, me, hand, others, rng):
    """奔着最近的那条晋升线走，够了先兑现，兑现不了才去干扰领先者。"""
    cashed = _cash_in(game, me, hand)
    if cashed:
        return cashed, None
    cfg = game.cfg
    mc, tc = cfg.money_cost(me.rank), cfg.merit_cost(me.rank)
    merit_gap = (tc - me.merit) if tc is not None else 10**9
    money_gap = (mc - me.money) if mc is not None else 10**9

    if merit_gap <= 0 or money_gap <= 0:
        leader = _public_leader(game, me.id)
        if leader is not None:
            card = _fallback(hand, Card.ATTACK, Card.REPORT)
            if card:
                return card, leader.id
    if merit_gap <= money_gap and Card.WORK in hand:
        return Card.WORK, None
    card = _fallback(hand, Card.CORRUPT, Card.GRAFT, Card.WORK)
    if card:
        return card, None
    return s_random(game, me, hand, others, rng)


def s_safe_climber(game, me, hand, others, rng):
    """climber 的谨慎版：只走政绩线，完全不碰贪污，所以永远不怕被举报。"""
    cashed = _cash_in(game, me, hand)
    if cashed:
        return cashed, None
    if Card.WORK in hand:
        return Card.WORK, None
    leader = _public_leader(game, me.id)
    if leader is not None:
        card = _fallback(hand, Card.ATTACK, Card.REPORT)
        if card:
            return card, leader.id
    return s_random(game, me, hand, others, rng)


def s_builder(game, me, hand, others, rng):
    """只管建设，从不干扰。够门槛就兑现。用来演"第一名闷头努力"。"""
    cashed = _cash_in(game, me, hand)
    if cashed:
        return cashed, None
    card = _fallback(hand, Card.WORK, Card.GRAFT, Card.CORRUPT)
    return (card, None) if card else (None, None)


def s_challenger(game, me, hand, others, rng):
    """只干扰当前公开领先者；手里没干扰牌才建设。用来演"第二名去搞第一名"。"""
    cashed = _cash_in(game, me, hand)
    if cashed:
        return cashed, None
    leader = _public_leader(game, me.id)
    if leader is not None:
        card = _fallback(hand, Card.ATTACK, Card.REPORT)
        if card:
            return card, leader.id
    card = _fallback(hand, Card.WORK, Card.GRAFT, Card.CORRUPT)
    return (card, None) if card else (None, None)


def make_duelist(rival_id: int) -> Strategy:
    """死盯着某一个指定对手打。用来演"#1 和 #2 互相搞"。

    不能用 challenger 演这一幕：challenger 打的是**公开领先者**，
    两个 challenger 会一起去打那个埋头建设的渔翁，根本不会互相攻击。
    """

    def s(game, me, hand, others, rng):
        cashed = _cash_in(game, me, hand)
        if cashed:
            return cashed, None
        if rival_id in game.players and rival_id != me.id:
            card = _fallback(hand, Card.ATTACK, Card.REPORT)
            if card:
                return card, rival_id
        card = _fallback(hand, Card.WORK, Card.GRAFT, Card.CORRUPT)
        return (card, None) if card else (None, None)

    return multi(s)


def s_passive(game, me, hand, others, rng):
    """完全不作为，只吃工龄。用作"什么都不干能走多远"的基准线。"""
    return None, None


# ---- 思考型 AI（ai.py）：只吃公开 payload + 自己的私密状态 ----------------


def _smart_factory(**flags):
    """每局要给 AI 一个新的 AgentPool（记忆不能跨局）。"""

    def make(cfg: Config, rng: random.Random):
        pool = ai.AgentPool(cfg=cfg, rng=rng, **flags)

        def strategy(game, me, hand, others, _rng):
            # 和服务器一样走 ai.turn：该换牌就先换（含富二代的免费换牌）
            return ai.turn(game, me.id, pool)

        return strategy

    return make


SMART_FACTORIES: dict[str, Any] = {
    "smart": _smart_factory(),
    "smart_no_attack": _smart_factory(allow_attack=False),
    "smart_no_report": _smart_factory(allow_report=False),
    "smart_no_corrupt": _smart_factory(allow_corrupt=False),
    "smart_clean": _smart_factory(allow_attack=False, allow_report=False),
}

STRATEGIES: dict[str, Strategy] = {
    name: multi(fn)
    for name, fn in {
        "random": s_random,
        "worker": s_worker,
        "corrupt": s_corrupt,
        "reporter": s_reporter,
        "attacker": s_attacker,
        "saboteur": s_saboteur,
        "climber": s_climber,
        "safe_climber": s_safe_climber,
        "builder": s_builder,
        # 和 builder 行为完全一样，只是换个名字，好在报告里把"渔翁"单独读出来
        "fisherman": s_builder,
        "challenger": s_challenger,
        "passive": s_passive,
    }.items()
}


def build_assignment(names: list[str], cfg: Config, rng: random.Random) -> list[Strategy]:
    """把策略名列表变成本局可用的策略函数（思考型 AI 每局重建记忆）。"""
    return [
        SMART_FACTORIES[n](cfg, rng) if n in SMART_FACTORIES else STRATEGIES[n]
        for n in names
    ]

# --------------------------------------------------------------------------
# 带全程快照的对局
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Snap:
    rank: int
    money: int
    merit: int


@dataclass
class Record:
    n_players: int
    rounds: int
    winners: list[int]
    ended_by_president: bool
    # round -> {pid: Snap}，round 从 1 开始
    timeline: list[dict[int, Snap]] = field(default_factory=list)
    cards: dict[int, Counter] = field(default_factory=dict)
    demotions: dict[int, Counter] = field(default_factory=dict)
    reported: Counter = field(default_factory=Counter)
    attacked: Counter = field(default_factory=Counter)
    reward: Counter = field(default_factory=Counter)  # 分到的赃款
    reached_rank: dict[int, dict[int, int]] = field(default_factory=dict)  # pid -> rank -> 首次到达轮
    strategy_of: dict[int, str] = field(default_factory=dict)
    offered: Counter = field(default_factory=Counter)  # 某张牌在手上出现过几次
    chosen: Counter = field(default_factory=Counter)  # 其中被选中打出几次
    attack_gap: list[int] = field(default_factory=list)  # 每次攻击时，目标离政绩线还差多少
    attack_target_rank: Counter = field(default_factory=Counter)  # 攻击目标的官职分布
    rank_population: Counter = field(default_factory=Counter)  # 同期各官职的人数，用作基准
    # 死因分析用的逐人计数
    diag: dict[int, Counter] = field(default_factory=dict)
    # 每轮一条：事件 + 当轮的全局聚合，用来分析事件卡有没有存在感
    rounds_detail: list[dict[str, Any]] = field(default_factory=list)
    # 谁被瞄准过几次（攻击/举报的目标），用来算"被针对感"
    targeted: Counter = field(default_factory=Counter)
    # --- 举报专项 ---
    # 每打出一张举报牌记一条：命中没有、抄到多少、目标是什么状态、几个人一起举报
    report_shots: list[dict[str, Any]] = field(default_factory=list)
    # 每轮每个被举报者被几个真人同时举报（用来看撞车摊薄）
    report_crowding: Counter = field(default_factory=Counter)
    # 每个座位的出身（--origins melee 时才有）
    origin_of: dict[int, str | None] = field(default_factory=dict)


def play(n_players: int, rng: random.Random, cfg: Config, assign: list[Strategy],
         names: list[str] | None = None,
         origins: list[str | None] | None = None) -> Record:
    game = Game(game_id="analysis", cfg=cfg, rng=rng)
    for i in range(n_players):
        game.add_player(f"P{i + 1}")
    if origins:
        for pid, oid in zip(sorted(game.players), origins):
            game.players[pid].origin = Origin(oid) if oid else None

    rec = Record(n_players=n_players, rounds=0, winners=[], ended_by_president=False)
    rec.cards = {pid: Counter() for pid in game.players}
    rec.demotions = {pid: Counter() for pid in game.players}
    rec.reached_rank = {pid: {0: 0} for pid in game.players}
    rec.diag = {pid: Counter() for pid in game.players}
    if names:
        rec.strategy_of = {pid: names[i] for i, pid in enumerate(sorted(game.players))}
    rec.origin_of = {
        pid: (game.players[pid].origin.value if game.players[pid].origin else None)
        for pid in game.players
    }

    game.start_game()
    while not game.is_over:
        pending_reports: list[dict[str, Any]] = []
        for i, pid in enumerate(sorted(game.players)):
            me = game.players[pid]
            picks = assign[i](game, me, game.hands[pid], _others(game, pid), rng) or []
            hand = game.hands[pid]  # 策略里可能换过牌，统计要看最后打的那一手
            for c in {d.card for d in hand}:
                rec.offered[c.value] += 1

            # --- 死因分析：这一轮的手牌运气 ---
            dg = rec.diag[pid]
            dg["rounds"] += 1
            if not any(d.card.is_production for d in hand):
                dg["no_production_card"] += 1
            tc, mc = cfg.merit_cost(me.rank), cfg.money_cost(me.rank)
            eligible = (tc is not None and me.merit >= tc) or (mc is not None and me.money >= mc)
            if eligible:
                dg["rounds_eligible"] += 1
                usable = any(
                    (d.card.can_use_merit and tc is not None and me.merit >= tc)
                    or (d.card.can_use_money and mc is not None and me.money >= mc)
                    for d in hand
                )
                if not usable:
                    dg["eligible_but_no_card"] += 1
            for pick in picks:
                rec.chosen[pick["action"]] += 1
                if pick["action"] == "REPORT" and pick.get("target") is not None:
                    t = game.players[pick["target"]]
                    pending_reports.append({
                        "by": pid,
                        "target": pick["target"],
                        "target_rank": t.rank,
                        "target_merit": t.merit,
                    })
                if pick["action"] == "ATTACK" and pick.get("target") is not None:
                    t = game.players[pick["target"]]
                    tc = cfg.merit_cost(t.rank)
                    rec.attack_gap.append(10**6 if tc is None else tc - t.merit)
                    rec.attack_target_rank[t.rank] += 1
            for other in _others(game, pid):
                rec.rank_population[game.players[other].rank] += 1
            game.select_actions(pid, picks)
            if picks:
                game.lock_action(pid)
        game.force_lock_all()
        game.reveal_event()
        outcome = game.resolve()

        crowd = Counter(r["target"] for r in pending_reports)
        for tid, n in crowd.items():
            rec.report_crowding[n] += 1
        for shot in pending_reports:
            ot = outcome.outcomes[shot["target"]]
            mine = outcome.outcomes[shot["by"]]
            shot.update({
                "landed": ot.report_effective,
                "target_corrupted": ot.corrupt_amount > 0,
                "corrupt_amount": ot.corrupt_amount,
                "demotion": ot.demotion.value,
                "loot_pool": ot.money_confiscated,
                "co_reporters": crowd[shot["target"]],
                # 这一轮这个举报人一共分到多少（多份举报时按人头摊）
                "my_take_total": mine.money_from_reports,
                "my_reports": sum(1 for c in mine.cards if c is Card.REPORT),
            })
            rec.report_shots.append(shot)

        for pid, o in outcome.outcomes.items():
            dg = rec.diag[pid]
            for c in o.cards:
                rec.cards[pid][c.value] += 1
                if c.is_production:
                    dg["production_turns"] += 1
                elif c in (Card.REPORT, Card.ATTACK):
                    dg["interference_turns"] += 1
                else:
                    dg["promotion_turns"] += 1
            dg["merit_earned"] += o.merit_gained
            dg["money_earned"] += o.money_gained
            dg["merit_stolen_from_me"] += o.merit_stolen_by_attackers
            dg["money_confiscated"] += o.money_confiscated
            dg["merit_i_stole"] += o.merit_from_attacks
            dg["money_i_took"] += o.money_from_reports
            dg["tenure_wrecked"] += o.tenure_reset_by_attack
            if o.promotion_card_played and o.promotion is PromotionKind.NONE:
                dg["wasted_promotion_card"] += 1
            # "没拿到好处"不等于"白打"：举报把人打回基层、攻击拦下别人的晋升，
            # 本身就是战果，而这两种情况下行动者一分钱/一点政绩都拿不到。
            if Card.REPORT in o.cards and o.reports_landed <= 0:
                dg["wasted_report"] += 1
            if Card.ATTACK in o.cards and o.attacks_landed <= 0:
                dg["wasted_attack"] += 1
            if o.demotion is not DemotionKind.NONE:
                dg["demoted"] += 1
            if o.promotion is not PromotionKind.NONE:
                dg["promotions"] += 1
            if o.demotion is not DemotionKind.NONE:
                rec.demotions[pid][o.demotion.value] += 1
            if o.report_effective:
                rec.reported[pid] += 1
            if o.attacked:
                rec.attacked[pid] += 1
            rec.reward[pid] += o.money_from_reports

        rec.rounds_detail.append({
            "event": outcome.event.id,
            "event_name": outcome.event.name,
            "merit_gained": sum(o.merit_gained for o in outcome.outcomes.values()),
            "money_gained": sum(o.money_gained for o in outcome.outcomes.values()),
            "promotions": sum(
                1 for o in outcome.outcomes.values() if o.promotion is not PromotionKind.NONE
            ),
            "reports_landed": sum(1 for o in outcome.outcomes.values() if o.report_effective),
            "demotions": sum(
                1 for o in outcome.outcomes.values() if o.demotion is not DemotionKind.NONE
            ),
            "attacks_landed": sum(1 for o in outcome.outcomes.values() if o.attacked),
            "merit_destroyed": sum(o.attack_merit_loss for o in outcome.outcomes.values()),
            "money_confiscated": sum(o.money_confiscated for o in outcome.outcomes.values()),
        })
        for o in outcome.outcomes.values():
            for t in o.targets:
                if t is not None:
                    rec.targeted[t] += 1

        snap = {p.id: Snap(p.rank, p.money, p.merit) for p in game.ordered_players()}
        rec.timeline.append(snap)
        for pid, s in snap.items():
            rec.reached_rank[pid].setdefault(s.rank, game.round_number)

        if not game.is_over:
            game.advance_round()

    rec.rounds = game.round_number
    rec.winners = list(game.winners)
    rec.ended_by_president = any(
        p.rank >= cfg.president_rank for p in game.players.values()
    )
    return rec


# --------------------------------------------------------------------------
# 领先者定义
# --------------------------------------------------------------------------


def promo_progress(s: Snap, cfg: Config) -> float:
    """离下一级还差多远，0~1。已是最高级返回 1。"""
    mc, tc = cfg.money_cost(s.rank), cfg.merit_cost(s.rank)
    if mc is None or tc is None:
        return 1.0
    return max(min(s.money / mc, 1.0), min(s.merit / tc, 1.0))


def rank_leaders(snap: dict[int, Snap], cfg: Config) -> frozenset[int]:
    """官场领先：官职高者领先，同官职看谁离下一级更近。"""
    keyed = {pid: (s.rank, round(promo_progress(s, cfg), 6)) for pid, s in snap.items()}
    best = max(keyed.values())
    return frozenset(pid for pid, k in keyed.items() if k == best)


def score_leaders(snap: dict[int, Snap]) -> frozenset[int]:
    """终局计分领先：金钱 > 政绩 > 官职（就是第 10 轮的判定口径）。"""
    keyed = {pid: (s.money, s.merit, s.rank) for pid, s in snap.items()}
    best = max(keyed.values())
    return frozenset(pid for pid, k in keyed.items() if k == best)


# --------------------------------------------------------------------------
# 分析
# --------------------------------------------------------------------------


def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 2) if d else 0.0


def analyse_leadership(records: list[Record], cfg: Config) -> dict[str, Any]:
    changes_rank, changes_score = [], []
    holders_rank, holders_score = [], []
    win_given_leader_at: dict[int, list[int]] = defaultdict(list)  # 轮次 -> [是否夺冠]
    lead_kept: dict[int, list[int]] = defaultdict(list)  # 轮次 -> [下一轮是否还领先]
    winner_ever_led, last_round_flip, wire_to_wire = 0, 0, 0
    total = len(records)

    for rec in records:
        seq_r = [rank_leaders(s, cfg) for s in rec.timeline]
        seq_s = [score_leaders(s) for s in rec.timeline]
        changes_rank.append(sum(1 for i in range(1, len(seq_r)) if seq_r[i] != seq_r[i - 1]))
        changes_score.append(sum(1 for i in range(1, len(seq_s)) if seq_s[i] != seq_s[i - 1]))
        holders_rank.append(len(set().union(*seq_r)) if seq_r else 0)
        holders_score.append(len(set().union(*seq_s)) if seq_s else 0)

        win = set(rec.winners)
        # 用哪套领先口径？主席结束看官职，打满 10 轮看计分
        seq = seq_r if rec.ended_by_president else seq_s
        for i, leaders in enumerate(seq, start=1):
            for pid in leaders:
                win_given_leader_at[i].append(1 if pid in win else 0)
            if i < len(seq):
                for pid in leaders:
                    lead_kept[i].append(1 if pid in seq[i] else 0)

        if any(win & s for s in seq):
            winner_ever_led += 1
        if len(seq) >= 2 and not (win & seq[-2]):
            last_round_flip += 1
        if all(win & s for s in seq):
            wire_to_wire += 1

    n_winners = [len(r.winners) for r in records]
    return {
        "games": total,
        "president_pct": pct(sum(1 for r in records if r.ended_by_president), total),
        "avg_rounds": round(statistics.fmean(r.rounds for r in records), 2),
        "end_round_hist": dict(sorted(Counter(r.rounds for r in records).items())),
        "avg_winners_per_game": round(statistics.fmean(n_winners), 3),
        "co_winner_games_pct": pct(sum(1 for n in n_winners if n > 1), total),
        "co_winner_hist": dict(sorted(Counter(n_winners).items())),
        "lead_changes_rank_avg": round(statistics.fmean(changes_rank), 2),
        "lead_changes_score_avg": round(statistics.fmean(changes_score), 2),
        "lead_changes_rank_hist": dict(sorted(Counter(changes_rank).items())),
        "distinct_leaders_rank_avg": round(statistics.fmean(holders_rank), 2),
        "distinct_leaders_score_avg": round(statistics.fmean(holders_score), 2),
        "wire_to_wire_pct": pct(wire_to_wire, total),
        "winner_never_led_until_the_end_pct": pct(total - winner_ever_led, total),
        "last_round_flip_pct": pct(last_round_flip, total),
        "win_rate_given_leading_at_round": {
            str(r): pct(sum(v), len(v)) for r, v in sorted(win_given_leader_at.items())
        },
        "lead_retention_next_round": {
            str(r): pct(sum(v), len(v)) for r, v in sorted(lead_kept.items())
        },
    }


def analyse_rank_dynamics(records: list[Record], cfg: Config) -> dict[str, Any]:
    trans: dict[int, Counter] = defaultdict(Counter)  # 起始 rank -> {"up","same","down"}
    reached_and_won: dict[int, list[int]] = defaultdict(list)
    first_to_reach_won: dict[int, list[int]] = defaultdict(list)
    knocked_back: dict[int, list[int]] = defaultdict(list)  # 到过该 rank 的人里，后来掉下去的
    rounds_to_president: list[int] = []

    for rec in records:
        win = set(rec.winners)
        prev = {pid: 0 for pid in rec.cards}
        peak = {pid: 0 for pid in rec.cards}
        fell = {pid: False for pid in rec.cards}
        for snap in rec.timeline:
            for pid, s in snap.items():
                a, b = prev[pid], s.rank
                trans[a]["up" if b > a else "down" if b < a else "same"] += 1
                if b < peak[pid]:
                    fell[pid] = True
                peak[pid] = max(peak[pid], b)
                prev[pid] = b

        for pid in rec.cards:
            for r in range(1, cfg.president_rank + 1):
                if peak[pid] >= r:
                    reached_and_won[r].append(1 if pid in win else 0)
            if peak[pid] >= 3:
                knocked_back[3].append(1 if fell[pid] else 0)
            if peak[pid] >= 2:
                knocked_back[2].append(1 if fell[pid] else 0)

        for r in (2, 3):
            firsts = [
                (rec.reached_rank[pid][r], pid)
                for pid in rec.cards
                if r in rec.reached_rank[pid]
            ]
            if firsts:
                earliest = min(firsts)[0]
                tied = [pid for t, pid in firsts if t == earliest]
                for pid in tied:
                    first_to_reach_won[r].append(1 if pid in win else 0)

        if rec.ended_by_president:
            for pid in rec.winners:
                if 3 in rec.reached_rank[pid]:
                    rounds_to_president.append(rec.rounds - rec.reached_rank[pid][3])

    transition_pct = {}
    for r, counter in sorted(trans.items()):
        total = sum(counter.values())
        if total:
            transition_pct[cfg.rank_name(r)] = {
                k: pct(counter.get(k, 0), total) for k in ("up", "same", "down")
            }

    return {
        "rank_transition_pct": transition_pct,
        "win_rate_after_reaching": {
            cfg.rank_name(r): pct(sum(v), len(v)) for r, v in sorted(reached_and_won.items())
        },
        "win_rate_if_first_to_reach": {
            cfg.rank_name(r): pct(sum(v), len(v)) for r, v in sorted(first_to_reach_won.items())
        },
        "knocked_back_after_reaching_pct": {
            cfg.rank_name(r): pct(sum(v), len(v)) for r, v in sorted(knocked_back.items())
        },
        "rounds_from_provincial_to_president_avg": (
            round(statistics.fmean(rounds_to_president), 2) if rounds_to_president else None
        ),
    }


def analyse_actions(records: list[Record], cfg: Config) -> dict[str, Any]:
    """全员随机时，多打一张某种牌对胜率的影响。"""
    per_card_counts: dict[str, list[tuple[int, int]]] = defaultdict(list)  # 卡 -> [(次数, 是否赢)]
    baseline: list[int] = []
    victim_win: list[tuple[int, int]] = []
    attacked_win: list[tuple[int, int]] = []
    nuked_win: list[tuple[int, int]] = []

    for rec in records:
        win = set(rec.winners)
        for pid, counts in rec.cards.items():
            w = 1 if pid in win else 0
            baseline.append(w)
            for card in ("WORK", "CORRUPT", "GRAFT", "REPORT", "ATTACK"):
                per_card_counts[card].append((counts.get(card, 0), w))
            victim_win.append((rec.reported[pid], w))
            attacked_win.append((rec.attacked[pid], w))
            nuked_win.append((rec.demotions[pid].get("MAJOR", 0), w))

    base = statistics.fmean(baseline)

    def by_bucket(pairs: list[tuple[int, int]], edges: list[int]) -> dict[str, Any]:
        out = {}
        for i, lo in enumerate(edges):
            hi = edges[i + 1] - 1 if i + 1 < len(edges) else None
            sel = [w for c, w in pairs if c >= lo and (hi is None or c <= hi)]
            label = f"{lo}" if hi == lo else (f"{lo}+" if hi is None else f"{lo}-{hi}")
            if sel:
                out[label] = {"n": len(sel), "win_pct": pct(sum(sel), len(sel))}
        return out

    result: dict[str, Any] = {
        "baseline_win_pct": round(100 * base, 2),
        "by_card_usage": {},
        "mean_usage_winners_vs_others": {},
    }
    for card, pairs in per_card_counts.items():
        result["by_card_usage"][card] = by_bucket(pairs, [0, 1, 2, 3, 4, 5])
        w = [c for c, won in pairs if won]
        l = [c for c, won in pairs if not won]
        result["mean_usage_winners_vs_others"][card] = {
            "winners": round(statistics.fmean(w), 3) if w else 0,
            "others": round(statistics.fmean(l), 3) if l else 0,
            "delta": round((statistics.fmean(w) if w else 0) - (statistics.fmean(l) if l else 0), 3),
        }
    result["win_pct_by_times_reported"] = by_bucket(victim_win, [0, 1, 2, 3])
    result["win_pct_by_times_attacked"] = by_bucket(attacked_win, [0, 1, 2, 3])
    result["win_pct_by_times_nuked_to_base"] = by_bucket(nuked_win, [0, 1, 2])
    return result


def analyse_reports(records: list[Record], cfg: Config) -> dict[str, Any]:
    """举报专项：这张牌到底值不值一个回合。

    举报是全场唯一能惩罚贪污的手段，而贪污收益是隐藏信息——
    所以它的强弱不能只看"分到多少钱"，还要看命中率和摊薄程度。
    """
    shots = [s for rec in records for s in rec.report_shots]
    if not shots:
        return {"shots": 0}

    landed = [s for s in shots if s["landed"]]
    crowd = Counter()
    for rec in records:
        crowd.update(rec.report_crowding)

    # 每张举报牌摊到手的钱（同一轮打两张举报就把总收入摊开算）
    per_card_take = [
        s["my_take_total"] / max(1, s["my_reports"]) for s in shots
    ]
    landed_take = [
        s["my_take_total"] / max(1, s["my_reports"]) for s in landed
    ]

    # 机会成本的参照：一张 WORK 的期望点数（基层倍率下就是期望政绩）
    dist = cfg.work_card_distribution
    work_ev = sum(v * w for v, w in dist) / sum(w for _, w in dist)

    by_rank: dict[str, dict[str, Any]] = {}
    for r in sorted({s["target_rank"] for s in shots}):
        sel = [s for s in shots if s["target_rank"] == r]
        hit = [s for s in sel if s["landed"]]
        by_rank[cfg.rank_name(r)] = {
            "shots": len(sel),
            "share_pct": pct(len(sel), len(shots)),
            "hit_pct": pct(len(hit), len(sel)),
            "mean_take": round(statistics.fmean(
                [s["my_take_total"] / max(1, s["my_reports"]) for s in sel]), 2),
        }

    dem = Counter(s["demotion"] for s in landed)
    return {
        "shots": len(shots),
        "hit_pct": pct(len(landed), len(shots)),
        "target_was_corrupting_pct": pct(
            sum(1 for s in shots if s["target_corrupted"]), len(shots)
        ),
        "mean_take_per_card": round(statistics.fmean(per_card_take), 2),
        "mean_take_when_landed": round(
            statistics.fmean(landed_take) if landed_take else 0, 2
        ),
        "zero_take_despite_landing_pct": pct(
            sum(1 for s in landed if s["my_take_total"] <= 0), max(1, len(landed))
        ),
        "work_card_expected_merit": round(work_ev, 2),
        "demotion_mix": {k: pct(v, max(1, len(landed))) for k, v in dem.items()},
        "crowding": {
            f"{n} 人同时举报同一目标": pct(c, max(1, sum(crowd.values())))
            for n, c in sorted(crowd.items())
        },
        "mean_loot_pool_when_landed": round(
            statistics.fmean([s["loot_pool"] for s in landed]) if landed else 0, 2
        ),
        "by_target_rank": by_rank,
    }


def analyse_choices(records: list[Record], cfg: Config) -> dict[str, Any]:
    """一张牌在手上时，被真正打出来的比例——直接看出玩家觉得它值不值。"""
    offered, chosen = Counter(), Counter()
    gaps: list[int] = []
    for rec in records:
        offered.update(rec.offered)
        chosen.update(rec.chosen)
        gaps.extend(rec.attack_gap)

    pick_rate = {
        c: pct(chosen.get(c, 0), offered.get(c, 0))
        for c in ("WORK", "CORRUPT", "GRAFT", "REPORT", "ATTACK")
    }
    buckets = Counter()
    for g in gaps:
        if g <= 0:
            buckets["已经够线（必然挡下）"] += 1
        elif g <= 12:
            buckets["差 1-12（一手 WORK 就能过）"] += 1
        elif g <= 25:
            buckets["差 13-25"] += 1
        else:
            buckets["差 25 以上（基本白打）"] += 1
    tgt_rank, pop = Counter(), Counter()
    for rec in records:
        tgt_rank.update(rec.attack_target_rank)
        pop.update(rec.rank_population)
    n_att = sum(tgt_rank.values()) or 1
    n_pop = sum(pop.values()) or 1
    by_rank = {}
    for r in sorted(set(tgt_rank) | set(pop)):
        share = 100 * tgt_rank.get(r, 0) / n_att
        avail = 100 * pop.get(r, 0) / n_pop
        by_rank[cfg.rank_name(r)] = {
            "attacked_share_pct": round(share, 1),
            "population_share_pct": round(avail, 1),
            "focus": round(share / avail, 2) if avail > 0 else 0.0,
        }

    return {
        "attack_target_by_rank": by_rank,
        "pick_rate_when_in_hand": pick_rate,
        "offered": dict(offered),
        "chosen": dict(chosen),
        "attack_target_merit_gap": {k: pct(v, len(gaps)) for k, v in buckets.items()},
        "attacks": len(gaps),
    }


# --------------------------------------------------------------------------
# 死因分析：最后一名为什么输
# --------------------------------------------------------------------------

# (键, 中文说明, 是"别人干的"还是"自己选的"还是"运气")
# 以权谋私只是"特殊一点的贪污卡"，统计时和中饱私囊归成一类
CORRUPTION_CARDS = ("CORRUPT", "GRAFT")

CAUSES: list[tuple[str, str, str]] = [
    ("merit_stolen_from_me", "政绩被人抢走", "被针对"),
    ("money_confiscated", "赃款被举报没收", "被针对"),
    ("tenure_wrecked", "资历被搅黄", "被针对"),
    ("eligible_but_no_card", "够门槛却没摸到晋升卡", "运气"),
    ("no_production_card", "整手牌没有一张生产牌", "运气"),
    ("wasted_report", "举报扑空，白费回合", "自己选的"),
    ("wasted_attack", "攻击扑空，白费回合", "自己选的"),
    ("wasted_promotion_card", "打了晋升卡却没升成", "自己选的"),
    ("interference_turns", "把回合花在干扰而不是建设上", "自己选的"),
]


def analyse_events(records: list[Record], cfg: Config) -> dict[str, Any]:
    """事件卡到底有没有存在感：把每个事件发生的那些轮，和"风平浪静"的轮对比。"""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        for row in rec.rounds_detail:
            buckets[row["event_name"]].append(row)

    metrics = ("merit_gained", "money_gained", "promotions", "reports_landed",
               "demotions", "merit_destroyed", "money_confiscated")
    base_name = None
    for d in cfg.event_definitions:
        if not d.get("effects"):
            base_name = d["name"]
            break
    base = buckets.get(base_name, [])
    base_avg = {m: statistics.fmean([r[m] for r in base]) if base else 0.0 for m in metrics}

    total_rounds = sum(len(v) for v in buckets.values()) or 1
    out = {"baseline": base_name, "baseline_avg": {k: round(v, 2) for k, v in base_avg.items()},
           "events": {}}
    weights = {d["name"]: d["weight"] for d in cfg.event_definitions}
    wsum = sum(weights.values()) or 1
    for name, rows in buckets.items():
        avg = {m: statistics.fmean([r[m] for r in rows]) for m in metrics}
        delta = {m: round(avg[m] - base_avg[m], 2) for m in metrics}
        # "存在感" = 各指标相对基准的最大变化幅度
        impact = max(
            (abs(delta[m]) / base_avg[m] if base_avg[m] > 0.5 else abs(delta[m]))
            for m in metrics
        )
        out["events"][name] = {
            "rounds": len(rows),
            "share_pct": round(100 * len(rows) / total_rounds, 1),
            "expected_share_pct": round(100 * weights.get(name, 0) / wsum, 1),
            "avg": {k: round(v, 2) for k, v in avg.items()},
            "delta_vs_baseline": delta,
            "impact_score": round(impact, 2),
        }
    return out


def analyse_event_swings(records: list[Record], cfg: Config) -> dict[str, Any]:
    """事件卡能不能造成局面反转：看它发生的那一轮，领先者有没有易手。"""
    fired: dict[str, dict[str, float]] = defaultdict(lambda: {"rounds": 0, "flips": 0,
                                                              "rank_changes": 0})
    for rec in records:
        seq = [rank_leaders(s, cfg) for s in rec.timeline]
        for i, row in enumerate(rec.rounds_detail):
            st = fired[row["event_name"]]
            st["rounds"] += 1
            if i > 0 and seq[i] != seq[i - 1]:
                st["flips"] += 1
            st["rank_changes"] += row["promotions"] + row["demotions"]
    out = {}
    for name, st in fired.items():
        n = max(1, st["rounds"])
        out[name] = {
            "rounds": int(st["rounds"]),
            "leader_flip_pct": round(100 * st["flips"] / n, 1),
            "rank_changes_per_round": round(st["rank_changes"] / n, 2),
        }
    return out


def analyse_origin_results(records: list[Record], cfg: Config) -> dict[str, Any]:
    """--origins melee：同一批对局里每个出身的战绩。

    胜率按"并列冠军 1/人数"折算；垫底率看最后一名；被瞄准看攻击/举报落在谁身上。
    """
    seats: Counter = Counter()
    wins: defaultdict = defaultdict(float)
    pres_wins: defaultdict = defaultdict(float)
    last: Counter = Counter()
    place_sum: Counter = Counter()
    promos: Counter = Counter()
    targeted: Counter = Counter()
    for rec in records:
        if not rec.origin_of:
            continue
        order = final_standing(rec, cfg)
        for place, pid in enumerate(order, start=1):
            oid = rec.origin_of.get(pid) or "NONE"
            seats[oid] += 1
            place_sum[oid] += place
            promos[oid] += sum(1 for r in rec.reached_rank.get(pid, {}) if r > 0)
            targeted[oid] += rec.targeted.get(pid, 0)
            if pid in rec.winners:
                wins[oid] += 1.0 / len(rec.winners)
                if rec.ended_by_president:
                    pres_wins[oid] += 1.0 / len(rec.winners)
        last[rec.origin_of.get(order[-1]) or "NONE"] += 1
    out: dict[str, Any] = {}
    for oid in sorted(seats, key=lambda o: -wins[o] / seats[o]):
        n = seats[oid]
        p = wins[oid] / n
        out[oid] = {
            "seats": n,
            "win_pct": round(100 * p, 2),
            "ci_half_width": round(100 * 1.96 * math.sqrt(p * (1 - p) / n), 2),
            "president_win_pct": round(100 * pres_wins[oid] / n, 2),
            "last_place_pct": round(100 * last[oid] / n, 2),
            "avg_place": round(place_sum[oid] / n, 2),
            "avg_promotions": round(promos[oid] / n, 2),
            "targeted_per_game": round(targeted[oid] / n, 2),
        }
    return out


def final_standing(rec: Record, cfg: Config) -> list[int]:
    """按真正的胜负口径排名（FINAL_RANKING_KEYS，默认 官职 > 金钱 > 政绩），赢家排最前。"""
    snap = rec.timeline[-1]
    keys = cfg.final_ranking_keys
    ordered = sorted(
        snap, key=lambda pid: tuple(getattr(snap[pid], k) for k in keys), reverse=True
    )
    winners = [pid for pid in ordered if pid in rec.winners]
    return winners + [pid for pid in ordered if pid not in rec.winners]


def diagnose_loser(rec: Record, cfg: Config) -> dict[str, Any]:
    """分析最后一名为什么输：把他和"其他人平均"的差距摊到各个成因上。"""
    order = final_standing(rec, cfg)
    loser, winner = order[-1], order[0]
    others = [pid for pid in rec.diag if pid != loser]

    gaps = []
    for key, label, kind in CAUSES:
        mine = rec.diag[loser].get(key, 0)
        avg = statistics.fmean([rec.diag[p].get(key, 0) for p in others]) if others else 0.0
        if mine > avg:  # 只列出"他比别人吃亏/浪费得更多"的项
            gaps.append({"key": key, "label": label, "kind": kind,
                         "mine": mine, "others_avg": round(avg, 2),
                         "excess": round(mine - avg, 2)})
    gaps.sort(key=lambda g: -g["excess"])

    snap = rec.timeline[-1]
    return {
        "loser": loser,
        "winner": winner,
        "loser_strategy": rec.strategy_of.get(loser),
        "rank": snap[loser].rank,
        "winner_rank": snap[winner].rank,
        "promotions": rec.diag[loser].get("promotions", 0),
        "avg_promotions": round(
            statistics.fmean([rec.diag[p].get("promotions", 0) for p in others]), 2
        ) if others else 0,
        "top_causes": gaps[:4],
        "blame": _blame_split(gaps),
    }


def analyse_loser_experience(records: list[Record], cfg: Config) -> dict[str, Any]:
    """垫底的人这一局过得怎么样：有没有过高光时刻、有没有被集火、有没有得玩。"""
    ever_led, rounds_at_base, agency, targeted_ratio = [], [], [], []
    promos, hit_taken, never_promoted = [], [], 0
    for rec in records:
        order = final_standing(rec, cfg)
        loser = order[-1]
        n = len(order)
        seq = [rank_leaders(s, cfg) for s in rec.timeline]
        ever_led.append(1 if any(loser in x for x in seq) else 0)
        rounds_at_base.append(sum(1 for s in rec.timeline if s[loser].rank == 0))
        dg = rec.diag[loser]
        used = dg.get("production_turns", 0) + dg.get("promotions", 0)
        total = dg.get("rounds", 1) * cfg.picks_per_round
        agency.append(used / total if total else 0)      # 有多少回合真的推进了自己
        fair = sum(rec.targeted.values()) / n if n else 0
        targeted_ratio.append(rec.targeted.get(loser, 0) / fair if fair > 0 else 1.0)
        promos.append(dg.get("promotions", 0))
        hit_taken.append(dg.get("demoted", 0))
        if dg.get("promotions", 0) == 0:
            never_promoted += 1
    g = max(1, len(records))
    return {
        "ever_held_the_lead_pct": round(100 * sum(ever_led) / g, 1),
        "avg_rounds_stuck_at_base_rank": round(statistics.fmean(rounds_at_base), 2),
        "avg_promotions": round(statistics.fmean(promos), 2),
        "never_promoted_pct": round(100 * never_promoted / g, 1),
        "productive_turn_share": round(100 * statistics.fmean(agency), 1),
        "targeted_vs_fair_share": round(statistics.fmean(targeted_ratio), 2),
        "avg_times_demoted": round(statistics.fmean(hit_taken), 2),
    }


def _blame_split(gaps: list[dict]) -> dict[str, float]:
    """把成因按"被针对 / 运气 / 自己选的"三类归口，给出百分比。"""
    # 不同成因单位不同（政绩点数 vs 轮数），先按类内归一再合并权重
    weights = {"政绩被人抢走": 0.05, "赃款被举报没收": 0.03, "资历被搅黄": 0.3,
               "够门槛却没摸到晋升卡": 1.0, "整手牌没有一张生产牌": 1.0,
               "举报扑空，白费回合": 1.0, "攻击扑空，白费回合": 1.0,
               "打了晋升卡却没升成": 1.0, "把回合花在干扰而不是建设上": 0.5}
    tally: dict[str, float] = defaultdict(float)
    for g in gaps:
        tally[g["kind"]] += g["excess"] * weights.get(g["label"], 1.0)
    total = sum(tally.values())
    if total <= 0:
        return {}
    return {k: round(100 * v / total, 1) for k, v in sorted(tally.items(), key=lambda kv: -kv[1])}


def narrate(d: dict[str, Any], cfg: Config) -> str:
    """把死因分析写成一句人话。"""
    head = (
        f"P{d['loser']} 垫底（{cfg.rank_name(d['rank'])}，"
        f"冠军 P{d['winner']} 是{cfg.rank_name(d['winner_rank'])}）"
        f"，全场只晋升 {d['promotions']} 次，别人平均 {d['avg_promotions']} 次。"
    )
    if not d["top_causes"]:
        return head + " 各项指标都不落后，纯粹是终局比大小输了。"
    bits = [f"{c['label']}（{c['mine']} vs 别人 {c['others_avg']}）" for c in d["top_causes"][:3]]
    blame = "、".join(f"{k} {v}%" for k, v in d["blame"].items())
    return head + " 主要吃亏在：" + "；".join(bits) + f"。归口：{blame}。"


def tournament(
    n_players: int, games: int, cfg: Config, seed: int, names: list[str]
) -> dict[str, Any]:
    """策略对抗赛：每局把策略轮换座位，消掉先后手偏差。"""
    rng = random.Random(seed)
    wins: Counter = Counter()
    share: dict[str, float] = defaultdict(float)  # 并列冠军按 1/人数 计
    seats: Counter = Counter()
    ranks: dict[str, list[int]] = defaultdict(list)
    presidents: Counter = Counter()

    for g in range(games):
        order = [names[(g + i) % len(names)] for i in range(n_players)]
        assign = build_assignment(order, cfg, rng)
        rec = play(n_players, rng, cfg, assign, names=order)
        win = set(rec.winners)
        for i, pid in enumerate(sorted(rec.cards)):
            seats[order[i]] += 1
            if pid in win:
                wins[order[i]] += 1
                share[order[i]] += 1.0 / len(win)
                if rec.ended_by_president:
                    presidents[order[i]] += 1
            ranks[order[i]].append(rec.timeline[-1][pid].rank)

    ordered = sorted(seats, key=lambda x: -share[x])
    return {
        "games": games,
        "seats_per_strategy": dict(seats),
        # 并列冠军也算赢，所以各家相加会超过 100%
        "win_pct": {n: pct(wins.get(n, 0), seats[n]) for n in ordered},
        # 并列时按 1/并列人数 折算，各家相加 = 100%
        "win_share_pct": {n: round(100 * share[n] / seats[n], 2) for n in ordered},
        "president_pct": {n: pct(presidents.get(n, 0), seats[n]) for n in ordered},
        "final_rank_avg": {n: round(statistics.fmean(ranks[n]), 2) for n in ordered},
        "fair_share_pct": round(100 / n_players, 2),
    }


def analyse_triangle(
    n_players: int, games: int, cfg: Config, rng: random.Random
) -> dict[str, Any]:
    """直接验证「鹬蚌相争，渔翁得利」这个目标动态成不成立。

    不看胜率，看**剧本化角色的终局名次**——这才是那句话的可执行版本：

      场景一  1 个 builder（闷头建设）+ 1 个 challenger（专搞领先者）+ 其余围观
              判据：challenger 的平均名次要**好于** builder
              （「第一名闷头努力，第二名去搞他 -> 第二名反超」）

      场景二  2 个 challenger 互搞 + 1 个 fisherman（闷头建设）+ 其余围观
              判据：fisherman 要**同时好于**两个 challenger
              （「他俩互相搞 -> 第三名得利」）

    名次用 1 = 第一名，越小越好。
    """

    def run(roles: list[str]) -> dict[str, Any]:
        places: dict[str, list[int]] = defaultdict(list)
        wins: Counter = Counter()
        for g in range(games):
            # 轮换座位，抵消先后手
            order = roles[g % len(roles):] + roles[: g % len(roles)]
            assign = build_assignment(order, cfg, rng)
            rec = play(n_players, rng, cfg, assign, names=order)
            # 复用真正的胜负口径（赢家排最前），别自己另搞一套
            placing = {pid: i + 1 for i, pid in enumerate(final_standing(rec, cfg))}
            win = set(rec.winners)
            for i, pid in enumerate(sorted(rec.cards)):
                places[order[i]].append(placing[pid])
                if pid in win:
                    wins[order[i]] += 1
        seats = {r: len(v) for r, v in places.items()}
        return {
            "avg_place": {
                r: round(statistics.fmean(v), 3) for r, v in sorted(places.items())
            },
            "win_pct": {
                r: pct(wins.get(r, 0), seats[r]) for r in sorted(places)
            },
            "seats": seats,
        }

    def run_duel() -> dict[str, Any]:
        """场景二：两个决斗者死盯着对方打，渔翁在旁边闷头建设。"""
        labels = ["duelist", "duelist", "fisherman"] + [
            "passive"
        ] * max(0, n_players - 3)
        places: dict[str, list[int]] = defaultdict(list)
        wins: Counter = Counter()
        for g in range(games):
            shift = g % n_players
            order = labels[-shift:] + labels[:-shift] if shift else list(labels)
            seats = sorted(range(1, n_players + 1))
            duel_seats = [seats[i] for i, r in enumerate(order) if r == "duelist"]
            assign = []
            for i, role in enumerate(order):
                if role == "duelist":
                    rival = [x for x in duel_seats if x != seats[i]][0]
                    assign.append(make_duelist(rival))
                else:
                    assign.append(STRATEGIES[role])
            rec = play(n_players, rng, cfg, assign, names=order)
            placing = {pid: i + 1 for i, pid in enumerate(final_standing(rec, cfg))}
            win = set(rec.winners)
            for i, pid in enumerate(sorted(rec.cards)):
                places[order[i]].append(placing[pid])
                if pid in win:
                    wins[order[i]] += 1
        seats_n = {r: len(v) for r, v in places.items()}
        return {
            "avg_place": {
                r: round(statistics.fmean(v), 3) for r, v in sorted(places.items())
            },
            "win_pct": {r: pct(wins.get(r, 0), seats_n[r]) for r in sorted(places)},
            "seats": seats_n,
        }

    # 围观席用 passive（什么都不做），免得干扰这场对照
    one = run(["builder", "challenger"] + ["passive"] * max(0, n_players - 2))
    two = run_duel()
    return {"games": games, "scenario_one": one, "scenario_two": two}


ABLATION_CARDS = (
    ("allow_report", "举报"),
    ("allow_attack", "攻击"),
    ("allow_corrupt", "贪污"),
)


def _ablation_once(
    n_players: int, games: int, cfg: Config, rng: random.Random,
    flag: str, muted_seats: int,
) -> dict[str, Any]:
    """跑一档消融：muted_seats 个人被禁用这张牌，其余正常，逐局轮换座位。"""
    wins = {"normal": 0.0, "muted": 0.0}
    seats = {"normal": 0, "muted": 0}
    for g in range(games):
        game = Game(game_id="ablation", cfg=cfg, rng=rng)
        for i in range(n_players):
            game.add_player(f"P{i + 1}")
        role = {
            pid: ("muted" if (i + g) % n_players < muted_seats else "normal")
            for i, pid in enumerate(sorted(game.players))
        }
        pools = {
            "normal": ai.AgentPool(cfg=cfg, rng=rng),
            "muted": ai.AgentPool(cfg=cfg, rng=rng, **{flag: False}),
        }
        game.start_game()
        while not game.is_over:
            for pid in sorted(game.players):
                game.select_actions(pid, ai.turn(game, pid, pools[role[pid]]))
                game.lock_action(pid)
            game.reveal_event()
            game.resolve()
            if not game.is_over:
                game.advance_round()
        win = set(game.winners)
        for pid, r in role.items():
            seats[r] += 1
            if pid in win:
                wins[r] += 1.0 / len(win)

    n_m, n_n = max(1, seats["muted"]), max(1, seats["normal"])
    p_m, p_n = wins["muted"] / n_m, wins["normal"] / n_n
    # 两个比例之差的 95% 区间。没有它就会把噪声当结论——
    # 同一配置换个种子能测出 -4.24 和 -2.38，这种差别其实分不出来。
    se = math.sqrt(p_m * (1 - p_m) / n_m + p_n * (1 - p_n) / n_n)
    half = 1.96 * se * 100
    delta = (p_m - p_n) * 100
    return {
        "muted_win_pct": round(p_m * 100, 2),
        "normal_win_pct": round(p_n * 100, 2),
        "delta": round(delta, 2),
        "ci_half_width": round(half, 2),
        "muted_seats": muted_seats,
        "samples_muted": seats["muted"],
    }


def analyse_ablation(
    n_players: int, games: int, cfg: Config, rng: random.Random,
    muted_seats: int = 1, sweep: bool = False,
) -> dict[str, Any]:
    """消融测试：一张牌到底值不值得用。

    禁掉某个人的这张牌，其余一切相同，看他和正常人的胜率差。
    改的是"能不能用"，所以差值就是这张牌的因果贡献，比
    "冠军用了几次"之类的相关性指标可靠得多。

        差 > 0  禁用的人反而赢得多 -> 这张牌是**负收益**
        差 ≈ 0  定价合理
        差 < 0  这张牌很强

    **默认只禁 1 席**，因为那才是"别人都照常玩、只有我少一个选项"这个
    真正该问的问题。禁半桌会把牌桌推离均衡点，对"人多才安全"的牌
    （比如贪污：一起贪的人越多每人越安全）会系统性地低估它——
    实测同一张贪污牌，禁 1 席是 -0.80，禁 3 席就变成 +3.05。

    sweep=True 时对 k=1..n-1 各跑一遍，输出整条曲线：
    单调上升 = 这张牌人多才安全；基本持平 = 和人数无关。
    """
    out: dict[str, Any] = {"games": games, "muted_seats": muted_seats, "cards": {}}
    for flag, label in ABLATION_CARDS:
        out["cards"][label] = _ablation_once(
            n_players, games, cfg, rng, flag, muted_seats
        )
    if sweep:
        out["sweep"] = {
            label: [
                _ablation_once(n_players, games, cfg, rng, flag, k)["delta"]
                for k in range(1, n_players)
            ]
            for flag, label in ABLATION_CARDS
        }
    return out


def analyse_origins(
    n_players: int, games: int, cfg: Config, rng: random.Random
) -> dict[str, Any]:
    """出身卡的强弱。

    **强制随机分配，不让 AI 挑。** 让人或 AI 挑身份的话，弱身份被选得少，
    它的胜率反而好看（选择偏差）。所以这里照 `_ablation_once` 那套：
    指定座位、逐局轮换，谁拿什么完全由座位决定。

    出四组数，**按这个顺序看**：
      1. 触发率 —— 技能一局实际生效几次。触发率接近 0 的，
         它的胜率差一定是噪声，不要去解读
      2. 单身份对照 —— 1 席拿身份 X、其余 5 席无出身，差值就是它值多少个点
      3. 混战 —— 6 人各一个身份，看胜率份额
      4. 权力曲线 —— 平均夺冠轮次。富二代前重、贫农后重，这个数直接看得出来
    """
    origin_ids = cfg.origin_ids()
    out: dict[str, Any] = {"games": games, "solo": {}, "melee": {}, "fires": {}}
    if not origin_ids:
        return out
    # 每张出身用自己的 rng 流，而不是六张共用一条。共用的话改其中一张
    # 会把排在它后面那几张抽到的局全换掉，于是"我只动了红二代，怎么富二代
    # 也差了 4 个点"——那是流位移，不是效应。踩过一次。
    base = rng.randrange(1 << 30)
    streams = {
        oid: random.Random(base + 1000 * i) for i, oid in enumerate(origin_ids)
    }
    brains = {
        oid: random.Random(base + 1000 * i + 7) for i, oid in enumerate(origin_ids)
    }

    def play_once(
        assign: dict[int, str | None], stream: random.Random, brain: random.Random
    ) -> tuple[set[int], int, dict[int, int]]:
        # **发牌和 AI 用两条独立的流。** 共用一条的话，任何让 AI 多抽或少抽
        # 一个随机数的代码改动（哪怕只是打分差一点、平局判定走了另一支）
        # 都会把后面所有的发牌整个错位 —— 等于每次改代码都重洗一副牌。
        # 踩过：只改了官二代的折扣，红二代的数字从 +4.80 跳到 +8.10。
        # 分开之后，同一个种子发的牌永远一样，差异才真的来自规则本身。
        game = Game(game_id="origins", cfg=cfg, rng=stream)
        for i in range(n_players):
            game.add_player(f"P{i + 1}")
        for pid, oid in assign.items():
            if oid:
                game.players[pid].origin = Origin(oid)
        pool = ai.AgentPool(cfg=cfg, rng=brain)
        fires = {pid: 0 for pid in assign}
        game.start_game()
        while not game.is_over:
            ranks_before = {p.id: p.rank for p in game.players.values()}
            for pid in sorted(game.players):
                game.select_actions(pid, ai.turn(game, pid, pool))
                game.lock_action(pid)
            game.reveal_event()
            res = game.resolve()
            for pid, o in res.outcomes.items():
                fires[pid] += _origin_fired(
                    assign.get(pid), o, game.players[pid], ranks_before[pid]
                )
            if not game.is_over:
                game.advance_round()
        return set(game.winners), game.round_number, fires

    # ---- 1+2. 单身份对照：1 席拿 X，其余无出身，逐局轮换座位 ----
    for oid in origin_ids:
        wins = {"with": 0.0, "without": 0.0}
        seats = {"with": 0, "without": 0}
        fired = won_round = won_games = 0
        for g in range(games):
            pids = list(range(1, n_players + 1))
            lucky = pids[g % n_players]
            assign = {pid: (oid if pid == lucky else None) for pid in pids}
            winners, rounds, fires = play_once(assign, streams[oid], brains[oid])
            fired += fires[lucky]
            for pid in pids:
                key = "with" if pid == lucky else "without"
                seats[key] += 1
                if pid in winners:
                    wins[key] += 1.0 / len(winners)
            if lucky in winners:
                won_round += rounds
                won_games += 1
        n_w, n_o = max(1, seats["with"]), max(1, seats["without"])
        p_w, p_o = wins["with"] / n_w, wins["without"] / n_o
        se = math.sqrt(p_w * (1 - p_w) / n_w + p_o * (1 - p_o) / n_o)
        out["solo"][oid] = {
            "win_pct": round(p_w * 100, 2),
            "others_pct": round(p_o * 100, 2),
            "delta": round((p_w - p_o) * 100, 2),
            "ci_half_width": round(1.96 * se * 100, 2),
            "fires_per_game": round(fired / max(1, games), 2),
            "avg_win_round": round(won_round / won_games, 2) if won_games else None,
        }

    # ---- 3. 混战：6 人各一个身份，随机排列 ----
    melee_wins = {oid: 0.0 for oid in origin_ids}
    melee_seats = {oid: 0 for oid in origin_ids}
    melee_rng = random.Random(base - 1)
    melee_brain = random.Random(base - 2)
    for _ in range(games):
        order = origin_ids[:]
        melee_rng.shuffle(order)
        assign = {i + 1: order[i % len(order)] for i in range(n_players)}
        winners, _, _ = play_once(assign, melee_rng, melee_brain)
        for pid, oid in assign.items():
            melee_seats[oid] += 1
            if pid in winners:
                melee_wins[oid] += 1.0 / len(winners)
    out["melee"] = {
        oid: round(100 * melee_wins[oid] / max(1, melee_seats[oid]), 2)
        for oid in origin_ids
    }
    return out


def _origin_fired(origin_id, outcome, player, rank_before) -> int:
    """这一轮这个人的出身技能到底生效了没有。

    先看这个再看胜率：技能压根没触发的话，胜率差一定是噪声。
    比如红二代「开后门」要求同时够两级，很可能整局都摸不到那个条件。
    """
    if not origin_id:
        return 0
    if origin_id == "RICH":
        # 开局那笔钱一次性、看不出来；这里数的是"每轮一次免费换牌"用了没有
        return 1 if outcome.redraw_count else 0
    if origin_id == "ACCOUNTANT":
        return 1 if outcome.laundered and outcome.report_effective else 0
    if origin_id == "RED":
        return 1 if (outcome.origin_shielded_demotion or outcome.family_promotion) else 0
    if origin_id == "PEASANT":
        # 挨了打、而且这一轮确实在走政绩升职 —— 换成别人就被拦下了
        return 1 if (outcome.attacked and outcome.promotion.value == "MERIT") else 0
    if origin_id == "GRINDER":
        return 1 if outcome.overtime_merit else 0  # 真的连干两张、加了班才算
    if origin_id == "OFFICIAL":
        return 1 if outcome.promotion_merit_cost else 0
    return 0


def analyse_funnel(
    n_players: int, games: int, cfg: Config, rng: random.Random
) -> dict[str, Any]:
    """金钱路线的逐级漏斗：贪到手的钱，最后有多少真的换成了官职。

    为什么要这把尺子：胜率消融是"一局一个样本"，3000 局的区间半宽还有 ±1.45，
    而很多规则改动的真实效应就在 ±1 以内，永远测不出来。
    漏斗是"一个事件一个样本"，几万个样本，两个配置之间的差别是确定的。

    同时算政绩路线的同口径存活率作为对照 —— 单看一个数字没有意义，
    要看两条路线**相比之下**谁更难走。
    """
    c: Counter = Counter()
    for g in range(games):
        game = Game(game_id="funnel", cfg=cfg, rng=rng)
        for i in range(n_players):
            game.add_player(f"P{i + 1}")
        pool = ai.AgentPool(cfg=cfg, rng=rng)
        game.start_game()
        while not game.is_over:
            for pid in sorted(game.players):
                game.select_actions(pid, ai.turn(game, pid, pool))
                game.lock_action(pid)
            game.reveal_event()
            outcome = game.resolve()
            for o in outcome.outcomes.values():
                # --- 金钱路线：贪 -> 扛过举报 -> 拿去买官 -> 官升成 ---
                if o.corrupt_amount > 0:
                    c["corrupt_plays"] += 1
                    c["corrupt_gross"] += o.corrupt_amount
                    c["corrupt_kept"] += o.net_corrupt_gain
                    if o.money_confiscated or o.hush_money_paid:
                        c["corrupt_seized"] += 1
                    else:
                        c["corrupt_survived"] += 1
                # 想拿钱买官的回合（不管成没成）
                if o.pending_bribe > 0 or o.promotion is PromotionKind.MONEY:
                    c["bribe_attempts"] += 1
                    if o.promotion is PromotionKind.MONEY:
                        c["bribe_succeeded"] += 1
                    else:
                        c["bribe_lost_events"] += 1
                        c["bribe_burned"] += o.bribe_lost
                # --- 政绩路线对照：干活 -> 扛过抢功 -> 政绩升职成 ---
                if o.merit_gained > 0:
                    c["work_plays"] += 1
                    c["work_gross"] += o.merit_gained
                    c["work_kept"] += max(
                        0, o.merit_gained - o.merit_stolen_by_attackers
                    )
                    if o.merit_stolen_by_attackers:
                        c["work_robbed"] += 1
                    else:
                        c["work_survived"] += 1
                if o.tried_merit_promotion:
                    c["merit_promo_attempts"] += 1
                    if o.promotion is PromotionKind.MERIT:
                        c["merit_promo_succeeded"] += 1
            if not game.is_over:
                game.advance_round()

    def rate(a: str, b: str) -> float:
        return round(100 * c[a] / c[b], 2) if c[b] else 0.0

    money_end_to_end = (
        (c["corrupt_survived"] / c["corrupt_plays"]) *
        (c["bribe_succeeded"] / c["bribe_attempts"])
        if c["corrupt_plays"] and c["bribe_attempts"] else 0.0
    )
    merit_end_to_end = (
        (c["work_survived"] / c["work_plays"]) *
        (c["merit_promo_succeeded"] / c["merit_promo_attempts"])
        if c["work_plays"] and c["merit_promo_attempts"] else 0.0
    )
    return {
        "games": games,
        "money": {
            "plays": c["corrupt_plays"],
            "survived_pct": rate("corrupt_survived", "corrupt_plays"),
            "kept_pct": round(100 * c["corrupt_kept"] / max(1, c["corrupt_gross"]), 2),
            "bribe_attempts": c["bribe_attempts"],
            "bribe_success_pct": rate("bribe_succeeded", "bribe_attempts"),
            "burned_total": c["bribe_burned"],
            "end_to_end_pct": round(100 * money_end_to_end, 2),
        },
        "merit": {
            "plays": c["work_plays"],
            "survived_pct": rate("work_survived", "work_plays"),
            "kept_pct": round(100 * c["work_kept"] / max(1, c["work_gross"]), 2),
            "promo_attempts": c["merit_promo_attempts"],
            "promo_success_pct": rate("merit_promo_succeeded", "merit_promo_attempts"),
            "end_to_end_pct": round(100 * merit_end_to_end, 2),
        },
    }


# --------------------------------------------------------------------------
# 配置覆盖：让"改一条规则再跑一遍"变成一条命令
# --------------------------------------------------------------------------


def _coerce(name: str, raw: str, declared: Any) -> Any:
    """按 Config 上声明的类型把命令行字符串转成值。"""
    t = str(declared)
    if t == "bool":
        low = raw.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{name} 是 bool，给个 true/false，不是 {raw!r}")
    if t == "int":
        return int(raw)
    if t == "float":
        return float(raw)
    if t == "Fraction":
        return Fraction(raw)
    if t == "str":
        return raw
    raise ValueError(
        f"{name} 的类型是 {t}，命令行覆盖只支持 bool/int/float/Fraction/str。"
        f"这种复合字段请直接改 config.py"
    )


def apply_overrides(cfg: Config, settings: list[str]) -> Config:
    """把 --set key=value 落到一个新的 Config 上。

    用 dataclasses.replace，和 tests/test_rules.py 里那些
    `dataclasses.replace(CFG, attack_mode="denial")` 的夹具是同一套做法。
    字段名或值不对就当场报错——静默忽略会让整轮实验白跑。
    """
    if not settings:
        return cfg
    types = {f.name: f.type for f in dataclasses.fields(Config)}
    changes: dict[str, Any] = {}
    for item in settings:
        if "=" not in item:
            raise SystemExit(f"--set 要写成 key=value，收到 {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in types:
            near = [n for n in types if key in n or n in key][:5]
            hint = f"（是不是想写：{', '.join(near)}）" if near else ""
            raise SystemExit(f"config.py 里没有 {key!r} 这个字段{hint}")
        try:
            changes[key] = _coerce(key, raw, types[key])
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    return dataclasses.replace(cfg, **changes)


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------


def h(title: str) -> None:
    print()
    print(title)
    print("-" * max(56, len(title)))


def _parse_rounds(spec: str | None) -> set[int] | None:
    """'8-10' / '3,7' / '9' -> {轮次}"""
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.strip().partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def run_replay(args: argparse.Namespace, cfg: Config) -> int:
    """--section replay：按库里的手牌/事件/出牌重演一局，打印 AI 每轮怎么想的。"""
    import replay as rp
    from storage import GameStore

    if not Path(args.db).exists():
        print(f"找不到库：{args.db}", file=sys.stderr)
        return 1
    store = GameStore(args.db)
    try:
        gid = args.game_id or store.latest_game_id()
        history = store.history(gid) if gid else None
        if history is None:
            print(f"库里没有对局 {gid or ''}".rstrip(), file=sys.stderr)
            return 1
        result = rp.replay(
            history, cfg,
            overrides=dict(rp.parse_override(o) for o in args.override),
            rounds=_parse_rounds(args.rounds),
            top=args.top,
        )
    except rp.ReplayError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        store.close()

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    names = result.names

    def picks_str(picks) -> str:
        return "、".join(c + (f"@{names.get(t, t)}" if t is not None else "") for c, t in picks) or "（没出牌）"

    print("=" * 56)
    print(f"复盘 {result.game_id}   重演到第 {result.rounds_played} 轮")
    print("=" * 56)
    for d in result.decisions:
        h(f"第 {d.round} 轮 · {d.name}（AI）  {cfg.rank_name(d.rank)}  钱 {d.money}")
        for v in d.views:
            flag = "  <<< 这一轮就可能登顶" if v["about_to_win"] else ""
            print(f"  看 {v['name']}：{cfg.rank_name(v['rank'])}  政绩 {v['merit']}/{v['merit_cost']}"
                  f"  估钱 {v['money_est']}/{v['money_cost']}  威胁 {v['threat']}{flag}")
            cover = (f"  拦截把握 攻击/举报/一起 {v['cover']}" if v["cover"] else "")
            print(f"      攻击 {v['attack']:+.3f}  举报 {v['report']:+.3f}{cover}")
        print("  打分最高的组合：")
        for score, label in d.top:
            print(f"      {score:+.3f}  {label}")
        mark = "✓ 一致" if d.matched else "✗ 不一致"
        print(f"  重演会出：{picks_str(d.predicted)}")
        print(f"  当时出了：{picks_str(d.actual)}   {mark}")

    hit, total = result.match_rate
    h("汇总")
    if total:
        print(f"AI 决策复现 {hit}/{total}"
              + ("" if hit == total else
                 "（不一致不等于有 bug：服务器 AI 的噪声/打平挑目标是随机的，"
                 "服务器中途重启过 AI 记忆会清空；AI 代码改过之后旧局本来就对不上）"))
    if result.drifts:
        print(f"重演结果和库里对不上 {len(result.drifts)} 处（已按库校正）：")
        for line in result.drifts:
            print(f"  {line}")
    f = result.final
    if result.overridden:
        h(f"反事实：换掉出牌之后第 {f['round']} 轮的结局")
    else:
        h(f"第 {f['round']} 轮结局")
    print(f"  {f['game_over_reason'] or '游戏继续（' + f['phase'] + '）'}")
    for line in f["public_messages"]:
        print(f"  · {line}")
    for name, st in f["players"].items():
        print(f"  {name}：{cfg.rank_name(st['rank'])}  晋升 {st['promotion']}"
              f"  举报查实 {'是' if st['report_effective'] else '否'}"
              f"  挨攻击 {'是' if st['attacked'] else '否'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Meritocracy 平衡性分析")
    ap.add_argument("--players", type=int, default=6)
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20240923)
    ap.add_argument("--tournament-games", type=int, default=None)
    ap.add_argument(
        "--muted-seats", type=int, default=1,
        help="消融测试禁用这张牌的席位数。默认 1 = 真实均衡点"
             "（别人都照常玩，只有我少一个选项）",
    )
    ap.add_argument(
        "--sweep", action="store_true",
        help="消融测试额外输出 k=1..n-1 的整条曲线，看这张牌是不是'人多才安全'",
    )
    ap.add_argument(
        "--section",
        choices=("all", "leader", "rank", "actions", "choices", "reports", "triangle",
                 "ablation", "funnel", "strategy",
                 "postmortem", "events", "origins", "full", "replay"),
        default="all",
    )
    ap.add_argument(
        "--agents",
        default="random",
        help="第 1~3 节用谁来打：random（完全随机）或 smart（思考型 AI），"
             "也可以是任意策略名",
    )
    ap.add_argument(
        "--origins", choices=("none", "melee"), default="none",
        help="第 1~5 节对局要不要带出身。melee = 每局把所有出身随机发给各座位"
             "（6 人桌正好六张各一），并多出一节「身份战绩」",
    )
    ap.add_argument("--examples", type=int, default=5,
                    help="死因分析要打印几局具体例子")
    ap.add_argument("--json", action="store_true")
    # ---- --section replay：复盘库里的一局 ----
    ap.add_argument("--game-id", default=None, help="复盘哪一局（默认库里最新的一局）")
    ap.add_argument("--db", default=str(Path(__file__).resolve().parent / "meritocracy.db"))
    ap.add_argument("--rounds", default=None,
                    help="只详细打印这几轮的 AI 决策，例如 8-10 或 3,7（默认全部）")
    ap.add_argument("--top", type=int, default=6, help="每个 AI 打印打分最高的几组牌")
    ap.add_argument(
        "--override", action="append", default=[], metavar="轮:玩家=牌[@目标],...",
        help="换掉某人某一轮的出牌看反事实，可重复。例：--override 10:2=REPORT@1,PROMOTE_ANY",
    )
    ap.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", dest="settings",
        help="临时覆盖 config.py 里的一个字段，可重复。"
             "例：--set report_catches_bribery=false。用来做规则 A/B。",
    )
    args = ap.parse_args(argv)

    cfg = apply_overrides(DEFAULT_CONFIG, args.settings)
    if args.section == "replay":
        return run_replay(args, cfg)
    out: dict[str, Any] = {"players": args.players, "games": args.games, "seed": args.seed}

    FULL = args.section in ("all", "full")
    need_games = FULL or args.section in (
        "leader", "rank", "actions", "choices", "reports", "postmortem", "events")
    records: list[Record] = []
    if need_games:
        rng = random.Random(args.seed)
        names = [args.agents] * args.players
        origin_ids = cfg.origin_ids() if args.origins == "melee" else []

        def deal_origins() -> list[str | None] | None:
            if not origin_ids:
                return None
            pool = list(origin_ids)
            rng.shuffle(pool)
            return [pool[i % len(pool)] for i in range(args.players)]

        records = [
            play(args.players, rng, cfg, build_assignment(names, cfg, rng),
                 origins=deal_origins())
            for _ in range(args.games)
        ]
        if origin_ids:
            out["origin_results"] = analyse_origin_results(records, cfg)

    if FULL or args.section == "leader":
        out["leadership"] = analyse_leadership(records, cfg)
    if FULL or args.section == "rank":
        out["rank_dynamics"] = analyse_rank_dynamics(records, cfg)
    if FULL or args.section == "actions":
        out["actions"] = analyse_actions(records, cfg)
    if FULL or args.section == "choices":
        out["choices"] = analyse_choices(records, cfg)
    if FULL or args.section == "reports":
        out["reports"] = analyse_reports(records, cfg)
    if FULL or args.section == "events":
        out["events"] = analyse_events(records, cfg)
        out["event_swings"] = analyse_event_swings(records, cfg)
    if FULL or args.section == "postmortem":
        diags = [diagnose_loser(r, cfg) for r in records]
        blame_total: dict[str, float] = defaultdict(float)
        cause_count: Counter = Counter()
        for d in diags:
            for k, v in d["blame"].items():
                blame_total[k] += v
            for c in d["top_causes"][:1]:
                cause_count[c["label"]] += 1
        n = max(1, len(diags))
        out["postmortem"] = {
            "games": len(diags),
            "blame_avg": {k: round(v / n, 1) for k, v in
                          sorted(blame_total.items(), key=lambda kv: -kv[1])},
            "top_cause_frequency": {k: pct(v, n) for k, v in cause_count.most_common()},
            "avg_promotions_loser": round(
                statistics.fmean([d["promotions"] for d in diags]), 2),
            "avg_promotions_others": round(
                statistics.fmean([d["avg_promotions"] for d in diags]), 2),
            "examples": [narrate(d, cfg) for d in diags[: args.examples]],
            "loser_experience": analyse_loser_experience(records, cfg),
        }
    if args.section == "funnel":
        out["funnel"] = analyse_funnel(
            args.players, args.games, cfg, random.Random(args.seed)
        )

    if args.section == "ablation":
        out["ablation"] = analyse_ablation(
            args.players, args.games, cfg, random.Random(args.seed),
            muted_seats=args.muted_seats, sweep=args.sweep,
        )

    if args.section == "triangle":
        out["triangle"] = analyse_triangle(
            args.players, args.games, cfg, random.Random(args.seed)
        )

    if args.section == "origins":
        out["origins"] = analyse_origins(
            args.players, args.games, cfg, random.Random(args.seed)
        )

    if FULL or args.section == "strategy":
        tg = args.tournament_games or max(2000, args.games // 4)
        if args.agents == "smart":
            # 一桌**都是思考型 AI**，区别只在"允许用哪几张牌"。
            # 比脚本混战靠谱得多：脚本是死认一张牌打到底，赢输往往反映的是
            # 脚本本身有多蠢，而不是那条路线强不强。
            pool = ["smart", "smart", "smart_no_attack", "smart_no_report",
                    "smart_no_corrupt", "smart_clean"][: args.players]
            solo = ("smart_no_attack", "smart_no_report", "smart_no_corrupt",
                    "smart_clean")
            baseline = "smart"
        else:
            pool = ["climber", "safe_climber", "worker", "corrupt", "saboteur",
                    "random"]
            solo = ("climber", "safe_climber", "corrupt", "worker", "saboteur",
                    "reporter", "attacker", "passive")
            baseline = "random"
        out["tournament_mixed"] = tournament(args.players, tg, cfg, args.seed + 1, pool)
        out["tournament_vs_random"] = {
            name: tournament(args.players, tg // 2, cfg, args.seed + 2,
                             [name] + [baseline] * (args.players - 1))["win_share_pct"]
            for name in solo
        }
        out["tournament_baseline"] = baseline

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    print("=" * 56)
    print(f"Meritocracy 平衡性分析   {args.players} 人 x {args.games} 局"
          f"   对局玩家={args.agents}   seed={args.seed}")
    print("=" * 56)
    # --agents 只管前三节。消融/漏斗/三角这些自带对局循环，一律用思考型 AI，
    # 不然表头写着 random，读的人会以为消融结果和 AI 的改动无关。
    if args.agents != "smart" and {"ablation", "funnel", "triangle", "origins"} & set(out):
        which = "、".join(
            n for n in ("ablation", "funnel", "triangle", "origins") if n in out
        )
        print(f"  注：{which} 这几节固定用思考型 AI 对局，不受 --agents 影响")

    if "origin_results" in out:
        d = out["origin_results"]
        h("0. 身份战绩（同一批对局，每局六个出身随机发给六个座位）")
        print(f"  公平线 {100 / max(1, args.players):.2f}%；胜率按并列冠军 1/人数折算")
        print(f"  {'出身':<18}{'胜率':>8}{'±95%':>7}{'当主席赢':>9}{'垫底率':>8}"
              f"{'平均名次':>9}{'晋升次数':>9}{'被瞄准/局':>10}")
        for oid, r in d.items():
            name = (cfg.origin(oid) or {"name": "无出身"})["name"]
            print(f"  {name:<18}{r['win_pct']:>7.2f}%{r['ci_half_width']:>6.2f}"
                  f"{r['president_win_pct']:>8.2f}%{r['last_place_pct']:>7.2f}%"
                  f"{r['avg_place']:>9.2f}{r['avg_promotions']:>9.2f}{r['targeted_per_game']:>10.2f}")
        rates = [r["win_pct"] for r in d.values()]
        print(f"\n  最强 vs 最弱差 {max(rates) - min(rates):.2f} 个点")

    if "leadership" in out:
        d = out["leadership"]
        h("1. 领先权的易手频率")
        print(f"节奏: 主席率 {d['president_pct']}%，平均 {d['avg_rounds']} 轮结束，"
              f"打满 {cfg.max_rounds} 轮 {100 - d['president_pct']:.2f}%")
        print("结束轮次: " + "  ".join(
            f"{k}:{pct(v, args.games):.0f}%" for k, v in d["end_round_hist"].items()))
        print(f"每局冠军人数: 平均 {d['avg_winners_per_game']}，"
              f"出现并列冠军的对局 {d['co_winner_games_pct']}%  {d['co_winner_hist']}")
        print(f"每局领先者易手次数（官职口径）: {d['lead_changes_rank_avg']}")
        print(f"每局领先者易手次数（计分口径）: {d['lead_changes_score_avg']}")
        print(f"一局里当过领先者的人数（官职/计分）: "
              f"{d['distinct_leaders_rank_avg']} / {d['distinct_leaders_score_avg']}")
        print(f"全程独占领先直到夺冠（wire-to-wire）: {d['wire_to_wire_pct']}%")
        print(f"冠军全程一次都没领先过           : {d['winner_never_led_until_the_end_pct']}%")
        print(f"最后一轮完成反超                 : {d['last_round_flip_pct']}%")
        print("\n易手次数分布:")
        for k, v in d["lead_changes_rank_hist"].items():
            print(f"  {k} 次: {v:>7}  {pct(v, args.games):>6.2f}% {'#' * int(40 * v / args.games)}")
        print("\n第 N 轮领先者的最终夺冠率（领先锁定曲线）:")
        for r, v in d["win_rate_given_leading_at_round"].items():
            keep = d["lead_retention_next_round"].get(r)
            keep_s = f"  下一轮仍领先 {keep:>5.1f}%" if keep is not None else ""
            print(f"  第 {r:>2} 轮: {v:>6.2f}%{keep_s}")

    if "rank_dynamics" in out:
        d = out["rank_dynamics"]
        h("2. 爬上去之后还拦得住吗")
        print("每轮官职变化（按当前官职分组）:")
        print(f"  {'官职':<10} {'升':>8} {'不变':>8} {'降':>8}")
        for name, t in d["rank_transition_pct"].items():
            print(f"  {name:<10} {t.get('up', 0):>7.2f}% {t.get('same', 0):>7.2f}%"
                  f" {t.get('down', 0):>7.2f}%")
        print("\n到达过某官职的玩家，其最终夺冠率:")
        for name, v in d["win_rate_after_reaching"].items():
            print(f"  {name:<10}: {v:>6.2f}%")
        print("\n全场第一个到达某官职的玩家，其夺冠率:")
        for name, v in d["win_rate_if_first_to_reach"].items():
            print(f"  {name:<10}: {v:>6.2f}%")
        print("\n到达过某官职后又被打下来的比例:")
        for name, v in d["knocked_back_after_reaching_pct"].items():
            print(f"  {name:<10}: {v:>6.2f}%")
        print(f"\n从省级干部到国家主席平均还要 "
              f"{d['rounds_from_provincial_to_president_avg']} 轮")

    if "actions" in out:
        d = out["actions"]
        h("3. 四种行动的边际价值（全员随机对局）")
        print(f"基准胜率（1/{args.players}）: {d['baseline_win_pct']}%")
        print("\n按该牌本局打出的次数分组的胜率:")
        for card in [c for c in ("WORK", "CORRUPT", "GRAFT", "REPORT", "ATTACK")
                     if c in d["by_card_usage"]]:
            row = " ".join(
                f"{k}次:{v['win_pct']:>5.1f}%" for k, v in d["by_card_usage"][card].items()
            )
            print(f"  {card:<8} {row}")
        print("\n冠军 vs 其他人的平均使用次数:")
        for card, v in d["mean_usage_winners_vs_others"].items():
            flag = "  <<< 明显偏高" if v["delta"] > 0.3 else ("  <<< 明显偏低" if v["delta"] < -0.3 else "")
            print(f"  {card:<8} 冠军 {v['winners']:>5.2f}   其他 {v['others']:>5.2f}"
                  f"   差 {v['delta']:>+6.2f}{flag}")
        print("\n挨打的代价:")
        for label, key in (("被举报查实", "win_pct_by_times_reported"),
                           ("被攻击命中", "win_pct_by_times_attacked"),
                           ("被打回基层", "win_pct_by_times_nuked_to_base")):
            row = " ".join(f"{k}次:{v['win_pct']:>5.1f}%" for k, v in d[key].items())
            print(f"  {label:<10} {row}")

    if "choices" in out:
        d = out["choices"]
        h("3b. 手上有这张牌时，玩家真的会打它吗")
        for c, v in d["pick_rate_when_in_hand"].items():
            print(f"  {c:<14} 在手 {d['offered'].get(c, 0):>8} 次，打出 "
                  f"{d['chosen'].get(c, 0):>8} 次 -> 选择率 {v:>6.2f}%")
        if d.get("attack_target_by_rank"):
            print("\n  攻击目标的官职分布（focus = 挨打份额 ÷ 人口份额，>1 = 被重点照顾）:")
            print(f"    {'官职':<10}{'挨打份额':>9}{'场上人口':>9}{'focus':>8}")
            for name, v in d["attack_target_by_rank"].items():
                tag = "  <<< 重点照顾" if v["focus"] >= 1.5 else (
                    "  <<< 基本没人理" if v["focus"] < 0.5 else "")
                print(f"    {name:<10}{v['attacked_share_pct']:>8.1f}%"
                      f"{v['population_share_pct']:>8.1f}%{v['focus']:>8.2f}{tag}")
        if d["attacks"]:
            print(f"\n  每次发动攻击时，目标离政绩晋升线还差多少（共 {d['attacks']} 次）:")
            for k, v in sorted(d["attack_target_merit_gap"].items(), key=lambda kv: -kv[1]):
                print(f"    {k:<26}: {v:>6.2f}%")

    if "reports" in out and out["reports"].get("shots"):
        d = out["reports"]
        h("3c. 举报专项：这张牌值不值一个回合")
        print(f"  总共打出举报 {d['shots']} 次")
        print(f"  查实率                    : {d['hit_pct']:>6.2f}%"
              f"   （其中目标真的在贪的占 {d['target_was_corrupting_pct']:.2f}%，"
              f"其余是抓到拿钱买官）")
        print(f"  每张举报牌平均到手        : {d['mean_take_per_card']:>6.2f} 金钱")
        print(f"  查实时平均到手            : {d['mean_take_when_landed']:>6.2f} 金钱"
              f"   （查实后赃款池平均 {d['mean_loot_pool_when_landed']:.2f}）")
        print(f"  查实了却一分没分到        : {d['zero_take_despite_landing_pct']:>6.2f}%")
        print(f"  对照：一张 WORK 的期望点数 {d['work_card_expected_merit']:.2f}"
              f"（基层 = 这么多政绩）")
        print("\n  查实后的处分构成:")
        for k, v in sorted(d["demotion_mix"].items(), key=lambda kv: -kv[1]):
            cn = {
                "MAJOR": "重大贪腐，打回基层",
                "MINOR": "警告记满，降一级",
                # 警告制下，头一次被查实只记警告、不降级
                "NONE": "只记严重警告，未降级",
            }.get(k, k)
            print(f"    {cn:<22}: {v:>6.2f}%")
        print("\n  举报撞车（几个人同时举报同一个目标）:")
        for k, v in d["crowding"].items():
            print(f"    {k:<24}: {v:>6.2f}%")
        print("\n  按目标官职拆分:")
        print(f"    {'目标官职':<10}{'占比':>8}{'查实率':>9}{'平均到手':>10}")
        for name, v in d["by_target_rank"].items():
            print(f"    {name:<10}{v['share_pct']:>7.1f}%{v['hit_pct']:>8.1f}%"
                  f"{v['mean_take']:>10.2f}")

    if "funnel" in out:
        d = out["funnel"]
        m, w = d["money"], d["merit"]
        h("金钱路线 vs 政绩路线：逐级漏斗")
        print(f"  {d['games']} 局。看的是「赚到的东西最后有多少真换成了官职」。\n")
        print(f"  {'':<16}{'金钱路线':>14}{'政绩路线':>14}")
        print(f"  {'产出回合数':<14}{m['plays']:>14}{w['plays']:>14}")
        print(f"  {'扛过干扰':<15}{m['survived_pct']:>13.1f}%{w['survived_pct']:>13.1f}%"
              f"   <- 金钱怕举报，政绩怕抢功")
        print(f"  {'产出留存率':<14}{m['kept_pct']:>13.1f}%{w['kept_pct']:>13.1f}%")
        print(f"  {'兑现尝试次数':<13}{m['bribe_attempts']:>14}{w['promo_attempts']:>14}")
        print(f"  {'兑现成功率':<14}{m['bribe_success_pct']:>13.1f}%"
              f"{w['promo_success_pct']:>13.1f}%")
        print(f"  {'端到端存活率':<13}{m['end_to_end_pct']:>13.1f}%"
              f"{w['end_to_end_pct']:>13.1f}%   <- 赚+花两关都过")
        print(f"\n  行贿打水漂的钱总共 {m['burned_total']}")

    if "ablation" in out:
        d = out["ablation"]
        k = d["muted_seats"]
        h("消融测试：每张牌值不值得用")
        print(f"  {d['games']} 局，{args.players} 人桌里 {k} 席被禁用该牌、"
              f"其余照常，逐局轮换座位")
        if k == 1:
            print("  （1 席 = 真实均衡点：别人都照常玩，只有我少一个选项）\n")
        else:
            print(f"  ⚠ 禁了 {k} 席，牌桌已偏离均衡点；"
                  f"对'人多才安全'的牌会低估它，建议用 --muted-seats 1\n")
        print(f"  {'牌':<6}{'禁用者':>9}{'正常人':>9}{'差值':>9}"
              f"{'95% 区间':>18}   判定")
        for label, v in d["cards"].items():
            lo, hi = v["delta"] - v["ci_half_width"], v["delta"] + v["ci_half_width"]
            if lo <= 0 <= hi:
                verdict = "分不出来（样本不够）"
            elif lo > 0:
                verdict = "负收益，不如不用"
            else:
                verdict = "偏强"
            print(f"  {label:<6}{v['muted_win_pct']:>8.2f}%{v['normal_win_pct']:>8.2f}%"
                  f"{v['delta']:>+9.2f}{f'[{lo:+.2f}, {hi:+.2f}]':>18}   {verdict}")
        print(f"\n  每张牌被禁那一方的样本数：{d['cards']['贪污']['samples_muted']}"
              f"（想收窄区间就加 --games）")

        if "sweep" in d:
            print("\n  组成曲线：禁用席位数 -> 差值"
                  "（单调上升 = 这张牌人多才安全）")
            head = "".join(f"{i:>8}" for i in range(1, args.players))
            print(f"    {'禁用席位':<10}{head}")
            for label, row in d["sweep"].items():
                print(f"    {label:<10}" + "".join(f"{v:>+8.2f}" for v in row))

    if "triangle" in out:
        d = out["triangle"]
        h("鹬蚌相争：目标动态成不成立")
        print(f"  {d['games']} 局，名次 1 = 第一名，越小越好\n")

        one = d["scenario_one"]
        print("  场景一：#1 闷头建设，#2 专门搞他")
        for r in ("builder", "challenger"):
            if r in one["avg_place"]:
                print(f"    {r:<12} 平均名次 {one['avg_place'][r]:>6}"
                      f"   夺冠 {one['win_pct'].get(r, 0):>6.2f}%")
        ok1 = one["avg_place"].get("challenger", 9) < one["avg_place"].get("builder", 0)
        print(f"    判据「第二名能反超」: {'✔ 成立' if ok1 else '✘ 不成立'}")

        two = d["scenario_two"]
        print("\n  场景二：两个决斗者死盯着对方互相搞，渔翁闷头建设")
        for r in ("duelist", "fisherman"):
            if r in two["avg_place"]:
                print(f"    {r:<12} 平均名次 {two['avg_place'][r]:>6}"
                      f"   夺冠 {two['win_pct'].get(r, 0):>6.2f}%")
        ok2 = two["avg_place"].get("fisherman", 9) < two["avg_place"].get("duelist", 0)
        print(f"    判据「渔翁得利」:   {'✔ 成立' if ok2 else '✘ 不成立'}")

    if "origins" in out:
        d = out["origins"]
        h("出身卡：六张各值多少")
        print(f"  {d['games']} 局。**强制随机分配、逐局轮换座位**——")
        print("  让 AI 自己挑的话，弱身份被选得少、胜率反而好看（选择偏差）。\n")
        if not d["solo"]:
            print("  出身没开（origins_enabled=false）。")
        else:
            print("  单身份对照：1 席拿这个出身，其余 5 席无出身")
            print(f"    {'出身':<20}{'触发/局':>9}{'他的胜率':>10}{'旁人':>8}"
                  f"{'差值':>8}{'95%区间':>18}{'夺冠轮次':>10}")
            for oid, r in d["solo"].items():
                info = cfg.origin(oid) or {"name": oid}
                lo = r["delta"] - r["ci_half_width"]
                hi = r["delta"] + r["ci_half_width"]
                wr = f"{r['avg_win_round']:.1f}" if r["avg_win_round"] else "—"
                fr = f"{r['fires_per_game']}"
                print(f"    {info['name']:<20}{fr:>9}"
                      f"{r['win_pct']:>9.2f}%{r['others_pct']:>7.2f}%"
                      f"{r['delta']:>+8.2f}   [{lo:+.2f}, {hi:+.2f}]{wr:>10}")
            fires = [r["fires_per_game"] for r in d["solo"].values()]
            if min(fires) < 0.1:
                print("\n    注：触发率接近 0 的那几张，胜率差一定是噪声，别去解读它。")

            print("\n  混战：6 人各一个出身，随机排列（公平线"
                  f" {100 / max(1, len(d['melee'])):.2f}%）")
            # 循环变量别叫 pct —— 模块级有个同名函数，在 main() 里被它一遮，
            # 连不走这个分支的 --section full 都会 UnboundLocalError
            for oid, share in sorted(d["melee"].items(), key=lambda kv: -kv[1]):
                info = cfg.origin(oid) or {"name": oid}
                print(f"    {info['name']:<20}{share:>8.2f}%")

            deltas = [r["delta"] for r in d["solo"].values()]
            spread = max(deltas) - min(deltas)
            print(f"\n  最强 vs 最弱差 {spread:.2f} 个点"
                  f"（判据 < 3）：{'✔' if spread < 3 else '✘'}")

    if "postmortem" in out:
        d = out["postmortem"]
        h("5. 最后一名为什么输（逐局死因分析）")
        print(f"垫底者平均晋升 {d['avg_promotions_loser']} 次，其他人平均 "
              f"{d['avg_promotions_others']} 次")
        print("\n责任归口（每局归一化后取平均）:")
        for k, v in d["blame_avg"].items():
            bar = "#" * int(v / 2)
            print(f"  {k:<8}{v:>6.1f}%  {bar}")
        print("\n最主要死因的出现频率:")
        for k, v in d["top_cause_frequency"].items():
            print(f"  {k:<24}{v:>6.2f}%")
        print(f"\n随机 {len(d['examples'])} 局的具体分析:")
        for i, line in enumerate(d["examples"], 1):
            print(f"  [{i}] {line}")

    if "events" in out:
        d = out["events"]
        h("6. 事件卡有没有存在感")
        print(f"基准 = 「{d['baseline']}」（空效果）。下面是各事件所在轮次相对基准的变化。")
        print(f"  {'事件':<12}{'占比':>7}{'政绩':>8}{'金钱':>8}{'晋升':>7}"
              f"{'举报查实':>9}{'政绩被毁':>9}{'存在感':>8}")
        rows = sorted(d["events"].items(), key=lambda kv: -kv[1]["impact_score"])
        for name, e in rows:
            dl = e["delta_vs_baseline"]
            tag = "" if e["impact_score"] >= 0.15 else "  <<< 几乎无感"
            print(f"  {name:<12}{e['share_pct']:>6.1f}%{dl['merit_gained']:>+8.1f}"
                  f"{dl['money_gained']:>+8.1f}{dl['promotions']:>+7.2f}"
                  f"{dl['reports_landed']:>+9.2f}{dl['merit_destroyed']:>+9.1f}"
                  f"{e['impact_score']:>8.2f}{tag}")

    if "event_swings" in out:
        sw = out["event_swings"]
        base = out["events"]["baseline"]
        b = sw.get(base, {"leader_flip_pct": 0, "rank_changes_per_round": 0})
        print("\n  事件能不能造成局面反转（该事件那一轮，领先者易手的比例）:")
        print(f"  {'事件':<12}{'领先易手':>9}{'vs 基准':>9}{'官职变动/轮':>12}")
        for name, e in sorted(sw.items(), key=lambda kv: -kv[1]["leader_flip_pct"]):
            d = e["leader_flip_pct"] - b["leader_flip_pct"]
            tag = "  <<< 掀不起浪" if abs(d) < 2 and name != base else ""
            print(f"  {name:<12}{e['leader_flip_pct']:>8.1f}%{d:>+8.1f}%"
                  f"{e['rank_changes_per_round']:>12.2f}{tag}")

    if "postmortem" in out and "loser_experience" in out["postmortem"]:
        e = out["postmortem"]["loser_experience"]
        h("5b. 垫底的人这一局过得怎么样")
        print(f"  当过一次领先者             : {e['ever_held_the_lead_pct']}%")
        print(f"  平均晋升次数               : {e['avg_promotions']}"
              f"（全程一次没升过的占 {e['never_promoted_pct']}%）")
        print(f"  卡在基层公务员的轮数         : {e['avg_rounds_stuck_at_base_rank']}")
        print(f"  真正推进自己的回合占比       : {e['productive_turn_share']}%")
        print(f"  被瞄准次数 / 应得份额        : {e['targeted_vs_fair_share']}  "
              f"（1.0 = 没有被特别针对，>1.3 就会有被集火的感觉）")
        print(f"  平均被降级次数             : {e['avg_times_demoted']}")

    if "tournament_mixed" in out:
        d = out["tournament_mixed"]
        smart_pool = out.get("tournament_baseline") == "smart"
        h("4. 打法对抗赛（同桌混战，轮换座位）" if smart_pool
          else "4. 策略对抗赛（同桌混战，轮换座位）")
        print(f"{d['games']} 局，公平线 {d['fair_share_pct']}%"
              f"（夺冠份额把并列冠军按 1/人数 折算，相加为 100%）")
        if smart_pool:
            print("一桌都是思考型 AI，区别只在**允许用哪几张牌**")
        print(f"  {'策略':<14} {'夺冠份额':>9} {'含并列':>8} {'当主席':>8} {'终局官职':>9}")
        for name, v in d["win_share_pct"].items():
            mark = ""
            if v > d["fair_share_pct"] * 1.4:
                mark = "  <<< 偏强"
            elif v < d["fair_share_pct"] * 0.65:
                mark = "  <<< 偏弱"
            print(f"  {name:<14} {v:>8.2f}% {d['win_pct'][name]:>7.2f}%"
                  f" {d['president_pct'][name]:>7.2f}% {d['final_rank_avg'][name]:>8.2f}{mark}")
        base = out.get("tournament_baseline", "random")
        d2 = out["tournament_vs_random"]
        label = "随机玩家" if base == "random" else "完整版思考型 AI"
        print(f"\n单挑测试：1 个该打法 + {args.players - 1} 个{label}"
              f"（公平线 {round(100 / args.players, 2)}%）")
        rows = sorted(d2.items(), key=lambda kv: -kv[1].get(kv[0], 0))
        for name, wp in rows:
            print(f"  {name:<14} {wp.get(name, 0):>7.2f}%")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
