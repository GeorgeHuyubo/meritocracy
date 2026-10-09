"""对局遥测 + "焦点座位"实验框架：搜索探针（probe.py）、大模型混坐（llm_play.py）、
对照局、对比工具（crosscheck.py）共用一套口径。

焦点座位设计：一局 = 1 个被测座位 + 5 个陪练。第 g 局的被测身份 = ORIGINS[g % 6]、
座位 = (g // 6) % 6，每 36 局把（身份 × 座位）各覆盖一次；其余 5 个身份按种子洗牌。

配对：每轮发牌前把 game.rng 换成 Random(f"{game_seed}/r{轮}")，陪练和被测各用自己的 rng。
这样同一个 g 下"大模型坐焦点""搜索 AI 坐焦点""学习型 AI 坐焦点"三份对局的身份、座位、
每轮手牌和事件都一样（只有这一轮里换牌、结算的随机数会因打法不同而分叉），可以逐局配对。
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from typing import Any, Callable

from config import DEFAULT_CONFIG, Config
from game import Game
from models import Card, Origin, RoundOutcome

TELEMETRY_VERSION = 1
NAMES = ["老张", "老李", "老王", "老赵", "老刘", "老陈"]

Decider = Callable[[Game, int], list[dict]]


def focal_assignment(g: int, seed: int, cfg: Config = DEFAULT_CONFIG,
                     n_players: int = 6) -> tuple[str, int, list[str]]:
    """第 g 局：(被测身份, 被测座位下标, 每个座位的身份)。"""
    origins = list(cfg.origin_ids())
    focal_origin = origins[g % len(origins)]
    focal_seat = (g // len(origins)) % n_players
    rest = [o for o in origins if o != focal_origin]
    random.Random(f"{seed}/{g}/origins").shuffle(rest)
    seats = rest[:focal_seat] + [focal_origin] + rest[focal_seat:]
    return focal_origin, focal_seat, seats[:n_players]


def game_seed(seed: int, g: int) -> int:
    return seed * 1000 + g


def reseed_round(game: Game, gseed: int) -> None:
    """下一轮发牌前调用：第 r 轮的手牌和事件只取决于 (种子, r)。"""
    game.rng = random.Random(f"{gseed}/r{game.round_number + 1}")


def rules_fingerprint(cfg: Config) -> str:
    """规则指纹：配置去掉 AI 权重路径后的 sha1（同一套规则才能放在一起比）。"""
    d = asdict(cfg)
    d.pop("ai_policy", None)
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def new_game(cfg: Config, origins: list[str | None], gseed: int, game_id: str = "tm",
             names: list[str] | None = None) -> Game:
    """建一局、按座位发好出身、开局（第 1 轮已按种子发牌）。"""
    game = Game(game_id=game_id, cfg=cfg, rng=random.Random(f"{gseed}/start"))
    for name in (names or NAMES)[: len(origins)]:
        game.add_player(name, is_ai=True)
    for pid, oid in zip(sorted(game.players), origins):
        game.players[pid].origin = Origin(oid) if oid else None
    reseed_round(game, gseed)
    game.start_game()
    return game


def round_record(game: Game, outcome: RoundOutcome,
                 controller: dict[int, str] | None = None) -> dict[str, Any]:
    """一轮结算的完整记录（含私密的钱，只用于分析，不进任何广播）。"""
    controller = controller or {}
    players = []
    for pid, o in sorted(outcome.outcomes.items()):
        p = game.players[pid]
        players.append({
            "pid": pid,
            "origin": p.origin.value if p.origin else None,
            "controller": controller.get(pid),
            "cards": [c.value for c in o.cards],
            "targets": list(o.targets),
            "redraws": o.redraw_count,
            "redraw_spent": o.redraw_spent,
            "rank_before": o.rank_before,
            "rank_after": o.rank_after,
            "money_after": o.money_after,
            "merit_after": o.merit_after,
            "warnings_after": o.warnings_after,
            "tenure_after": o.tenure_after,
            "promotion": o.promotion.value,
            "family": o.family_promotion,
            "promo_money_cost": o.promotion_money_cost,
            "promo_merit_cost": o.promotion_merit_cost,
            "promotion_card_played": o.promotion_card_played,
            "promotion_blocked": bool(o.merit_promotion_blocked or o.promotion_frozen_by_report),
            "attacked": o.attacked,
            "attacked_by": list(o.attacked_by),
            "attacker_count": o.attacker_count,
            "attack_merit_loss": o.attack_merit_loss,
            "merit_from_attacks": o.merit_from_attacks,
            "attacks_landed": o.attacks_landed,
            "reported": o.reported,
            "report_count_players": o.report_count_players,
            "report_from_event": o.report_from_event,
            "report_effective": o.report_effective,
            "reports_landed": o.reports_landed,
            "demotion": o.demotion.value,
            "warnings_issued": o.warnings_issued,
            "money_confiscated": o.money_confiscated,
            "corrupt_amount": o.corrupt_amount,
            "laundered": o.laundered,
            "money_gained": o.money_gained,
            "merit_gained": o.merit_gained,
            "money_from_reports": o.money_from_reports,
        })
    return {
        "v": TELEMETRY_VERSION,
        "round": outcome.round_number,
        "event": outcome.event.id,
        "presidents": list(outcome.presidents),
        "players": players,
    }


def game_summary(game: Game, controller: dict[int, str] | None = None) -> dict[str, Any]:
    """终局信息：每人身份 / 控制方 / 终局官职钱政绩、冠军、名次、怎么结束的。"""
    controller = controller or {}
    standing = game.final_standing()
    return {
        "rounds": game.round_number,
        "president": bool(game.history and game.history[-1].presidents),
        "end": "president" if game.history and game.history[-1].presidents else "timeout",
        "winners": list(game.winners),
        "final_standing": standing,
        "players": [{
            "pid": pid,
            "name": p.name,
            "origin": p.origin.value if p.origin else None,
            "controller": controller.get(pid),
            "rank": p.rank, "money": p.money, "merit": p.merit, "warnings": p.warnings,
            "winner": pid in game.winners,
            "place": standing.index(pid) + 1,
        } for pid, p in sorted(game.players.items())],
    }


def drive_game(game: Game, deciders: dict[int, Decider], gseed: int,
               order: list[int] | None = None,
               controller: dict[int, str] | None = None,
               on_round: Callable[[Game, RoundOutcome], None] | None = None,
               ) -> list[dict[str, Any]]:
    """把一局打到底，返回每轮遥测。order = 每轮的出牌顺序（陪练在前、焦点在后）。

    deciders[pid](game, pid) 负责这个座位这一轮的全部动作（可以换牌），返回 picks。
    """
    order = order or sorted(game.players)
    telemetry: list[dict[str, Any]] = []
    while not game.is_over:
        for pid in order:
            picks = deciders[pid](game, pid) or []
            if not game.selections[pid].picks:
                game.select_actions(pid, picks)
            game.lock_action(pid)
        game.force_lock_all()
        game.reveal_event()
        outcome = game.resolve()
        telemetry.append(round_record(game, outcome, controller))
        if on_round:
            on_round(game, outcome)
        if not game.is_over:
            reseed_round(game, gseed)
            game.advance_round()
    return telemetry


def focal_order(n_players: int, focal_pid: int) -> list[int]:
    """陪练按座位先出，被测座位最后出：陪练换牌抽到的牌在两份配对对局里一样。"""
    pids = list(range(1, n_players + 1))
    return [p for p in pids if p != focal_pid] + [focal_pid]


def card_names(picks: list[dict]) -> list[str]:
    out = []
    for p in picks:
        a = p.get("action")
        out.append(a.value if isinstance(a, Card) else str(a))
    return out
