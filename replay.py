"""复盘：把库里存下来的一局按原样重演，看 AI 每一轮是怎么想的。

手牌、事件、每个人的出牌都用库里记的那份，AI 按当时能看到的公开信息重新 observe，
然后让它重新 decide 一遍，和当时实际出的牌对比。打分直接读 decide 留下的
`last_scores`，和真 AI 同一份代码，不会漂。

还能换掉某个人某一轮的出牌（`overrides`）看反事实："要是老张这轮举报了他，
他还登得了顶吗？"——换掉之后后面的轮次就和原局对不上了，所以重演到那一轮为止。

**"复现不了"不等于有 bug**，已知两种原因：
  * 服务器上的 AI 用 SystemRandom：决策噪声、打平随机挑目标，同样的局面每次可能不一样
  * 服务器中途重启过，AI 的对手记忆（估了谁多少钱）会清空；这里是从头连续记下来的
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import ai
import rules
from config import Config, DEFAULT_CONFIG
from game import Game
from models import Card, DealtCard, Origin, Phase


class ReplayError(Exception):
    pass


@dataclass
class AIDecision:
    round: int
    player_id: int
    name: str
    money: int
    rank: int
    views: list[dict[str, Any]]  # SmartAgent.explain()：它眼里的每个对手
    top: list[tuple[float, str]]  # 打分最高的几组牌（不含噪声）
    predicted: list[tuple[str, int | None]]  # 重演时 AI 会出什么
    actual: list[tuple[str, int | None]]  # 当时实际出了什么

    @property
    def matched(self) -> bool:
        return Counter(self.predicted) == Counter(self.actual)


@dataclass
class ReplayResult:
    game_id: str
    names: dict[int, str]
    decisions: list[AIDecision] = field(default_factory=list)
    drifts: list[str] = field(default_factory=list)  # 重演结果和库里对不上的地方
    rounds_played: int = 0
    overridden: bool = False
    # 重演停下来那一轮的结局（有 override 时就是反事实结局）
    final: dict[str, Any] = field(default_factory=dict)

    @property
    def match_rate(self) -> tuple[int, int]:
        return sum(d.matched for d in self.decisions), len(self.decisions)

    def to_dict(self) -> dict[str, Any]:
        hit, total = self.match_rate
        return {
            "game_id": self.game_id,
            "rounds_played": self.rounds_played,
            "overridden": self.overridden,
            "matched": hit,
            "decisions_total": total,
            "drifts": list(self.drifts),
            "final": self.final,
            "decisions": [
                {
                    "round": d.round, "player_id": d.player_id, "name": d.name,
                    "money": d.money, "rank": d.rank, "views": d.views,
                    "top": [{"score": s, "cards": c} for s, c in d.top],
                    "predicted": d.predicted, "actual": d.actual, "matched": d.matched,
                }
                for d in self.decisions
            ],
        }


def parse_override(spec: str) -> tuple[tuple[int, int], list[dict[str, Any]]]:
    """'10:2=REPORT@1,PROMOTE_ANY' -> ((10, 2), [{card, target_id}, ...])"""
    try:
        head, body = spec.split("=", 1)
        rnd, pid = (int(x) for x in head.split(":"))
        picks = []
        for item in filter(None, (x.strip() for x in body.split(","))):
            card, _, target = item.partition("@")
            Card(card.upper())  # 牌名写错了当场报
            picks.append({"card": card.upper(), "target_id": int(target) if target else None})
    except (ValueError, KeyError) as exc:
        raise ReplayError(f"看不懂 --override {spec!r}：格式是 轮次:玩家=牌[@目标],牌[@目标]") from exc
    return (rnd, pid), picks


def _to_index_picks(hand: list[DealtCard], picks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """库里的出牌 -> select_actions 吃的格式。

    同名牌点数可能不同（两张 WORK 一张 6 一张 8），按 (牌, 点数) 定位到手牌下标，
    不然 select_actions 会取第一张同名牌、点数就错了。没记点数（override）就取第一张。
    """
    taken: set[int] = set()
    out = []
    for p in picks:
        card = Card(p["card"])
        if card is Card.PROMOTE_FAMILY:  # 一纸调令不在手牌里
            out.append({"action": card.value, "target": None})
            continue
        value = p.get("value")
        idx = next(
            (i for i, d in enumerate(hand)
             if i not in taken and d.card is card and (value is None or d.value == value)),
            None,
        )
        if idx is None:
            raise ReplayError(f"手牌里没有 {p['card']}（点数 {value}）")
        taken.add(idx)
        out.append({"index": idx, "target": p.get("target_id")})
    return out


def replay(
    history: dict[str, Any],
    cfg: Config = DEFAULT_CONFIG,
    *,
    overrides: dict[tuple[int, int], list[dict[str, Any]]] | None = None,
    rounds: set[int] | None = None,
    ai_rng: random.Random | None = None,
    top: int = 6,
) -> ReplayResult:
    """重演一局。`rounds` 是要记录 AI 决策的轮次（None = 全部）。

    `ai_rng` 给 AI 的噪声和打平挑目标用；测试里传同一个 seed 就能逐位复现。
    """
    overrides = overrides or {}
    players = history["players"]
    names = {p["id"]: p["name"] for p in players}
    result = ReplayResult(game_id=history["game_id"], names=names)
    if not history["events"]:
        raise ReplayError(f"对局 {history['game_id']} 还没有结算过任何一轮，没什么可复盘的")

    game = Game(game_id=f"replay-{history['game_id']}", cfg=cfg, rng=random.Random(0))
    for p in players:
        game._next_player_id = p["id"]  # 大厅里踢过 AI 的话编号会有空洞，照库里的来
        game.add_player(p["name"], is_ai=p["is_ai"])
    for p in players:
        game.players[p["id"]].origin = Origin(p["origin"]) if p["origin"] else None
    game._begin_first_round()

    pool = ai.AgentPool(cfg=cfg, rng=ai_rng or random.Random(0))
    ai_ids = [p["id"] for p in players if p["is_ai"]]
    archive = {r["round"]: r for r in history["archive"]}
    last_round = max(history["events"])

    while True:
        rnd = game.round_number
        hands = history["hands"].get(rnd)
        if hands is None or rnd not in history["events"]:
            raise ReplayError(f"第 {rnd} 轮的手牌或事件没存下来")
        for pid in game.players:
            game.hands[pid] = [
                DealtCard(Card(c["card"]), int(c.get("value", 0))) for c in hands[pid]
            ]
            # 换牌：库里存的是换完之后的那手，钱按流水账补扣
            for entry in history["ledger"].get(pid, []):
                if entry["round"] != rnd:
                    continue
                for row in entry["rows"]:
                    if row["label"] == "重新抽牌":
                        game.players[pid].money += row["money"]
                        game.redraw_spent[pid] = game.redraw_spent.get(pid, 0) - row["money"]

        # 官二代「透风」看到的是本轮真实发生的那个事件，复盘时得和当时一样
        game.next_event = rules.event_by_id(history["events"][rnd], cfg)
        pub = game.public_state()
        for pid in ai_ids:
            agent = pool.get(pid)
            priv = game.private_state(pid)
            if rounds is not None and rnd not in rounds:
                agent.observe(pub, priv)  # 不看这一轮也得让它记住发生了什么
                continue
            predicted = agent.decide(pub, priv)  # decide 里会先 observe
            # 同名牌拿了两张时同一组合会出现好几次，按牌的多重集去重
            ranked, seen = [], set()
            for score, cards in sorted(agent.last_scores, key=lambda sc: -sc[0]):
                key = tuple(sorted(c.value for c in cards))
                if key not in seen and len(ranked) < top:
                    seen.add(key)
                    ranked.append((score, cards))
            result.decisions.append(AIDecision(
                round=rnd, player_id=pid, name=names[pid],
                money=priv["money"], rank=priv["rank"],
                views=agent.explain(pub, priv),
                top=[(round(s, 3), "+".join(c.value for c in cards)) for s, cards in ranked],
                predicted=[(c, t) for c, t in predicted],
                actual=[(a["card"], a.get("target_id")) for a in history["actions"][rnd][pid]],
            ))

        for pid in game.players:
            picks = overrides.get((rnd, pid))
            if picks is not None:
                result.overridden = True
            else:
                picks = history["actions"][rnd][pid]
            game.select_actions(pid, _to_index_picks(game.hands[pid], picks))
            game.lock_action(pid)
        game.phase = Phase.REVEAL_EVENT
        game.current_event = rules.event_by_id(history["events"][rnd], cfg)
        outcome = game.resolve()
        result.rounds_played = rnd

        # 换过牌就不再和库里比了：从这一轮起本来就该不一样
        if not result.overridden and rnd in archive:
            for rec in archive[rnd]["players"]:
                gp = game.players[rec["player_id"]]
                got = (gp.money, gp.merit, gp.rank)
                want = (rec["money_after"], rec["merit_after"], rec["rank_after"])
                if got != want:
                    result.drifts.append(
                        f"第 {rnd} 轮 {names[gp.id]}：重演 (钱, 政绩, 官职)={got}，库里 {want}，已按库校正"
                    )
                    gp.money, gp.merit, gp.rank = want

        if game.is_over or rnd >= last_round or result.overridden:
            result.final = {
                "round": rnd,
                "phase": game.phase.value,
                "winners": [names[w] for w in game.winners],
                "game_over_reason": game.game_over_reason,
                "public_messages": list(outcome.public_messages),
                "players": {
                    names[pid]: {
                        "rank": game.players[pid].rank,
                        "promotion": o.promotion.value,
                        "report_effective": o.report_effective,
                        "attacked": o.attacked,
                    }
                    for pid, o in sorted(outcome.outcomes.items())
                },
            }
            return result
        game.advance_round()
