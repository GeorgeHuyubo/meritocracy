"""Meritocracy 对局状态机。

服务器是唯一权威：所有随机数、结算、晋升、胜负都在这里产生。
客户端只能提交四件事：加入游戏、选择行动、选择目标、锁定。

这一层不 import 任何 Web/FastAPI 代码，所以 simulator.py 可以直接驱动它。
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any

import rules
from config import Config, DEFAULT_CONFIG
from models import (
    Action,
    Card,
    DealtCard,
    GameEvent,
    Origin,
    Phase,
    PlayerState,
    RoundOutcome,
    Selection,
)


class GameError(Exception):
    """客户端提交了非法操作。"""


@dataclass
class Game:
    game_id: str = "default"
    cfg: Config = DEFAULT_CONFIG
    rng: Any = field(default_factory=secrets.SystemRandom)

    phase: Phase = Phase.LOBBY
    round_number: int = 0
    players: dict[int, PlayerState] = field(default_factory=dict)
    hands: dict[int, list[DealtCard]] = field(default_factory=dict)
    selections: dict[int, Selection] = field(default_factory=dict)
    ready: set[int] = field(default_factory=set)
    # 本轮每个人花在重新抽牌上的钱（进流水账，不然钱会莫名其妙变少）
    redraw_spent: dict[int, int] = field(default_factory=dict)
    # 本轮每个人换了几次牌——同一轮里每换一次，下一次就翻倍
    redraw_count: dict[int, int] = field(default_factory=dict)
    # 本轮开局发出去的工资（进流水账）
    salary_paid: dict[int, int] = field(default_factory=dict)
    # 开局的出身红利（富二代的老钱），记下来给流水账用
    origin_bonus: dict[int, int] = field(default_factory=dict)
    # 红二代「一纸调令」已经用掉的人（每局一次，打出去就算用了，不管成没成）
    family_used: set[int] = field(default_factory=set)
    # 挑出身阶段每人手上的候选 {玩家id: [出身id, ...]}
    origin_choices: dict[int, list[str]] = field(default_factory=dict)

    current_event: GameEvent | None = None
    # 本轮已经抽好、但还没揭晓的事件。发牌时就抽，只有官二代「透风」能提前看到。
    next_event: GameEvent | None = None
    last_outcome: RoundOutcome | None = None
    history: list[RoundOutcome] = field(default_factory=list)
    public_log: list[str] = field(default_factory=list)

    winners: list[int] = field(default_factory=list)
    game_over_reason: str = ""
    # 全程累计的逐人统计，只用来在终局给每个人做复盘
    stats: dict[int, dict[str, int]] = field(default_factory=dict)
    # 逐人流水账：每轮每笔收支，给 UI 当"银行对账单"用（含金钱，仅私密可见）
    ledger: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    # 终局全揭示用的对局档案：每轮每人出了什么牌、打了谁
    archive: list[dict[str, Any]] = field(default_factory=list)

    _next_player_id: int = 1
    # 从数据库恢复时只剩下公开结果（私密结算细节不再重放）
    _restored_last_result: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # 大厅
    # ------------------------------------------------------------------

    @property
    def host_id(self) -> int | None:
        """房主 = 第一个加入的玩家，只有他能开始游戏 / 强制推进。"""
        return min(self.players) if self.players else None

    def ordered_players(self) -> list[PlayerState]:
        return [self.players[pid] for pid in sorted(self.players)]

    def add_player(self, name: str, is_ai: bool = False) -> PlayerState:
        if self.phase is not Phase.LOBBY:
            raise GameError("游戏已经开始，无法加入。")
        if len(self.players) >= self.cfg.max_players:
            raise GameError(f"人数已满（最多 {self.cfg.max_players} 人）。")
        name = (name or "").strip()
        if not name:
            raise GameError("请输入名字。")
        if len(name) > 12:
            name = name[:12]
        if any(p.name == name for p in self.players.values()):
            raise GameError("这个名字已经有人用了。")

        player = PlayerState(
            id=self._next_player_id,
            name=name,
            token=secrets.token_urlsafe(16),
            connected=True,
            is_ai=is_ai,
        )
        self.players[player.id] = player
        self.selections[player.id] = Selection()
        self._next_player_id += 1
        self.log(f"{player.name} 加入了游戏。" if not is_ai else f"{player.name}（AI）入局了。")
        return player

    def remove_player(self, player_id: int, requester_id: int | None = None) -> None:
        """只能在大厅里踢人，而且只能踢 AI。"""
        if self.phase is not Phase.LOBBY:
            raise GameError("游戏已经开始，不能再调整人数。")
        if requester_id is not None and requester_id != self.host_id:
            raise GameError("只有房主可以调整 AI。")
        player = self.players.get(player_id)
        if player is None:
            raise GameError("没有这个玩家。")
        if not player.is_ai:
            raise GameError("只能移除 AI 玩家。")
        self.players.pop(player_id)
        self.selections.pop(player_id, None)
        self.log(f"{player.name} 被移出了房间。")

    def ai_player_ids(self) -> list[int]:
        return [pid for pid, p in sorted(self.players.items()) if p.is_ai]

    def player_by_token(self, token: str) -> PlayerState | None:
        if not token:
            return None
        for p in self.players.values():
            if secrets.compare_digest(p.token, token):
                return p
        return None

    def log(self, message: str) -> None:
        self.public_log.append(f"[第{self.round_number}轮] {message}" if self.round_number else message)

    # ------------------------------------------------------------------
    # 状态机
    # ------------------------------------------------------------------

    def restart(self, requester_id: int | None = None) -> None:
        """房主结束当前这局，所有人回到大厅重开。

        **保留原班人马**（编号、名字、token、AI 身份都不变）：
        token 一换，客户端存着的身份就作废了，所有人都得重新输一遍名字。
        分数、手牌、历史、流水账全部清零，等房主再点"开始游戏"。
        """
        if requester_id is not None and requester_id != self.host_id:
            raise GameError("只有房主可以结束这一局。")
        if not self.players:
            raise GameError("房间里还没有人。")

        for p in self.players.values():
            p.money = 0
            p.merit = 0
            p.rank = self.cfg.base_rank
            p.tenure = 0
            p.warnings = 0
            # 出身也要清：上一局的技能带进下一局是 bug，warnings 就这么漏过一次
            p.origin = None

        self.origin_bonus = {}
        self.family_used = set()
        self.origin_choices = {}
        self.phase = Phase.LOBBY
        self.round_number = 0
        self.hands = {}
        self.selections = {pid: Selection() for pid in self.players}
        self.ready.clear()
        self.current_event = None
        self.next_event = None
        self.last_outcome = None
        self._restored_last_result = None
        self.history.clear()
        self.public_log.clear()
        self.winners.clear()
        self.game_over_reason = ""
        self.stats.clear()
        self.ledger.clear()
        self.archive.clear()
        self.log("房主结束了上一局，大家回到大厅。")

    def start_game(
        self, requester_id: int | None = None, draft_origins: bool = False
    ) -> None:
        """开局。

        `draft_origins=True` 时先进"挑出身"阶段，每人发几个候选自己选一个；
        默认关着，这样模拟器、平衡工具和测试里想直接指定出身（或者根本不用
        出身）都不用绕路。线上由 server 打开。
        """
        if self.phase is not Phase.LOBBY:
            raise GameError("游戏已经开始了。")
        if requester_id is not None and requester_id != self.host_id:
            raise GameError("只有房主可以开始游戏。")
        if len(self.players) < self.cfg.min_players:
            raise GameError(f"至少需要 {self.cfg.min_players} 名玩家。")
        self.log("游戏开始！")
        if draft_origins and self.cfg.origin_ids():
            self._begin_origin_select()
            return
        self._begin_first_round()

    def _begin_origin_select(self) -> None:
        """给每人发一手出身候选。**放回抽样**，所以几个人可能撞同一个。"""
        pool = self.cfg.origin_ids()
        n = min(self.cfg.origin_choices_offered, len(pool))
        self.origin_choices = {
            pid: self.rng.sample(pool, n) for pid in sorted(self.players)
        }
        self.phase = Phase.ORIGIN_SELECT
        self.log("各人亮出身。")

    def choose_origin(self, player_id: int, origin_id: str) -> None:
        if self.phase is not Phase.ORIGIN_SELECT:
            raise GameError("现在不是挑出身的时候。")
        player = self.players.get(player_id)
        if player is None:
            raise GameError("你不在这局游戏里。")
        if player.origin is not None:
            raise GameError("你已经定下出身了。")
        if origin_id not in self.origin_choices.get(player_id, []):
            raise GameError("这个出身不在你能选的范围里。")
        player.origin = Origin(origin_id)
        info = self.cfg.origin(origin_id)
        self.log(f"{player.name} 出身{info['name']}（{info['skill']}）。")
        if all(p.origin is not None for p in self.players.values()):
            self._begin_first_round()

    def force_origins(self, requester_id: int | None = None) -> None:
        """房主强推：还没选的随机给一个。掉线/挂机时用，照 force_lock_all 的规矩。"""
        if requester_id is not None and requester_id != self.host_id:
            raise GameError("只有房主可以强制推进。")
        if self.phase is not Phase.ORIGIN_SELECT:
            raise GameError("现在不是挑出身的时候。")
        for pid in sorted(self.players):
            if self.players[pid].origin is None:
                self.choose_origin(pid, self.rng.choice(self.origin_choices[pid]))

    def _begin_first_round(self) -> None:
        for pid, amount in rules.apply_origin_start_bonuses(
            self.ordered_players(), self.cfg
        ).items():
            # 金额不播（钱是暗的），只说他有这门家底——出身本来就是公开的
            self.log(f"{self.players[pid].name} 家里有底子，出手比别人宽裕。")
            self.origin_bonus[pid] = amount
        self.origin_choices = {}
        self.begin_round()

    def begin_round(self) -> None:
        """发牌并进入行动选择阶段。"""
        self.round_number += 1
        self.current_event = None  # 玩家选择行动时绝对看不到本轮事件
        self.ready.clear()
        self.redraw_spent = {}
        self.redraw_count = {}
        # 工资在回合一开始就到账，这样当轮就能拿去换牌
        self.salary_paid = rules.pay_salaries(self.ordered_players(), self.cfg)
        self.hands = {
            pid: rules.deal_hand_for(self.players[pid], self.rng, self.cfg)
            for pid in sorted(self.players)
        }
        # 事件在发牌时就抽好（官二代「透风」要在选牌阶段看到），但不写进 current_event：
        # 公开状态里它仍然是"未揭晓"，直到所有人锁定。
        self.next_event = rules.pick_event(self.rng, self.cfg)
        self.selections = {pid: Selection() for pid in sorted(self.players)}
        self.phase = Phase.ACTION_SELECTION

    def select_actions(
        self, player_id: int, picks: list[dict | tuple | Action] | None
    ) -> None:
        """整批提交本轮要打的牌。picks 的每一项是 {"action": 卡名, "target": 目标id}。

        校验的是"这一整套出牌合不合法"，而不是逐张校验，因为同名牌能不能出两张
        取决于手里有没有两张。
        """
        if self.phase is not Phase.ACTION_SELECTION:
            raise GameError("现在不是选择行动的阶段。")
        sel = self.selections.get(player_id)
        if sel is None:
            raise GameError("你不在这局游戏里。")
        if sel.locked:
            raise GameError("你已经锁定了行动。")

        picks = picks or []
        # 一纸调令不算行动卡，不占出牌位
        def is_family(item) -> bool:
            card = item.card if isinstance(item, Action) else (
                item.get("action") if isinstance(item, dict) else item[0]
            )
            return card is not None and Card(card) is Card.PROMOTE_FAMILY
        if sum(1 for it in picks if not is_family(it)) > self.cfg.picks_per_round:
            raise GameError(f"每轮最多打出 {self.cfg.picks_per_round} 张牌。")

        hand = list(self.hands.get(player_id, []))
        taken: set[int] = set()
        parsed: list[Action] = []
        for item in picks:
            index = None
            if isinstance(item, Action):
                card, target_id = item.card, item.target_id
            elif isinstance(item, dict):
                card, target_id = item.get("action"), item.get("target")
                index = item.get("index")
            else:
                card, target_id = item
            if index is None and card is None:
                continue

            # 红二代「一纸调令」不在手牌里，按牌名单独处理
            if index is None and card is not None and Card(card) is Card.PROMOTE_FAMILY:
                why = self._family_card_blocker(self.players[player_id])
                if why:
                    raise GameError(why)
                if any(a.card is Card.PROMOTE_FAMILY for a in parsed):
                    raise GameError("一纸调令一轮只能打一张。")
                parsed.append(Action(card=Card.PROMOTE_FAMILY, target_id=None, value=0))
                continue

            # 优先按手牌下标定位（同名牌点数可能不同，必须指明是哪一张）；
            # 只给牌名时，取第一张还没被选走的同名牌。
            if index is not None:
                index = int(index)
                if not (0 <= index < len(hand)) or index in taken:
                    raise GameError("这张牌不在你的手牌里（或者已经选过了）。")
            else:
                card = Card(card)
                index = next(
                    (i for i, d in enumerate(hand) if d.card is card and i not in taken), None
                )
                if index is None:
                    raise GameError("这张牌不在你的手牌里（或者你只有一张）。")
            taken.add(index)
            dealt = hand[index]
            card = dealt.card

            if card.needs_target:
                if target_id is None:
                    raise GameError("这张卡需要选择一个目标。")
                target_id = int(target_id)
                if target_id == player_id:
                    raise GameError("不能选择自己作为目标。")
                if target_id not in self.players:
                    raise GameError("目标玩家不存在。")
            else:
                target_id = None
            parsed.append(Action(card=card, target_id=target_id, value=dealt.value))

        # 一纸调令永远最先结算，不管提交时排在哪
        sel.picks = sorted(parsed, key=lambda a: a.card is not Card.PROMOTE_FAMILY)

    # 兼容旧签名：一次只提交一张牌（追加到已选里）
    def select_action(
        self, player_id: int, card: Card | str | None, target_id: int | None = None
    ) -> None:
        sel = self.selections.get(player_id)
        if sel is None:
            raise GameError("你不在这局游戏里。")
        existing = [] if card is None else list(sel.picks)
        if card is not None:
            existing.append(Action(card=Card(card), target_id=target_id))
        self.select_actions(player_id, existing)

    def redraw(self, player_id: int) -> int:
        """花钱重新抽一手牌。

        底价按**当前官职**算（官越大越贵），而且同一轮里**每换一次就翻倍**：
        1 -> 2 -> 4 -> 8。想换几次换几次，但越换越肉疼——
        没有这条的话，有钱人可以在一轮里反复重抽直到摸出想要的牌。
        富二代每轮第一次免费（rules.redraw_cost_for）。
        返回这次花掉的钱。
        """
        if self.phase is not Phase.ACTION_SELECTION:
            raise GameError("现在不能换牌。")
        player = self.players.get(player_id)
        sel = self.selections.get(player_id)
        if player is None or sel is None:
            raise GameError("你不在这局游戏里。")
        if sel.locked:
            raise GameError("已经锁定了，不能再换牌。")
        used = self.redraw_count.get(player_id, 0)
        cost = rules.redraw_cost_for(player, used, self.cfg)
        if cost is None:
            raise GameError("你这一级不能换牌。")
        if player.money < cost:
            raise GameError(f"换一手牌要 {cost} 金钱，你不够。")
        player.money -= cost
        self.redraw_spent[player_id] = self.redraw_spent.get(player_id, 0) + cost
        self.redraw_count[player_id] = used + 1
        sel.picks = []
        self.hands[player_id] = rules.deal_hand_for(player, self.rng, self.cfg)
        return cost

    def _family_card_blocker(self, player: PlayerState) -> str | None:
        """这个人现在能不能打一纸调令。能打返回 None，否则返回原因。"""
        if not rules.origin_is(player, "RED", self.cfg):
            return "只有红二代才有一纸调令。"
        if player.id in self.family_used:
            return "一纸调令每局只能用一次，已经用过了。"
        if player.rank + 1 >= self.cfg.president_rank:
            return "一纸调令不能用来升国家主席。"
        return None

    def _family_card_info(self, player: PlayerState) -> dict[str, Any] | None:
        """给私密状态用：红二代才有，没用过才显示"""
        if not rules.origin_is(player, "RED", self.cfg) or player.id in self.family_used:
            return None
        why = self._family_card_blocker(player)
        return {"usable": why is None, "why": why or ""}

    def _redraw_info(self, player: PlayerState) -> dict[str, Any]:
        used = self.redraw_count.get(player.id, 0)
        cost = rules.redraw_cost_for(player, used, self.cfg)
        nxt = rules.redraw_cost_for(player, used + 1, self.cfg)
        return {
            "redraw_available": cost is not None,
            "redraw_cost": cost or 0,
            "redraw_next_cost": nxt or 0,
            "redraw_affordable": cost is not None and player.money >= cost,
        }

    def lock_action(self, player_id: int) -> None:
        if self.phase is not Phase.ACTION_SELECTION:
            raise GameError("现在不是选择行动的阶段。")
        sel = self.selections.get(player_id)
        if sel is None:
            raise GameError("你不在这局游戏里。")
        # 允许少打甚至不打：选 0~PICKS_PER_ROUND 张都行，少打就是主动弃权
        if sum(1 for a in sel.picks if a.card is not Card.PROMOTE_FAMILY) > self.cfg.picks_per_round:
            raise GameError(f"每轮最多打出 {self.cfg.picks_per_round} 张牌。")
        sel.locked = True

    def force_lock_all(self, requester_id: int | None = None) -> None:
        """房主强制推进：还没锁定的玩家按"本轮不出牌"处理。"""
        if requester_id is not None and requester_id != self.host_id:
            raise GameError("只有房主可以强制推进。")
        if self.phase is not Phase.ACTION_SELECTION:
            raise GameError("现在不是选择行动的阶段。")
        for sel in self.selections.values():
            sel.locked = True

    def all_locked(self) -> bool:
        return bool(self.selections) and all(s.locked for s in self.selections.values())

    def locked_player_ids(self) -> list[int]:
        return [pid for pid, s in sorted(self.selections.items()) if s.locked]

    def reveal_event(self) -> GameEvent:
        """所有人锁定之后才抽事件。"""
        if self.phase is not Phase.ACTION_SELECTION:
            raise GameError("现在不能揭示事件。")
        if not self.all_locked():
            raise GameError("还有玩家没有锁定行动。")
        self.current_event = self.next_event or rules.pick_event(self.rng, self.cfg)
        self.next_event = None
        self.phase = Phase.REVEAL_EVENT
        return self.current_event

    def resolve(self) -> RoundOutcome:
        """结算本轮。"""
        if self.phase is not Phase.REVEAL_EVENT:
            raise GameError("现在不能结算。")
        assert self.current_event is not None
        self.phase = Phase.RESOLUTION

        actions: dict[int, list[Action]] = {
            pid: sel.as_actions() for pid, sel in self.selections.items()
        }
        # 一纸调令打出去就算用掉，不管这轮升没升成
        for pid, acts in actions.items():
            if any(a.card is Card.PROMOTE_FAMILY for a in acts):
                self.family_used.add(pid)

        outcome = rules.resolve_round(
            players=self.ordered_players(),
            actions=actions,
            event=self.current_event,
            rng=self.rng,
            round_number=self.round_number,
            cfg=self.cfg,
            salaries=dict(self.salary_paid),
        )
        for pid, pay in self.salary_paid.items():
            if pid in outcome.outcomes:
                outcome.outcomes[pid].salary = pay
        for pid, spent in self.redraw_spent.items():
            if pid in outcome.outcomes:
                outcome.outcomes[pid].redraw_spent = spent
        for pid, count in self.redraw_count.items():
            if pid in outcome.outcomes:
                outcome.outcomes[pid].redraw_count = count
        self.last_outcome = outcome
        self.history.append(outcome)
        self._accumulate_stats(outcome)
        self.log(f"全局事件：{outcome.event.name}——{outcome.event.description}")
        for msg in outcome.public_messages:
            self.log(msg)
        for msg in outcome.wealth_broadcast:
            self.log(msg)

        self.ready.clear()
        if outcome.presidents:
            winners = list(outcome.presidents)
            tied_count = len(winners)
            if tied_count > 1 and self.cfg.president_tiebreak:
                # 同一轮多人登顶：按家底（金钱 > 政绩）分高下
                keyed = {
                    pid: (self.players[pid].money, self.players[pid].merit) for pid in winners
                }
                best = max(keyed.values())
                winners = [pid for pid in winners if keyed[pid] == best]
            self.winners = winners
            names = "、".join(self.players[pid].name for pid in self.winners)
            if tied_count > len(self.winners):
                # 主席只有一个位子（真平局除外）：落选的退回省级，钱和政绩不动。
                # 不然终局结算里会同时挂着好几个"国家主席"。
                for pid in outcome.presidents:
                    if pid not in self.winners:
                        self.players[pid].rank = self.cfg.president_rank - 1
                        if pid in outcome.outcomes:
                            outcome.outcomes[pid].rank_after = self.players[pid].rank
                candidates = "、".join(self.players[pid].name for pid in outcome.presidents)
                verdict = (
                    f"{names} 家底更厚，成功当选国家主席"
                    if len(self.winners) == 1
                    else f"{names} 家底一样厚，共同当选国家主席"
                )
                self.game_over_reason = (
                    f"国家主席最大候选人为：{candidates}，最终还是因为{verdict}。"
                )
                outcome.presidents = list(self.winners)
            else:
                self.game_over_reason = f"{names} 登上国家主席之位，游戏结束！"
            self.log(self.game_over_reason)
            self.phase = Phase.GAME_OVER
        elif self.round_number >= self.cfg.max_rounds:
            winners = rules.final_winners(self.ordered_players(), self.cfg)
            self.winners = [p.id for p in winners]
            names = "、".join(p.name for p in winners)
            if len(winners) > 1:
                self.game_over_reason = f"{self.cfg.max_rounds} 轮结束，{names} 各项指标完全相同，平局。"
            else:
                top = self.players[self.winners[0]]
                self.game_over_reason = (
                    f"{self.cfg.max_rounds} 轮结束，无人登顶；"
                    f"{names} 家底最厚（{self.cfg.rank_name(top.rank)}），获胜！"
                )
            self.log(self.game_over_reason)
            self.phase = Phase.GAME_OVER
        else:
            self.phase = Phase.ROUND_RESULT
        return outcome

    # ------------------------------------------------------------------
    # 终局复盘
    # ------------------------------------------------------------------

    _STAT_KEYS = (
        "merit_earned", "money_earned", "promotions", "demoted",
        "merit_stolen_from_me", "money_confiscated", "tenure_wrecked",
        "merit_i_stole", "money_i_took", "wasted_report", "wasted_attack",
        "wasted_promotion_card", "interference_turns", "production_turns",
    )

    def _accumulate_stats(self, outcome: RoundOutcome) -> None:
        rnd = outcome.round_number
        self.archive.append({
            "round": rnd,
            "event": outcome.event.name,
            "players": [
                {
                    "player_id": pid,
                    "cards": [c.value for c in o.cards],
                    "targets": list(o.targets),
                    "money_after": o.money_after,
                    "merit_after": o.merit_after,
                    "rank_after": o.rank_after,
                    "promotion": o.promotion.value,
                    "demotion": o.demotion.value,
                }
                for pid, o in sorted(outcome.outcomes.items())
            ],
        })
        for pid, o in outcome.outcomes.items():
            rows = o.ledger_lines(self.cfg.rank_name(o.rank_after))
            if rows:
                self.ledger.setdefault(pid, []).append(
                    {"round": rnd, "rows": rows,
                     "money_after": o.money_after, "merit_after": o.merit_after}
                )
            st = self.stats.setdefault(pid, {k: 0 for k in self._STAT_KEYS})
            st["merit_earned"] += o.merit_gained
            st["money_earned"] += o.money_gained
            st["merit_stolen_from_me"] += o.merit_stolen_by_attackers
            st["money_confiscated"] += o.money_confiscated
            st["tenure_wrecked"] += o.tenure_reset_by_attack
            st["merit_i_stole"] += o.merit_from_attacks
            st["money_i_took"] += o.money_from_reports
            for c in o.cards:
                if c.is_production:
                    st["production_turns"] += 1
                elif c in (Card.REPORT, Card.ATTACK):
                    st["interference_turns"] += 1
            if o.promotion.value != "NONE":
                st["promotions"] += 1
            if o.demotion.value != "NONE":
                st["demoted"] += 1
            if o.promotion_card_played and o.promotion.value == "NONE":
                st["wasted_promotion_card"] += 1
            # 把人打回基层本身就是战果，哪怕对方已经没钱可抄
            if Card.REPORT in o.cards and o.reports_landed <= 0:
                st["wasted_report"] += 1
            # 拦下别人的晋升是 denial 模式下最有价值的一刀，但攻击者拿不到政绩，
            # 所以"白费"要看有没有打中，不能只看收益
            if Card.ATTACK in o.cards and o.attacks_landed <= 0:
                st["wasted_attack"] += 1

    def final_standing(self) -> list[int]:
        """终局名次（高到低）。

        口径必须和真正的胜负判定一致（见 FINAL_RANKING_KEYS，默认 官职 > 金钱 > 政绩）。
        已判定的赢家（比如当上国家主席的人）直接排在最前面，
        否则会出现"结算说你赢了、复盘说你垫底"这种自相矛盾。
        """
        ordered = [p.id for p in rules.final_ranking(self.ordered_players(), self.cfg)]
        winners = [pid for pid in ordered if pid in self.winners]
        return winners + [pid for pid in ordered if pid not in self.winners]

    def _pending_ledger(self, player_id: int) -> list[dict[str, Any]]:
        """本轮**还没结算**、但钱已经实际动了的那几笔。

        工资在回合开头就到账、换牌当场扣钱，可流水账要等结算才生成——
        中间这段时间玩家会看到余额变了、账上却找不到这一笔。
        """
        if self.phase not in (Phase.ACTION_SELECTION, Phase.REVEAL_EVENT):
            return []
        player = self.players.get(player_id)
        if player is None:
            return []
        rows: list[dict[str, Any]] = []
        pay = self.salary_paid.get(player_id, 0)
        if pay:
            rows.append({"label": "合法工资", "money": pay, "merit": 0})
        spent = self.redraw_spent.get(player_id, 0)
        if spent:
            rows.append({"label": "重新抽牌", "money": -spent, "merit": 0})
        if not rows:
            return []
        return [{
            "round": self.round_number,
            "rows": rows,
            "pending": True,  # UI 标成"本轮进行中"
            "money_after": player.money,
            "merit_after": player.merit,
        }]

    def full_reveal(self) -> dict[str, Any] | None:
        """终局全揭示：把整局的底牌摊开。

        只在游戏结束后提供——此时再没有可保护的秘密，摊开来复盘反而是乐趣所在。
        """
        if not self.is_over:
            return None
        order = self.final_standing()
        return {
            "standing": [
                {
                    "place": i + 1,
                    "player_id": pid,
                    "name": self.players[pid].name,
                    "is_ai": self.players[pid].is_ai,
                    "rank": self.players[pid].rank,
                    "rank_name": self.cfg.rank_name(self.players[pid].rank),
                    "money": self.players[pid].money,   # 终局才公开
                    "merit": self.players[pid].merit,
                    "won": pid in self.winners,
                    "stats": self.stats.get(pid, {}),
                }
                for i, pid in enumerate(order)
            ],
            "rounds": list(self.archive),
            "names": {str(pid): p.name for pid, p in self.players.items()},
        }

    def postmortem(self, player_id: int) -> dict[str, Any] | None:
        """给这名玩家的私人复盘。只在终局提供，只发给他本人。

        里面含有金钱相关的数字，所以绝对不能进公开广播。
        """
        if not self.is_over or player_id not in self.players:
            return None
        st = self.stats.get(player_id)
        if st is None:
            return None
        order = self.final_standing()
        place = order.index(player_id) + 1
        others = [p for p in self.stats if p != player_id]

        def avg(key: str) -> float:
            if not others:
                return 0.0
            return sum(self.stats[p][key] for p in others) / len(others)

        causes = []
        for key, label in (
            ("merit_stolen_from_me", "政绩被人抢走"),
            ("money_confiscated", "赃款被举报没收"),
            ("tenure_wrecked", "资历被搅黄"),
            ("wasted_report", "举报扑空"),
            ("wasted_attack", "攻击扑空"),
            ("wasted_promotion_card", "晋升卡没用上"),
            ("interference_turns", "回合花在干扰上"),
        ):
            mine, other = st[key], avg(key)
            if mine > other:
                causes.append(
                    {"label": label, "mine": mine, "others_avg": round(other, 1),
                     "excess": round(mine - other, 1)}
                )
        causes.sort(key=lambda c: -c["excess"])

        return {
            "place": place,
            "total": len(order),
            "is_last": place == len(order),
            "promotions": st["promotions"],
            "others_avg_promotions": round(avg("promotions"), 1),
            "production_turns": st["production_turns"],
            "others_avg_production": round(avg("production_turns"), 1),
            "top_causes": causes[:3],
            "stats": dict(st),
        }

    def mark_ready(self, player_id: int) -> None:
        if self.phase is not Phase.ROUND_RESULT:
            raise GameError("现在不需要确认。")
        if player_id not in self.players:
            raise GameError("你不在这局游戏里。")
        self.ready.add(player_id)

    def everyone_ready(self) -> bool:
        """只统计在线玩家，掉线的人不会卡住整局游戏。"""
        if self.phase is not Phase.ROUND_RESULT:
            return False
        online = {pid for pid, p in self.players.items() if p.connected}
        if not online:
            return False
        return online.issubset(self.ready)

    def advance_round(self) -> None:
        if self.phase is not Phase.ROUND_RESULT:
            raise GameError("现在不能进入下一轮。")
        self.begin_round()

    @property
    def is_over(self) -> bool:
        return self.phase is Phase.GAME_OVER

    # ------------------------------------------------------------------
    # 对外状态（严格区分公开/私密）
    # ------------------------------------------------------------------

    def public_state(self) -> dict[str, Any]:
        """任何人都可以看到的状态。

        绝不包含：money / hand / selected_action / selected_target /
        举报者身份 / 攻击者身份。
        """
        event = None
        if self.phase in (Phase.REVEAL_EVENT, Phase.RESOLUTION, Phase.ROUND_RESULT, Phase.GAME_OVER):
            event = self.current_event.public_view() if self.current_event else None

        return {
            "game_id": self.game_id,
            "phase": self.phase.value,
            "round": self.round_number,
            # 六张出身的名字/技能/说明。出身是公开信息，前端要拿它渲染徽章，
            # 数值和文案都只在 config 里定义一处
            "origins": [dict(o) for o in self.cfg.origin_definitions]
            if self.cfg.origins_enabled else [],
            # 谁还没挑出身（挑出身阶段用，和 locked_players 一个意思）
            "origin_pending": sorted(
                pid for pid, pl in self.players.items() if pl.origin is None
            ) if self.phase is Phase.ORIGIN_SELECT else [],
            "max_rounds": self.cfg.max_rounds,
            "host_id": self.host_id,
            "min_players": self.cfg.min_players,
            "max_players": self.cfg.max_players,
            "rank_names": list(self.cfg.rank_names),
            "players": [
                p.public_view(self.cfg.rank_name(p.rank)) for p in self.ordered_players()
            ],
            "current_event": event,
            "locked_players": self.locked_player_ids(),
            "ready_players": sorted(self.ready),
            "last_result": (
                self.last_outcome.public_view()
                if self.last_outcome
                else self._restored_last_result
            ),
            "public_messages": self.public_log[-60:],
            "winners": list(self.winners),
            "game_over_reason": self.game_over_reason,
            "reveal": self.full_reveal(),
            "promotion_costs": [
                {
                    "from": self.cfg.rank_name(r),
                    "to": self.cfg.rank_name(r + 1),
                    "money": self.cfg.promotion_money_costs[r],
                    "merit": self.cfg.promotion_merit_costs[r],
                }
                for r in range(self.cfg.president_rank)
            ],
            "tenure_required": self.cfg.tenure_required,
            "merit_overflow_divisor": self.cfg.merit_overflow_divisor,
            "warnings_before_demotion": self.cfg.warnings_before_demotion,
            "picks_per_round": self.cfg.picks_per_round,
            "hand_size": self.cfg.hand_size,
        }

    def private_state(self, player_id: int) -> dict[str, Any]:
        """只发给 player_id 本人。"""
        player = self.players.get(player_id)
        if player is None:
            raise GameError("你不在这局游戏里。")
        sel = self.selections.get(player_id, Selection())
        origin_id = player.origin.value if player.origin else None
        private_result = None
        if self.last_outcome is not None:
            o = self.last_outcome.outcomes.get(player_id)
            if o is not None:
                private_result = o.private_view()
        # 官二代「透风」：选牌阶段就知道本轮事件。只进他自己的私密状态，别人连这个键都没有。
        tipoff = {}
        if (
            self.phase is Phase.ACTION_SELECTION
            and self.next_event is not None
            and rules.origin_is(player, "OFFICIAL", self.cfg)
        ):
            tipoff = {"tipoff_event": self.next_event.public_view()}
        family = self._family_card_info(player)
        return {
            **tipoff,
            # 红二代「一纸调令」：None = 没有这张卡（不是红二代或者已经用掉）
            "family_card": family,
            "player_id": player.id,
            "name": player.name,
            "money": player.money,
            "merit": player.merit,
            "rank": player.rank,
            "rank_name": self.cfg.rank_name(player.rank),
            "tenure": player.tenure,
            "origin": player.origin.value if player.origin else None,
            # 这个玩家**自己的**晋升门槛阶梯。出身会改门槛（官二代的政绩打折），
            # 前端的规则表要是直接用 /api/config 里那份通用的，他看到的
            # "还差多少"就和结算对不上——UI 说要 15、实际只要 10。
            "promotion_money_costs": [
                rules.money_cost_at(r, origin_id, self.cfg)
                for r in range(self.cfg.president_rank)
            ],
            "promotion_merit_costs": [
                rules.merit_cost_at(r, origin_id, self.cfg)
                for r in range(self.cfg.president_rank)
            ],
            # 挑出身阶段自己手上的候选（带上文案，前端不用再查一遍表）
            "origin_choices": [
                dict(self.cfg.origin(oid) or {})
                for oid in self.origin_choices.get(player_id, [])
            ],
            "hand": [d.view() for d in self.hands.get(player_id, [])],
            "picks": [
                {"action": a.card.value, "target": a.target_id, "value": a.value}
                for a in sel.picks
            ],
            "picks_per_round": self.cfg.picks_per_round,
            "locked": sel.locked,
            "is_host": player.id == self.host_id,
            "private_result": private_result,
            "next_promotion": self._next_promotion_info(player),
            "card_preview": self._card_preview(player),
            "warnings": player.warnings,
            # 下一次换牌的价钱（本轮换过几次就翻几次倍；富二代第一次是 0）
            # 和再下一次的价钱（确认框里提示用）。这一级不能换牌时 available=False。
            **self._redraw_info(player),
            "redraws_used_this_round": self.redraw_count.get(player_id, 0),
            "redraw_spent_this_round": self.redraw_spent.get(player_id, 0),
            "ledger": self.ledger.get(player_id, []) + self._pending_ledger(player_id),
            "postmortem": self.postmortem(player_id),
        }

    def _card_preview(self, player: PlayerState) -> dict[str, Any]:
        """每张牌打出去大概是什么效果，按这个玩家当前的官职折算。

        全部由公开规则 + 他自己的私密状态推出来，不含任何别人的信息。
        """
        cfg = self.cfg
        rank = player.rank
        # 走 rules 那两个认得出身的门槛函数，不然官二代看到的 UI 会和结算对不上
        mc, tc = rules.money_cost_for(player, cfg), rules.merit_cost_for(player, cfg)

        def span(dist):
            """牌面的最小/最大点数。UI 上写区间比写期望值实在。"""
            values = [v for v, _ in dist]
            return min(values), max(values)

        wlo, whi = span(cfg.work_card_distribution)
        clo, chi = span(cfg.corrupt_card_distribution)
        glo, ghi = span(cfg.graft_card_distribution)

        work_lo = rules.work_merit(wlo, rank, None, cfg)
        work_hi = rules.work_merit(whi, rank, None, cfg)
        corrupt_lo = rules.corrupt_money(clo, rank, None, cfg)
        corrupt_hi = rules.corrupt_money(chi, rank, None, cfg)
        graft_money_lo = rules.corrupt_money(glo, rank, None, cfg)
        graft_money_hi = rules.corrupt_money(ghi, rank, None, cfg)
        graft_merit_lo = rules.graft_merit(glo, rank, None, cfg)
        graft_merit_hi = rules.graft_merit(ghi, rank, None, cfg)

        line = cfg.major_corruption_threshold
        # 我去打别人、对方无所事事时扣他多少（按**我**看不到的对方官职算不了，
        # 这里给的是"打同级的人"的参考值，UI 只用来说明量级）
        idle_penalty = rules.work_merit(cfg.attack_merit_penalty, rank, None, cfg)

        def risk(lo: int, hi: int) -> str:
            """这一笔会不会踩到重大贪腐线（踩到 = 被举报直接打回基层）。"""
            if lo >= line:
                return "major"      # 必然重大
            if hi >= line:
                return "maybe"      # 运气不好会重大
            return "minor"          # 只会降一级

        def promo(uses_merit: bool, uses_money: bool) -> dict[str, Any]:
            ok_merit = uses_merit and tc is not None and player.merit >= tc
            ok_money = uses_money and mc is not None and player.money >= mc
            gaps = []
            if uses_merit and tc is not None and not ok_merit:
                gaps.append(f"政绩还差 {tc - player.merit}")
            if uses_money and mc is not None and not ok_money:
                gaps.append(f"金钱还差 {mc - player.money}")
            return {"usable": bool(ok_merit or ok_money), "why": "、".join(gaps)}

        grinder = rules.origin_is(player, "GRINDER", cfg)
        return {
            "WORK": {
                "merit_lo": work_lo,
                "merit_hi": work_hi,
                # 卷王「加班」：同一轮打两张，这两张的政绩最后 ×几（不是卷王就是 0）
                "overtime": cfg.origin_grinder_overtime_multiplier if grinder else 0,
            },
            "CORRUPT": {
                "money_lo": corrupt_lo,
                "money_hi": corrupt_hi,
                "risk": risk(corrupt_lo, corrupt_hi),
            },
            "GRAFT": {
                "money_lo": graft_money_lo,
                "money_hi": graft_money_hi,
                "merit_lo": graft_merit_lo,
                "merit_hi": graft_merit_hi,
                "risk": risk(graft_money_lo, graft_money_hi),
            },
            "rank_multiplier": str(cfg.work_multiplier(rank)),
            "idle_penalty": idle_penalty,
            "mult_num": cfg.work_multiplier(rank).numerator,
            "mult_den": cfg.work_multiplier(rank).denominator,
            "PROMOTE_MERIT": promo(True, False),
            "PROMOTE_MONEY": promo(False, True),
            "PROMOTE_ANY": promo(True, True),
            "major_corruption_threshold": line,
        }

    def _next_promotion_info(self, player: PlayerState) -> dict[str, Any] | None:
        money_cost = rules.money_cost_for(player, self.cfg)
        merit_cost = rules.merit_cost_for(player, self.cfg)
        if money_cost is None or merit_cost is None:
            return None
        needs_both = self.cfg.needs_both(player.rank)
        tenure_ceiling = (
            self.cfg.president_rank
            if self.cfg.tenure_can_reach_president
            else self.cfg.president_rank - 1
        )
        return {
            "to": self.cfg.rank_name(player.rank + 1),
            "money": money_cost,
            "merit": merit_cost,
            "money_gap": max(0, money_cost - player.money),
            "merit_gap": max(0, merit_cost - player.merit),
            "tenure_gap": max(0, self.cfg.tenure_required - player.tenure),
            "needs_both": needs_both,
            "tenure_works": player.rank < tenure_ceiling,
            "salary": self.cfg.salary(player.rank),
        }

    # ------------------------------------------------------------------
    # 持久化快照
    # ------------------------------------------------------------------

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "phase": self.phase.value,
            "round_number": self.round_number,
            "next_player_id": self._next_player_id,
            "current_event_id": self.current_event.id if self.current_event else None,
            # 已经抽好还没揭晓的事件。不存的话重启后会重抽，官二代刚听到的风声就变了
            "next_event_id": self.next_event.id if self.next_event else None,
            "family_used": sorted(self.family_used),
            "winners": list(self.winners),
            "game_over_reason": self.game_over_reason,
            "reveal": self.full_reveal(),
            "public_log": list(self.public_log),
            "players": [
                {
                    "id": p.id,
                    "name": p.name,
                    "money": p.money,
                    "merit": p.merit,
                    "rank": p.rank,
                    "tenure": p.tenure,
                    "warnings": p.warnings,
                    "origin": p.origin.value if p.origin else None,
                    "token": p.token,
                    "is_ai": p.is_ai,
                }
                for p in self.ordered_players()
            ],
            "hands": {
                str(pid): [d.view() for d in cards] for pid, cards in self.hands.items()
            },
            "selections": {
                str(pid): {
                    "picks": [
                        {"card": a.card.value, "target_id": a.target_id, "value": a.value}
                        for a in s.picks
                    ],
                    "locked": s.locked,
                }
                for pid, s in self.selections.items()
            },
            "ready": sorted(self.ready),
            "results": [o.public_view() for o in self.history],
            # 复盘用的累计数据。不存的话重启一次，流水账、全场揭晓、
            # 终局统计就全空了——而这些正是玩家最后看到的东西。
            "ledger": {str(pid): rows for pid, rows in self.ledger.items()},
            "archive": list(self.archive),
            "stats": {str(pid): dict(st) for pid, st in self.stats.items()},
            # 本回合内的计数。不存的话，中途重启会把换牌价重置回底价，
            # 而且这一轮的工资/换牌开销会从流水账里消失。
            "salary_paid": {str(pid): v for pid, v in self.salary_paid.items()},
            "redraw_spent": {str(pid): v for pid, v in self.redraw_spent.items()},
            "redraw_count": {str(pid): v for pid, v in self.redraw_count.items()},
            # 挑出身阶段的候选。不存的话，选到一半重启页面候选就没了
            "origin_choices": {str(pid): list(v) for pid, v in self.origin_choices.items()},
            "origin_bonus": {str(pid): v for pid, v in self.origin_bonus.items()},
        }

    @classmethod
    def from_snapshot(
        cls, data: dict[str, Any], cfg: Config = DEFAULT_CONFIG, rng: Any = None
    ) -> "Game":
        game = cls(game_id=data["game_id"], cfg=cfg, rng=rng or secrets.SystemRandom())
        game.phase = Phase(data["phase"])
        game.round_number = int(data["round_number"])
        game._next_player_id = int(data["next_player_id"])
        game.winners = list(data.get("winners") or [])
        game.game_over_reason = data.get("game_over_reason", "")
        game.public_log = list(data.get("public_log") or [])
        for row in data.get("players", []):
            game.players[int(row["id"])] = PlayerState(
                id=int(row["id"]),
                name=row["name"],
                money=int(row["money"]),
                merit=int(row["merit"]),
                rank=int(row["rank"]),
                tenure=int(row["tenure"]),
                warnings=int(row.get("warnings") or 0),
                origin=Origin(row["origin"]) if row.get("origin") else None,
                token=row["token"],
                connected=bool(row.get("is_ai")),  # AI 永远在线
                is_ai=bool(row.get("is_ai")),
            )
        game.hands = {
            int(pid): [
                DealtCard(card=Card(c["card"]), value=int(c.get("value", 0)))
                if isinstance(c, dict)
                else DealtCard(card=Card(c))
                for c in cards
            ]
            for pid, cards in (data.get("hands") or {}).items()
        }
        game.selections = {
            int(pid): Selection(
                picks=[
                    Action(card=Card(a["card"]), target_id=a.get("target_id"),
                           value=int(a.get("value", 0)))
                    for a in (s.get("picks") or [])
                ],
                locked=bool(s.get("locked")),
            )
            for pid, s in (data.get("selections") or {}).items()
        }
        for pid in game.players:
            game.selections.setdefault(pid, Selection())
        game.ready = set(data.get("ready") or [])
        game.ledger = {
            int(pid): list(rows) for pid, rows in (data.get("ledger") or {}).items()
        }
        game.archive = list(data.get("archive") or [])
        for attr in ("salary_paid", "redraw_spent", "redraw_count", "origin_bonus"):
            setattr(game, attr, {
                int(pid): int(v) for pid, v in (data.get(attr) or {}).items()
            })
        game.origin_choices = {
            int(pid): list(v) for pid, v in (data.get("origin_choices") or {}).items()
        }
        game.stats = {
            int(pid): dict(st) for pid, st in (data.get("stats") or {}).items()
        }
        results = data.get("results") or []
        game._restored_last_result = results[-1] if results else None
        event_id = data.get("current_event_id")
        if event_id:
            game.current_event = rules.event_by_id(event_id, cfg)
        game.family_used = {int(pid) for pid in (data.get("family_used") or [])}
        next_id = data.get("next_event_id")
        if next_id:
            game.next_event = rules.event_by_id(next_id, cfg)

        # 结算中途崩溃时，把阶段退回到可以继续操作的地方
        if game.phase in (Phase.RESOLUTION,):
            game.phase = Phase.REVEAL_EVENT
        return game
