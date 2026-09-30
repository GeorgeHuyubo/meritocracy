"""纯数据模型：枚举、玩家状态、单轮行动、结算产物。

这里不放任何规则计算（规则在 rules.py），也不放任何 Web/IO 代码，
这样 simulator.py 和 server.py 才能共用同一套模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Card(str, Enum):
    """行动卡。

    四张功能牌 + 三张晋升卡。晋升**必须**打出晋升卡（工龄晋升除外，那条仍然自动）。
    """

    WORK = "WORK"  # 埋头工作：赚政绩
    CORRUPT = "CORRUPT"  # 中饱私囊：赚金钱
    GRAFT = "GRAFT"  # 以权谋私：钱少一些，但顺带带一点政绩（钱仍算贪污，会被举报）
    REPORT = "REPORT"  # 匿名举报
    ATTACK = "ATTACK"  # 政治攻击
    PROMOTE_MERIT = "PROMOTE_MERIT"  # 政绩升职
    PROMOTE_MONEY = "PROMOTE_MONEY"  # 贿赂升职
    PROMOTE_ANY = "PROMOTE_ANY"  # 通用升职：政绩或金钱，哪条够走哪条

    @property
    def needs_target(self) -> bool:
        return self in (Card.REPORT, Card.ATTACK)

    @property
    def is_promotion(self) -> bool:
        return self in (Card.PROMOTE_MERIT, Card.PROMOTE_MONEY, Card.PROMOTE_ANY)

    @property
    def is_production(self) -> bool:
        return self in (Card.WORK, Card.CORRUPT, Card.GRAFT)

    @property
    def yields_corruption(self) -> bool:
        """赚到的钱算不算"本轮贪污额"（举报和攻击都看这个）。"""
        return self in (Card.CORRUPT, Card.GRAFT)

    @property
    def can_use_merit(self) -> bool:
        return self in (Card.PROMOTE_MERIT, Card.PROMOTE_ANY)

    @property
    def can_use_money(self) -> bool:
        return self in (Card.PROMOTE_MONEY, Card.PROMOTE_ANY)


class Origin(str, Enum):
    """出身。**公开信息**——写在记分板上，人人看得见。

    值和 config.ORIGIN_DEFINITIONS 里的 id 一一对应；名字、技能名、说明文案
    都在 config 里，这里只保留枚举本身，免得数值和文案散到两个地方。
    """

    RICH = "RICH"  # 富二代 · 老钱
    OFFICIAL = "OFFICIAL"  # 官二代 · 提携
    RED = "RED"  # 红二代 · 开后门
    PEASANT = "PEASANT"  # 贫农 · 政治正确
    GRINDER = "GRINDER"  # 小镇做题家·技术员 · 卷王
    ACCOUNTANT = "ACCOUNTANT"  # 小镇做题家·会计 · 做账


class Phase(str, Enum):
    """服务器权威状态机。客户端不允许自己推断阶段。"""

    LOBBY = "LOBBY"
    ACTION_SELECTION = "ACTION_SELECTION"
    REVEAL_EVENT = "REVEAL_EVENT"
    RESOLUTION = "RESOLUTION"
    ROUND_RESULT = "ROUND_RESULT"
    GAME_OVER = "GAME_OVER"


class PromotionKind(str, Enum):
    NONE = "NONE"
    MERIT = "MERIT"  # 政绩晋升
    MONEY = "MONEY"  # 金钱晋升
    BOTH = "BOTH"  # 双条件晋升（政绩和金钱一起消耗）
    TENURE = "TENURE"  # 工龄晋升


class DemotionKind(str, Enum):
    NONE = "NONE"
    MINOR = "MINOR"  # 小额贪污被举报，降一级
    MAJOR = "MAJOR"  # 重大贪腐，打回基层


@dataclass
class PlayerState:
    """玩家状态。

    money 是私密数据，merit / rank / tenure 是公开数据。
    token 只用于刷新页面后恢复身份，绝对不进入任何广播。
    """

    id: int
    name: str
    money: int = 0
    merit: int = 0
    rank: int = 0
    tenure: int = 0
    # 累计的严重警告。攒够 WARNINGS_BEFORE_DEMOTION 次就降一级、清空重来。
    # 这是公开信息：官场上谁挨过处分，大家都知道。
    warnings: int = 0
    # 出身。公开信息，进 public_view。None = 这局没开出身，或者还没选。
    origin: Origin | None = None
    token: str = ""
    connected: bool = False
    is_ai: bool = False  # 由服务器上的思考型 AI 代打

    def clone(self) -> "PlayerState":
        return PlayerState(
            id=self.id,
            name=self.name,
            money=self.money,
            merit=self.merit,
            rank=self.rank,
            tenure=self.tenure,
            warnings=self.warnings,
            origin=self.origin,
            token=self.token,
            connected=self.connected,
            is_ai=self.is_ai,
        )

    def public_view(self, rank_name: str) -> dict[str, Any]:
        """公开状态：绝不包含 money / hand / action / target。"""
        return {
            "id": self.id,
            "name": self.name,
            "rank": self.rank,
            "rank_name": rank_name,
            "merit": self.merit,
            "tenure": self.tenure,
            "warnings": self.warnings,
            "origin": self.origin.value if self.origin else None,
            "connected": self.connected,
            "is_ai": self.is_ai,
        }


@dataclass(frozen=True)
class DealtCard:
    """手牌里的一张具体的牌：牌型 + **发牌时就摇好的点数**。

    点数在发牌那一刻就定下来，玩家选牌之前就能看见自己这张 WORK 到底值几点。
    需要保密的是"本轮全局事件"，不是"你自己手上这张牌有多大"。
    """

    card: Card
    value: int = 0  # WORK / CORRUPT / GRAFT 的牌面点数；其他牌恒为 0

    def view(self) -> dict[str, Any]:
        return {"card": self.card.value, "value": self.value}


@dataclass(frozen=True)
class Action:
    """结算时喂给 rules.resolve_round 的一个行动。"""

    card: Card
    target_id: int | None = None
    value: int = 0  # 发牌时摇好的点数；<=0 表示让结算时临时摇（老测试走这条）


@dataclass
class Selection:
    """玩家本轮的秘密选择：一次出 PICKS_PER_ROUND 张牌。"""

    picks: list[Action] = field(default_factory=list)
    locked: bool = False

    def as_actions(self) -> list[Action]:
        return list(self.picks)

    def cards(self) -> list[Card]:
        return [a.card for a in self.picks]


@dataclass(frozen=True)
class GameEvent:
    id: str
    name: str
    description: str
    effects: dict[str, Any]

    def public_view(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "description": self.description}

    def flag(self, key: str) -> bool:
        return bool(self.effects.get(key, False))


@dataclass
class PlayerRoundOutcome:
    """单个玩家在某一轮的结算结果。

    公开部分进 public messages，私密部分只发给本人。
    """

    player_id: int
    cards: list[Card] = field(default_factory=list)  # 本轮打出的全部牌
    targets: list[int | None] = field(default_factory=list)

    base_values: list[int] = field(default_factory=list)  # 各张生产牌的原始点数
    salary: int = 0  # 本轮到账的合法工资
    merit_gained: int = 0
    money_gained: int = 0
    corrupt_amount: int = 0  # 本轮"最终"贪污金额，举报与财富广播都看它

    attacked: bool = False  # 被有效的政治攻击命中
    attacked_by: list[int] = field(default_factory=list)  # 谁打的（明攻击才公开）
    attack_merit_loss: int = 0  # 因被攻击损失的政绩（扣罚 or 被抢走）
    merit_stolen_by_attackers: int = 0  # 其中被攻击者抢走的部分
    merit_wiped_by_attack: int = 0  # 因晋升被拦而作废的政绩
    merit_from_attacks: int = 0  # 作为攻击者抢到的政绩
    attacks_landed: int = 0  # 作为攻击者，有几刀真的起了作用（拦下晋升也算）
    reports_landed: int = 0  # 作为举报者，有几份举报查实了（哪怕没抄到钱）
    tenure_reset_by_attack: int = 0  # 被攻击搅黄的工龄
    merit_promotion_blocked: bool = False
    # 本轮确实打出了"够得着的政绩升职"（不管后来有没有被拦、有没有被延后）。
    # 用来判断"这一轮有没有在干正事"，必须和结算顺序无关。
    tried_merit_promotion: bool = False

    reported: bool = False  # 被举报（不论是否有效）
    # 举报来源要分清楚：真人打的举报牌 vs 全局事件（反腐风暴/大筛查）扫到的。
    # 效果一样，但公报里必须说明白——不然玩家会以为桌上有人在盯着自己。
    report_count_players: int = 0  # 其中来自真人举报牌的份数
    report_from_event: str = ""  # 触发查办的全局事件名（没有就是空串）
    report_count: int = 0  # 本轮收到几份举报（含"中央反腐大筛查"事件的那一份）
    report_effective: bool = False  # 举报是否造成后果
    promotion_frozen_by_report: bool = False  # 停职待查：本轮晋升被举报冻结
    pending_bribe: int = 0  # 本轮摆上桌、但还没兑现的行贿金额
    bribe_lost: int = 0  # 拿钱买晋升却被举报：礼送出去了，事没办成
    warnings_issued: int = 0  # 本轮新记的严重警告
    warnings_after: int = 0  # 记完之后累计几次（降级后已清空）
    promotion_merit_decay: int = 0  # 晋升后政绩打折掉的量（贿赂升/熬工龄也有）
    redraw_spent: int = 0  # 本轮花在重新抽牌上的钱
    demotion: DemotionKind = DemotionKind.NONE
    money_confiscated: int = 0  # 被举报/被攻击起获而没收的赃款
    hush_money_paid: int = 0  # 被攻击后为压事花掉的打点费
    money_from_reports: int = 0  # 作为举报人分到的赃款

    promotion: PromotionKind = PromotionKind.NONE
    promotion_card_played: bool = False  # 本轮打了晋升卡（不管成没成）
    promoted_before_production: bool = False  # 在动手干活之前就先把官升了
    # 晋升消耗的明细，用来在 UI 上把"钱去哪了"讲清楚
    promotion_money_cost: int = 0
    promotion_merit_cost: int = 0
    money_before_promotion: int = 0
    merit_before_promotion: int = 0
    promotion_blocked_by_attack_report: bool = False  # 规则书第 16 节的特殊情况

    rank_before: int = 0
    rank_after: int = 0
    tenure_after: int = 0
    money_after: int = 0
    merit_after: int = 0

    # 会计「做账」当场洗白成合法收入的金额。只挡没收，不影响分赃池。
    laundered: int = 0
    # 红二代「开后门」这一轮已经连升过一次了（防止一轮升三级）
    origin_double_promoted: bool = False

    # 私密提示（只发给本人），例如"你的举报无效"
    private_notes: list[str] = field(default_factory=list)

    @property
    def net_corrupt_gain(self) -> int:
        """本轮贪污**真正落进自己口袋**的部分。

        财富广播看的是这个，不是毛收入：钱当场被没收/抄走的人，
        坊间不会传他住上洋房。被抓与否仍然按毛收入（corrupt_amount）判定——
        那是"你干没干过"，和"你有没有留住"是两回事。
        """
        return max(0, self.corrupt_amount - self.money_confiscated - self.hush_money_paid)

    def ledger_lines(self, rank_name: str) -> list[dict[str, Any]]:
        """把这一轮的结算摊成流水账：每一笔钱、每一点政绩是怎么来的、怎么没的。

        含金钱数字，只能进 private state。
        """
        rows: list[dict[str, Any]] = []

        def add(label: str, money: int = 0, merit: int = 0) -> None:
            if money or merit:
                rows.append({"label": label, "money": money, "merit": merit})

        add("合法工资", money=self.salary)
        if self.merit_gained:
            add("干活所得", merit=self.merit_gained)
        if self.money_gained:
            add("贪污进账", money=self.money_gained)
        if self.merit_from_attacks:
            add("揭发/抢功记功", merit=self.merit_from_attacks)
        if self.money_from_reports:
            add("举报分赃", money=self.money_from_reports)
        if self.attack_merit_loss:
            add("被政治攻击", merit=-self.attack_merit_loss)
        if self.hush_money_paid:
            add("上下打点压事", money=-self.hush_money_paid)
        if self.money_confiscated:
            add("赃款被没收", money=-self.money_confiscated)
        if self.bribe_lost:
            add("行贿的钱打了水漂", money=-self.bribe_lost)
        if self.promotion_money_cost or self.promotion_merit_cost:
            add(
                f"晋升为{rank_name}",
                money=-self.promotion_money_cost,
                merit=-self.promotion_merit_cost,
            )
        # 晋升后的 /5 衰减单独列一行，否则账对不上。
        # 贿赂升职和工龄晋升没有政绩成本，但照样打折，所以不能只看 merit_cost。
        if self.promotion_merit_decay:
            add("晋升后政绩打折", merit=-self.promotion_merit_decay)
        if self.redraw_spent:
            add("重新抽牌", money=-self.redraw_spent)
        return rows

    def public_facts(self) -> dict[str, Any]:
        """本轮结算里**任何人都能看到**的那部分事实。

        这些信息本来就写在公开通报里（"X 遭到政治攻击"、"X 因贪腐问题被匿名举报，
        降为 Y"、"X 政绩卓著，晋升为 Y"），这里只是把它变成结构化字段，
        方便前端渲染和 AI 推理，不额外泄露任何东西。

        绝不包含：money / corrupt_amount / 没收与分赃金额 / 出的牌 / 目标 /
        举报者与攻击者的身份。
        """
        return {
            "player_id": self.player_id,
            "attacked": self.attacked,
            # 明攻击：公报里本来就点名了，所以攻击者身份是公开信息。
            # （匿名举报不在此列，举报者身份永远不进任何 payload。）
            "attacked_by": list(self.attacked_by),
            "attack_merit_loss": self.attack_merit_loss,
            "tenure_reset_by_attack": self.tenure_reset_by_attack,
            # merit_from_attacks 不进公开事实：它会直接点出谁是攻击者。
            # （政绩是公开数据，抢功这个机制本身就让身份半可推断，见 README）
            "merit_promotion_blocked": self.merit_promotion_blocked,
            # 查办是谁挑起的，公报里本来就说了（"被匿名举报" vs "在反腐风暴中被查办"），
            # 所以可以公开——但仍然不说是**哪个**玩家举报的。
            "reported_by_player": self.report_count_players > 0,
            "report_from_event": self.report_from_event,
            "warnings_issued": self.warnings_issued,
            "warnings_after": self.warnings_after,
            "promotion": self.promotion.value,
            "demotion": self.demotion.value,
            "rank_before": self.rank_before,
            "rank_after": self.rank_after,
            "merit_after": self.merit_after,
            "tenure_after": self.tenure_after,
        }

    def private_view(self) -> dict[str, Any]:
        return {
            "cards": [c.value for c in self.cards],
            "targets": list(self.targets),
            "base_values": list(self.base_values),
            "salary": self.salary,
            "merit_gained": self.merit_gained,
            "money_gained": self.money_gained,
            "corrupt_amount": self.corrupt_amount,
            "attacked": self.attacked,
            "attack_merit_loss": self.attack_merit_loss,
            "merit_stolen_by_attackers": self.merit_stolen_by_attackers,
            "merit_wiped_by_attack": self.merit_wiped_by_attack,
            "merit_from_attacks": self.merit_from_attacks,
            "reported": self.reported,
            "report_effective": self.report_effective,
            "warnings_issued": self.warnings_issued,
            "warnings_after": self.warnings_after,
            "bribe_lost": self.bribe_lost,
            "demotion": self.demotion.value,
            "money_confiscated": self.money_confiscated,
            "hush_money_paid": self.hush_money_paid,
            "money_from_reports": self.money_from_reports,
            "promotion": self.promotion.value,
            "promotion_card_played": self.promotion_card_played,
            "promotion_money_cost": self.promotion_money_cost,
            "promotion_merit_cost": self.promotion_merit_cost,
            "promotion_merit_decay": self.promotion_merit_decay,
            "money_before_promotion": self.money_before_promotion,
            "merit_before_promotion": self.merit_before_promotion,
            "promotion_blocked": self.promotion_blocked_by_attack_report
            or self.merit_promotion_blocked,
            "rank_before": self.rank_before,
            "rank_after": self.rank_after,
            "money_after": self.money_after,
            "merit_after": self.merit_after,
            "tenure_after": self.tenure_after,
            "notes": list(self.private_notes),
        }


@dataclass
class RoundOutcome:
    """一整轮的结算产物。"""

    round_number: int
    event: GameEvent
    outcomes: dict[int, PlayerRoundOutcome] = field(default_factory=dict)
    public_messages: list[str] = field(default_factory=list)
    wealth_broadcast: list[str] = field(default_factory=list)
    # 财富广播点名的玩家，以及对应的档位下标（档位区间本来就写在广播词里）
    wealth_top_ids: list[int] = field(default_factory=list)
    presidents: list[int] = field(default_factory=list)  # 本轮达到国家主席的玩家

    def public_view(self) -> dict[str, Any]:
        return {
            "round": self.round_number,
            "event": self.event.public_view(),
            "messages": list(self.public_messages),
            "wealth_broadcast": list(self.wealth_broadcast),
            "wealth_top_ids": list(self.wealth_top_ids),
            "player_facts": [o.public_facts() for _, o in sorted(self.outcomes.items())],
            "presidents": list(self.presidents),
        }
