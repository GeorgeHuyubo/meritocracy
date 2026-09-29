"""Meritocracy 全部规则计算。

这里是纯函数 + 一个纯结算入口 `resolve_round`，不依赖 Web、不依赖数据库、
不持有全局状态。simulator.py 和 server.py 共用同一份实现。

结算顺序（规则书第 4 节第 7~10 步的精确展开）：

    0. 合法工资   按官职自动到账
    1. 锁定干扰   先把"谁攻击谁、谁举报谁"记下来（只记目标，不结算效果）
    2. 早期晋升   **生产之前就已经够门槛的，先升官再干活**，这样本轮产出能吃到
                  新官职的倍率，也不会被 /5 砍掉。被攻击或被举报盯上的人不走这条。
    3. 资源结算   WORK / CORRUPT  ->  卡牌基础值 → 事件修改 → 官职倍率 → 向下取整
    4. 政治攻击   结算效果：目标本轮没有 WORK 则扣政绩
    5. 匿名举报   按"本轮最终贪污金额"判定是否查实、是否重大贪腐
    6. 没收分赃   赃款没收后平分给举报人，被举报人降级（重大贪腐打回基层）
    7. 晚期晋升   生产之后才够门槛的，在这里结算
    8. 工龄       没晋升 +1，满 TENURE_REQUIRED 自动升一级（这条**不需要卡**）

每个玩家一轮打出 PICKS_PER_ROUND 张牌，所以 `actions` 是 玩家 -> 行动列表。
两张生产牌会各自结算一次；本轮贪污额是两笔之和（影响重大贪腐判定和财富广播）。

第 4 步先把所有没收金额算完再统一扣除、统一发放，所以"甲举报乙、乙同时举报甲"
这种互相举报不会因为结算先后顺序而产生差异。分赃在晋升之前完成，举报人当轮
就可以用分到的钱晋升。

关于倍率：一律用 Fraction 精确运算，只在最终收益处 floor 一次。
"""

from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction
from typing import Iterable, Sequence

from config import Config, DEFAULT_CONFIG

EXPECTED_CARD_VALUE = 10  # WORK/CORRUPT 牌面的期望点数，用作"一个回合当量"
from models import (
    Action,
    Card,
    DealtCard,
    DemotionKind,
    GameEvent,
    PlayerRoundOutcome,
    PlayerState,
    PromotionKind,
    RoundOutcome,
)

# --------------------------------------------------------------------------
# 随机
# --------------------------------------------------------------------------


def weighted_choice(rng, pairs: Sequence[tuple[object, int]]):
    """按权重抽一个。pairs = [(value, weight), ...]。"""
    total = sum(w for _, w in pairs)
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    roll = rng.randrange(total)
    upto = 0
    for value, weight in pairs:
        upto += weight
        if roll < upto:
            return value
    return pairs[-1][0]  # 理论上不可达


def roll_card_value(card: Card, rng, cfg: Config = DEFAULT_CONFIG) -> int:
    """这张牌的点数。只有生产牌有点数，其余恒为 0。"""
    if card is Card.WORK:
        return roll_work_base(rng, cfg)
    if card is Card.CORRUPT:
        return roll_corrupt_base(rng, cfg)
    if card is Card.GRAFT:
        return roll_graft_base(rng, cfg)
    return 0


def deal_hand(rng, cfg: Config = DEFAULT_CONFIG) -> list[DealtCard]:
    """从一副**真牌库**里不放回地发 HAND_SIZE 张，**点数当场摇好**。

    CARD_DEAL_DISTRIBUTION 里的数字就是"这副牌里有几张"，所以
    一手牌里同名牌最多就是那么多张——写着 2 张政绩升职，就绝不会摸到 3 张。
    （以前是按权重有放回抽样，26.7% 的手牌会出现 3 张以上同名牌，
    和配置看起来的意思对不上。）

    每个玩家每轮都从一副完整的新牌库里抓，所以人数多了也不会互相抢牌。

    先摇后选：玩家看着确定的数字做决策，不用赌自己这张牌大不大。
    需要保密的只有"本轮全局事件"，那个仍然在所有人锁定之后才抽。
    """
    deck: list[Card] = []
    for name, count in cfg.card_deal_distribution.items():
        deck.extend([Card(name)] * count)
    if len(deck) < cfg.hand_size:
        raise ValueError(
            f"牌库只有 {len(deck)} 张，发不出 {cfg.hand_size} 张的手牌"
        )
    picked = _sample_without_replacement(rng, deck, cfg.hand_size)
    return [
        DealtCard(card=card, value=roll_card_value(card, rng, cfg)) for card in picked
    ]


def _sample_without_replacement(rng, deck: list[Card], k: int) -> list[Card]:
    """从 deck 里不放回地抓 k 张。

    不用 rng.sample：测试里的 ScriptedRng 只实现了 randrange，
    而对局用的 secrets.SystemRandom 也要能走同一条路。
    """
    pool = list(deck)
    out: list[Card] = []
    for _ in range(k):
        out.append(pool.pop(rng.randrange(len(pool))))
    return out


def roll_work_base(rng, cfg: Config = DEFAULT_CONFIG) -> int:
    return int(weighted_choice(rng, cfg.work_card_distribution))


def roll_corrupt_base(rng, cfg: Config = DEFAULT_CONFIG) -> int:
    return int(weighted_choice(rng, cfg.corrupt_card_distribution))


def roll_graft_base(rng, cfg: Config = DEFAULT_CONFIG) -> int:
    return int(weighted_choice(rng, cfg.graft_card_distribution))


def graft_merit(base: int, rank: int, event: GameEvent | None = None,
                cfg: Config = DEFAULT_CONFIG) -> int:
    """以权谋私顺带的政绩：按牌面的一个比例折算，再走 WORK 的官职倍率。"""
    return work_merit(math.floor(Fraction(base) * cfg.graft_merit_ratio), rank, event, cfg)


def pick_event(rng, cfg: Config = DEFAULT_CONFIG) -> GameEvent:
    """抽取全局事件。必须在所有玩家锁定行动之后才调用。"""
    pairs = [(d, int(d["weight"])) for d in cfg.event_definitions]
    chosen = weighted_choice(rng, pairs)
    return GameEvent(
        id=chosen["id"],
        name=chosen["name"],
        description=chosen["description"],
        effects=dict(chosen.get("effects", {})),
    )


def event_by_id(event_id: str, cfg: Config = DEFAULT_CONFIG) -> GameEvent:
    for d in cfg.event_definitions:
        if d["id"] == event_id:
            return GameEvent(
                id=d["id"],
                name=d["name"],
                description=d["description"],
                effects=dict(d.get("effects", {})),
            )
    raise KeyError(f"unknown event id: {event_id}")


# --------------------------------------------------------------------------
# 收益计算：卡牌基础数值 -> 事件修改 -> 官职倍率 -> 最终收益
# --------------------------------------------------------------------------


def _finalize(value: Fraction) -> int:
    """收益一律向下取整，且不为负。"""
    return max(0, math.floor(value))


def work_merit(
    base: int, rank: int, event: GameEvent | None = None, cfg: Config = DEFAULT_CONFIG
) -> int:
    """WORK 的最终政绩收益。"""
    value = Fraction(base)
    if event is not None:
        value += Fraction(int(event.effects.get("work_bonus", 0)))
        value *= Fraction(event.effects.get("work_multiplier", 1))
    value *= cfg.work_multiplier(rank)
    return _finalize(value)


def corrupt_money(
    base: int, rank: int, event: GameEvent | None = None, cfg: Config = DEFAULT_CONFIG
) -> int:
    """CORRUPT 的最终金钱收益。这个数字同时用于举报判定和财富广播。"""
    value = Fraction(base)
    if event is not None:
        value += Fraction(int(event.effects.get("money_bonus", 0)))
        value *= Fraction(event.effects.get("money_multiplier", 1))
    value *= cfg.money_multiplier(rank)
    return _finalize(value)


PROMOTION_CARDS = {Card.PROMOTE_MERIT, Card.PROMOTE_MONEY, Card.PROMOTE_ANY}


def _promotion_kind_for(
    player: PlayerState, cards: set[Card], cfg: Config
) -> PromotionKind:
    """手上这组牌 + 现在的资源，能走哪种晋升。不能升就返回 NONE。

    双条件台阶上任意一张晋升卡都行；其他台阶要卡和资源对得上，政绩优先于金钱。
    """
    if cfg.promotion_requires_card:
        merit_card = any(c.can_use_merit for c in cards)
        money_card = any(c.can_use_money for c in cards)
        any_card = bool(cards & PROMOTION_CARDS)
    else:
        merit_card = money_card = any_card = True

    if cfg.needs_both(player.rank):
        return PromotionKind.BOTH if (any_card and can_promote_with_both(player, cfg)) \
            else PromotionKind.NONE
    if merit_card and can_promote_by_merit(player, cfg):
        return PromotionKind.MERIT
    if money_card and can_promote_by_money(player, cfg):
        return PromotionKind.MONEY
    return PromotionKind.NONE


def _apply_promotion(
    player: PlayerState, kind: PromotionKind, outcome, cfg: Config
) -> None:
    """执行晋升，并把消耗明细记下来给 UI 用。"""
    merit_before, money_before = player.merit, player.money
    if kind is PromotionKind.BOTH:
        outcome.promotion_money_cost = cfg.money_cost(player.rank) or 0
        outcome.promotion_merit_cost = cfg.merit_cost(player.rank) or 0
        apply_both_promotion(player, cfg)
    elif kind is PromotionKind.MERIT:
        outcome.promotion_merit_cost = cfg.merit_cost(player.rank) or 0
        apply_merit_promotion(player, cfg)
    elif kind is PromotionKind.MONEY:
        outcome.promotion_money_cost = cfg.money_cost(player.rank) or 0
        apply_money_promotion(player, cfg)
    else:
        return
    outcome.money_before_promotion = money_before
    outcome.merit_before_promotion = merit_before
    # 扣掉门槛之后还少了多少，就是"升职后打折"那一刀。
    # 必须当场量：贿赂升职和工龄晋升一分政绩成本都没有，却照样要打折，
    # 事后拿 merit_after 反推会把这一轮后面的产出也算进来，账就对不上了。
    outcome.promotion_merit_decay = (
        merit_before - outcome.promotion_merit_cost
    ) - player.merit
    outcome.promotion = kind
    outcome.promotion_card_played = True


def _resolve_promotion_card(
    player: PlayerState,
    card: Card,
    outcome,
    cfg: Config,
    names: dict[int, str],
    promo_msgs: list[str],
    already: PromotionKind,
    auto: bool = False,
) -> None:
    """玩家打出一张晋升卡，当场结算。

    克制矩阵：
        政绩升职  —— 政治攻击可以暂缓（政绩一点不掉），举报碰不到
        贿赂升职  —— 举报查实就失败、钱还要不回来，政治攻击碰不到
        通用升职  —— 优先走政绩；被攻击就改走金钱（这正是这张卡的价值）；
                     两样都被堵 -> 升职失败、金钱损失、政绩保留
        双条件台阶（省级->主席）钱和政绩都要花，所以两张牌都拦得住它。
    """
    if not auto:
        outcome.promotion_card_played = True
    if already is not PromotionKind.NONE:
        outcome.private_notes.append("这一轮已经升过一级了，这张晋升卡用不上。")
        return
    if cfg.money_cost(player.rank) is None:  # 已经到顶
        return

    merit_ok = card.can_use_merit and has_merit_for_promotion(player, cfg)
    money_ok = card.can_use_money and has_money_for_promotion(player, cfg)

    if outcome.attacked and merit_ok:
        # 暂缓升职：这一轮走不了政绩这条路，但政绩一点不掉。
        # （清零那套太重——一刀能削 43 点，而且攻击者一分不拿，纯利他。
        #   现在攻击的收益全部来自抢功，破坏这块降到最小。）
        outcome.merit_promotion_blocked = True
        if cfg.attack_wipes_merit_on_block and player.merit:
            outcome.merit_wiped_by_attack = player.merit
            outcome.attack_merit_loss += player.merit
            promo_msgs.append(
                f"{names[player.id]} 眼看就要凭政绩上位，被人一状告倒，"
                f"{player.merit} 点政绩付诸东流。"
            )
            player.merit = 0
        merit_ok = False
        money_ok = card.can_use_money and has_money_for_promotion(player, cfg)

    if cfg.needs_both(player.rank):
        # 这一级钱和政绩都要花，所以"走的是哪条路"不由资源决定，**由打出的卡决定**：
        #   政绩升职 -> 声明走正规程序：怕政治攻击，被拦下只是暂缓、零惩罚
        #   贿赂升职 -> 声明走关系：怕匿名举报，被查实要赔钱记警告
        #   通用升职 -> 先按政绩升职算；被攻击拦下就回退成贿赂升职
        # 上面那段已经把"被攻击 + 能用政绩"的情况标记成 merit_promotion_blocked
        # 并把 merit_ok 置了 False，所以这里只看还剩哪条路。
        if not can_promote_with_both(player, cfg):
            if not auto:
                gaps = []
                if not has_merit_for_promotion(player, cfg):
                    gaps.append("政绩不够")
                if not has_money_for_promotion(player, cfg):
                    gaps.append("钱不够")
                outcome.private_notes.append(
                    "这一级要政绩和金钱同时达标：" + "、".join(gaps or ["条件不足"])
                )
            return
        if outcome.merit_promotion_blocked and not card.can_use_money:
            # 政绩升职没有退路：暂缓，且**不受任何惩罚**（钱不掉、不记警告）
            if not auto:
                outcome.private_notes.append(
                    "最后一步被政治攻击按住了，只是暂缓——钱和政绩都还在。"
                )
            return
        if outcome.attacked and not card.can_use_merit:
            # 贿赂升职声明的是关系路线，政治攻击拦不住它
            pass
        kind = PromotionKind.BOTH
        if outcome.merit_promotion_blocked:
            outcome.private_notes.append(
                "政绩这条路被人挡下，改走打点上位。"
            )
    elif merit_ok:
        kind = PromotionKind.MERIT
    elif money_ok:
        kind = PromotionKind.MONEY
        if outcome.merit_promotion_blocked:
            outcome.private_notes.append("政绩晋升被人挡下，改用金钱打点，总算升上去了。")
    else:
        if not auto:
            if outcome.merit_promotion_blocked:
                outcome.private_notes.append(
                    "政绩晋升被政治攻击挡下了，这张卡又用不了金钱——本轮升不上去。"
                )
            else:
                outcome.private_notes.append("你的晋升卡没能用上：资源不够。")
        return

    _apply_promotion(player, kind, outcome, cfg)
    promo_msgs.append(_promotion_message(names[player.id], kind, cfg.rank_name(player.rank)))


def _promotion_message(name: str, kind: PromotionKind, rank_name: str) -> str:
    return {
        PromotionKind.MERIT: f"{name} 政绩卓著，晋升为{rank_name}。",
        PromotionKind.MONEY: f"{name} 四处打点，晋升为{rank_name}。",
        PromotionKind.BOTH: f"{name} 政绩金钱两手都硬，晋升为{rank_name}。",
    }.get(kind, f"{name} 晋升为{rank_name}。")

# --------------------------------------------------------------------------
# 晋升 / 降级
# --------------------------------------------------------------------------


def overflow_after_promotion(
    remaining: int, cfg: Config = DEFAULT_CONFIG, divisor: int | None = None
) -> int:
    """晋升后超额资源衰减：ceil(remaining / divisor)，不为负。divisor=1 表示不衰减。"""
    if remaining <= 0:
        return 0
    return math.ceil(Fraction(remaining, divisor if divisor is not None else cfg.overflow_divisor))


def has_money_for_promotion(player: PlayerState, cfg: Config = DEFAULT_CONFIG) -> bool:
    cost = cfg.money_cost(player.rank)
    return cost is not None and player.money >= cost


def has_merit_for_promotion(player: PlayerState, cfg: Config = DEFAULT_CONFIG) -> bool:
    cost = cfg.merit_cost(player.rank)
    return cost is not None and player.merit >= cost


def can_promote_by_money(player: PlayerState, cfg: Config = DEFAULT_CONFIG) -> bool:
    """能不能靠金钱这条线升上去。双条件台阶上，光有钱不够。"""
    if cfg.needs_both(player.rank):
        return False
    return has_money_for_promotion(player, cfg)


def can_promote_by_merit(player: PlayerState, cfg: Config = DEFAULT_CONFIG) -> bool:
    """能不能靠政绩这条线升上去。双条件台阶上，光有政绩不够。"""
    if cfg.needs_both(player.rank):
        return False
    return has_merit_for_promotion(player, cfg)


def can_promote_with_both(player: PlayerState, cfg: Config = DEFAULT_CONFIG) -> bool:
    """双条件台阶：金钱和政绩必须同时达标。"""
    return (
        cfg.needs_both(player.rank)
        and has_money_for_promotion(player, cfg)
        and has_merit_for_promotion(player, cfg)
    )


def apply_promotion_costs(
    player: PlayerState,
    cfg: Config = DEFAULT_CONFIG,
    *,
    pay_money: bool = False,
    pay_merit: bool = False,
) -> None:
    """升一级：扣掉用到的那份资源，然后**两样都做衰减**。

    "升职以后政绩除以 5" 对任何晋升方式都成立，贿赂上位也一样。
    不这么做的话，"攒政绩 + 花钱升职"能把政绩原封不动地带过一级，
    等于白拿一级，贿赂升职会严格优于政绩升职。
    金钱的除数默认是 1（不衰减），所以这条对钱没有实际影响。
    """
    money_left = player.money - (cfg.money_cost(player.rank) or 0 if pay_money else 0)
    merit_left = player.merit - (cfg.merit_cost(player.rank) or 0 if pay_merit else 0)

    # 金钱：只有真的花掉才衰减（没花就原样留着）
    player.money = (
        overflow_after_promotion(money_left, cfg, cfg.money_overflow_divisor)
        if pay_money
        else max(0, money_left)
    )
    # 政绩：开关打开时，不管这一级是怎么升上去的都要衰减
    player.merit = (
        overflow_after_promotion(merit_left, cfg, cfg.merit_overflow_divisor)
        if (pay_merit or cfg.promotion_always_decays_merit)
        else max(0, merit_left)
    )
    player.rank += 1


def apply_both_promotion(player: PlayerState, cfg: Config = DEFAULT_CONFIG) -> None:
    """双条件晋升：金钱和政绩各按各的成本扣，各自做衰减。"""
    mc, tc = cfg.money_cost(player.rank), cfg.merit_cost(player.rank)
    assert mc is not None and tc is not None
    assert player.money >= mc and player.merit >= tc
    apply_promotion_costs(player, cfg, pay_money=True, pay_merit=True)


def apply_money_promotion(player: PlayerState, cfg: Config = DEFAULT_CONFIG) -> None:
    """金钱晋升：扣掉金钱门槛；政绩没花掉，但一样要做 /5 衰减。"""
    cost = cfg.money_cost(player.rank)
    assert cost is not None and player.money >= cost
    apply_promotion_costs(player, cfg, pay_money=True)


def apply_merit_promotion(player: PlayerState, cfg: Config = DEFAULT_CONFIG) -> None:
    """政绩晋升：merit = ceil((merit - cost) / MERIT_OVERFLOW_DIVISOR)。"""
    cost = cfg.merit_cost(player.rank)
    assert cost is not None and player.merit >= cost
    apply_promotion_costs(player, cfg, pay_merit=True)


def apply_demotion(
    player: PlayerState, kind: DemotionKind, cfg: Config = DEFAULT_CONFIG
) -> None:
    """举报造成的降级。玩家永不被淘汰。"""
    if kind is DemotionKind.MAJOR:
        # 重大贪腐：打回基层，金钱清零，政绩保留
        player.rank = cfg.base_rank
        player.tenure = 0
        player.money = 0
    elif kind is DemotionKind.MINOR:
        # 小额贪污：降一级，金钱和政绩保留
        player.rank = max(cfg.base_rank, player.rank - 1)
        player.tenure = 0


def classify_report(
    corrupt_amount: int, cfg: Config = DEFAULT_CONFIG, bribe: int = 0
) -> DemotionKind:
    """这份举报查不查得实，以及算不算重大贪腐。

    本轮贪了钱一定算。本轮拿钱买官算不算，看 REPORT_CATCHES_BRIBERY：
    打开 = 赚钱和花钱两头都要过举报这一关，金钱路线被双重收税。
    返回 MAJOR/MINOR 只用来决定记几次警告，不再直接决定降几级。
    """
    if not cfg.report_catches_bribery:
        bribe = 0
    if corrupt_amount <= 0 and bribe <= 0:
        return DemotionKind.NONE
    if corrupt_amount >= cfg.major_corruption_threshold:
        return DemotionKind.MAJOR
    return DemotionKind.MINOR


def warnings_for(kind: DemotionKind, cfg: Config = DEFAULT_CONFIG) -> int:
    """这次查实记几个降职警告。"""
    if kind is DemotionKind.NONE:
        return 0
    if kind is DemotionKind.MAJOR:
        return max(1, cfg.major_corruption_warnings)
    return 1


# --------------------------------------------------------------------------
# 财富广播
# --------------------------------------------------------------------------


def _join_names(names_list: list[str], cfg: Config) -> str:
    """把一串名字连成中文里读得顺的列表：甲、乙和丙。

    直接用 " 和 " 连接在两人时没问题，五个人就变成
    "甲 和 乙 和 丙 和 丁 和 戊"，读起来很别扭。
    """
    if len(names_list) <= 1:
        return "".join(names_list)
    return "、".join(names_list[:-1]) + cfg.wealth_broadcast_name_joiner + names_list[-1]


def _report_source_phrase(outcome) -> str:
    """公报里怎么说这次查办是哪来的。

    真人举报和事件查办后果完全一样，但不说清楚的话，被反腐风暴扫到的人
    会以为桌上有人在针对自己，据此去报复一个根本不存在的敌人。
    """
    by_players = outcome.report_count_players > 0
    by_event = bool(outcome.report_from_event)
    if by_players and by_event:
        return f"被匿名举报，又撞上{outcome.report_from_event}"
    if by_event:
        return f"在{outcome.report_from_event}中被查办"
    return "被匿名举报"


def storm_targets(
    corrupt_by_id: dict[int, int], cfg: Config = DEFAULT_CONFIG
) -> list[int]:
    """「反腐风暴」查办谁：本轮贪污额排在前 fraction 的人。

    * 名额 = ceil(总人数 x fraction)，但本轮没贪污的人不会被卷进来
    * 并列卡在分界线上的一起查办（不按玩家编号随便截断）
    """
    guilty = {pid: amt for pid, amt in corrupt_by_id.items() if amt > 0}
    if not guilty:
        return []
    quota = math.ceil(Fraction(len(corrupt_by_id)) * cfg.event_storm_fraction)
    quota = max(1, min(quota, len(guilty)))
    ordered_amounts = sorted(guilty.values(), reverse=True)
    cutoff = ordered_amounts[quota - 1]
    return sorted(pid for pid, amt in guilty.items() if amt >= cutoff)


def wealth_broadcast_detail(
    corrupt_by_name: Iterable[tuple[str, int]],
    cfg: Config = DEFAULT_CONFIG,
    rng=None,
) -> tuple[list[str], list[str], int | None]:
    """返回 (广播文案, 榜首玩家名, 档位下标)。

    只广播本轮贪污金额最高的玩家，且绝不显示具体金额。
    没人贪污 -> 空。多人并列最高 -> 合并成一条消息全部点名。
    档位下标是公开信息：广播词本身就已经暗示了金额区间。
    """
    entries = [(name, amount) for name, amount in corrupt_by_name if amount > 0]
    if not entries:
        return [], [], None
    top = max(amount for _, amount in entries)
    names = [name for name, amount in entries if amount == top]

    for tier, (low, high, singles, multis) in enumerate(cfg.wealth_broadcast_tiers):
        if top >= low and (high is None or top <= high):
            joined = _join_names(names, cfg)
            pool = singles if len(names) == 1 else multis
            # 同一句每轮重复玩家就不看了，所以每档备了几条随机挑
            tpl = rng.choice(pool) if (rng is not None and pool) else pool[0]
            return [tpl.format(names=joined)], names, tier
    return [], [], None


def wealth_broadcast(
    corrupt_by_name: Iterable[tuple[str, int]], cfg: Config = DEFAULT_CONFIG
) -> list[str]:
    return wealth_broadcast_detail(corrupt_by_name, cfg)[0]


def wealth_tier_range(tier: int, cfg: Config = DEFAULT_CONFIG) -> tuple[int, int | None]:
    """档位下标 -> (最小金额, 最大金额或 None)。AI 用它把广播词翻译成区间。"""
    low, high, _, _ = cfg.wealth_broadcast_tiers[tier]
    return low, high


# --------------------------------------------------------------------------
# 终局判定
# --------------------------------------------------------------------------


def final_key(player: PlayerState, cfg: Config = DEFAULT_CONFIG) -> tuple[int, ...]:
    """终局比大小用的键，顺序由 FINAL_RANKING_KEYS 决定（默认 官职 > 金钱 > 政绩）。"""
    return tuple(getattr(player, k) for k in cfg.final_ranking_keys)


def final_ranking(
    players: Sequence[PlayerState], cfg: Config = DEFAULT_CONFIG
) -> list[PlayerState]:
    """打满轮数后的排序，全部降序。"""
    return sorted(players, key=lambda p: final_key(p, cfg), reverse=True)


def final_winners(
    players: Sequence[PlayerState], cfg: Config = DEFAULT_CONFIG
) -> list[PlayerState]:
    """终局赢家。三项全部相同则并列（平局）。"""
    if not players:
        return []
    ordered = final_ranking(players, cfg)
    best = final_key(ordered[0], cfg)
    return [p for p in ordered if final_key(p, cfg) == best]


def presidents(players: Sequence[PlayerState], cfg: Config = DEFAULT_CONFIG) -> list[PlayerState]:
    return [p for p in players if p.rank >= cfg.president_rank]


# --------------------------------------------------------------------------
# 单轮结算
# --------------------------------------------------------------------------


def pay_salaries(
    players: Sequence[PlayerState], cfg: Config = DEFAULT_CONFIG
) -> dict[int, int]:
    """按官职发本轮工资，**在回合开始、发牌的同时到账**。

    放在回合开头而不是结算里，是为了让这笔钱当轮就能用——
    尤其是拿去换一手牌（基层的一轮工资正好够换一次）。
    工资不占行动位，也不算贪污：举报和反腐风暴都碰不到它。

    返回 {player_id: 实发金额}，给流水账用。
    """
    paid: dict[int, int] = {}
    for p in players:
        pay = cfg.salary(p.rank)
        if pay > 0:
            p.money += pay
            paid[p.id] = pay
    return paid


def resolve_round(
    players: Sequence[PlayerState],
    actions: dict[int, list[Action]],
    event: GameEvent,
    rng,
    round_number: int = 1,
    cfg: Config = DEFAULT_CONFIG,
) -> RoundOutcome:
    """就地结算一轮。`players` 会被修改。

    `actions[pid]` 是该玩家本轮打出的全部行动，**顺序就是玩家自己选的顺序**，
    结算严格照着这个顺序走（长度 <= PICKS_PER_ROUND）。缺失或空列表 = 本轮没有出牌。

    工资不在这里发——它在回合**开始**时就到账了（见 pay_salaries），
    这样本轮的工资当轮就能拿去换牌。

    结算顺序：
      1. 锁定干扰目标（只记谁打谁）
      2. 每个玩家按自选顺序逐张结算自己的牌（生产 / 晋升交错）
      3. 政治攻击效果
      4. 匿名举报判定
      5. 没收分赃 + 降级
      6. 晋升提示
      7. 工龄
    """
    by_id = {p.id: p for p in players}
    names = {p.id: p.name for p in players}
    ordered = sorted(players, key=lambda p: p.id)

    def played(pid: int) -> list[Action]:
        return actions.get(pid) or []

    def has_card(pid: int, card: Card) -> bool:
        return any(a.card is card for a in played(pid))

    outcome = RoundOutcome(round_number=round_number, event=event)
    for p in ordered:
        acts = played(p.id)
        outcome.outcomes[p.id] = PlayerRoundOutcome(
            player_id=p.id,
            cards=[a.card for a in acts],
            targets=[a.target_id for a in acts],
            rank_before=p.rank,
        )

    attack_msgs: list[str] = []
    report_msgs: list[str] = []
    promo_msgs: list[str] = []

    # ---- 1. 锁定干扰目标 --------------------------------------------------
    # 只记"谁打谁"，效果留到生产之后再结算。先记下来是为了下一步的早期晋升
    # 能知道自己有没有被盯上。
    attack_disabled = event.flag("attack_disabled")
    attackers_of: dict[int, list[int]] = defaultdict(list)
    for p in ordered:
        for act in played(p.id):
            if act.card is not Card.ATTACK:
                continue
            o_actor = outcome.outcomes[p.id]
            target = by_id.get(act.target_id) if act.target_id is not None else None
            if target is None or target.id == p.id:
                o_actor.private_notes.append("攻击没有指定有效目标，行动作废。")
                continue
            if attack_disabled:
                o_actor.private_notes.append(
                    f"政治环境稳定，你对 {names[target.id]} 的政治攻击被化解了。"
                )
                continue
            outcome.outcomes[target.id].attacked = True
            if cfg.attack_announces_attacker:
                outcome.outcomes[target.id].attacked_by.append(p.id)
            if p.id not in attackers_of[target.id]:
                # 同一个人对同一目标打两张攻击不会翻倍，只是浪费一张牌
                attackers_of[target.id].append(p.id)
            o_actor.private_notes.append(f"你对 {names[target.id]} 发动了政治攻击。")

    report_counts: dict[int, int] = defaultdict(int)
    report_actors: dict[int, list[int]] = defaultdict(list)  # 被举报人 -> 真人举报者
    for p in ordered:
        for act in played(p.id):
            if act.card is not Card.REPORT:
                continue
            o_actor = outcome.outcomes[p.id]
            target = by_id.get(act.target_id) if act.target_id is not None else None
            if target is None or target.id == p.id:
                o_actor.private_notes.append("举报没有指定有效目标，行动作废。")
                continue
            if p.id in report_actors[target.id]:
                o_actor.private_notes.append(
                    f"你已经举报过 {names[target.id]} 了，这张牌白打了。"
                )
                continue
            report_counts[target.id] += 1
            report_actors[target.id].append(p.id)
            o_actor.private_notes.append(f"你匿名举报了 {names[target.id]}。")

    if event.flag("storm_report"):
        pass  # 风暴要看本轮贪污额，留到生产之后再算

    # ---- 2. 按玩家自选的顺序，逐张结算他自己的牌 ---------------------------
    # 出牌顺序是玩家的决策，这里**严格照着来**：
    #   先干活后升职 -> 这一轮的产出算进晋升前的账，但超额部分会被 /5 砍掉
    #   先升职后干活 -> 产出吃到新官职的倍率，而且不会被砍
    #   被攻击的人把升职卡放前面 -> 政绩在干活之前就被清零，这轮的活还能留住
    # 攻击和举报只在这一步"过一下"，它们的效果要等所有人产出都算完（见第 3、4 步）。
    # 有两种情况，晋升卡要等到全场结算完再兑现：
    #   1. 排在举报/攻击后面 —— 那两张牌的进账（赃款、封口费）还没到账，
    #      玩家把晋升卡放在后面，就是想花这笔钱。
    #   2. 排在贪污后面 —— 这一轮刚捞的钱当轮花不出去，得先熬过举报才算落袋。
    #      少了这一条，"贪一笔立刻洗成官职"就能躲开没收：举报人查实了却分不到钱，
    #      而且升一级又被降一级，官职净变化为零，等于举报白打。
    deferred_promotions: list[tuple[PlayerState, Card]] = []
    for p in ordered:
        o = outcome.outcomes[p.id]
        # 举报只管**要花钱**的那条路（贪污所得、拿钱买官）。
        # 走政绩升职的人举报碰不到他 —— 那是政治攻击的活。
        targeted = bool(report_counts.get(p.id))
        produced_yet = False
        interfered_yet = False
        corrupted_yet = False
        for act in played(p.id):
            card = act.card
            if card.is_production:
                produced_yet = True
            if card.yields_corruption:
                corrupted_yet = True
            if card is Card.WORK:
                base = act.value if act.value > 0 else roll_work_base(rng, cfg)
                gained = work_merit(base, p.rank, event, cfg)
                o.base_values.append(base)
                o.merit_gained += gained
                p.merit += gained
            elif card is Card.CORRUPT:
                base = act.value if act.value > 0 else roll_corrupt_base(rng, cfg)
                gained = corrupt_money(base, p.rank, event, cfg)
                o.base_values.append(base)
                o.money_gained += gained
                o.corrupt_amount += gained
                p.money += gained
            elif card is Card.GRAFT:
                base = act.value if act.value > 0 else roll_graft_base(rng, cfg)
                cash = corrupt_money(base, p.rank, event, cfg)
                bonus = graft_merit(base, p.rank, event, cfg)
                o.base_values.append(base)
                o.money_gained += cash
                o.corrupt_amount += cash  # 这笔钱照样算贪污
                o.merit_gained += bonus
                p.money += cash
                p.merit += bonus
            elif card in (Card.REPORT, Card.ATTACK):
                interfered_yet = True
            elif card.is_promotion:
                # 这一张实际会走哪条路？政绩优先，但被攻击就得改走金钱。
                #   * 双条件台阶（省级 -> 主席）钱和政绩都要花 -> 一定要掏钱
                #   * 其他台阶政绩够又没被攻击 -> 走政绩，一分钱不用出
                # 先记下"他本来就打算凭政绩升职"——和有没有被拦、有没有被延后
                # 都无关。攻击那一步要靠它判断他这轮是不是在干正事。
                if card.can_use_merit and has_merit_for_promotion(p, cfg):
                    o.tried_merit_promotion = True
                # 走的是政绩那条路还是金钱那条路，**由卡决定**——
                # 双条件台阶（省级->主席）钱和政绩一起花，但"我声明走哪条路"
                # 仍然看你打的是政绩升职还是贿赂升职，这决定了你怕谁：
                # 走政绩怕攻击（只暂缓、零惩罚），走金钱怕举报（赔钱记警告）。
                would_use_merit = (
                    card.can_use_merit
                    and has_merit_for_promotion(p, cfg)
                    and not o.attacked
                )
                # 不走政绩就得掏钱 —— 哪怕这会儿兜里还没钱：
                # 他可能正等着本轮举报/攻击抄来的赃款到账（见 interfered_yet）。
                may_need_money = card.can_use_money and not would_use_merit
                # 只有要掏钱的晋升才可能被举报掐掉，所以也只有它需要等：
                #   targeted       —— 有人举报我，这笔钱可能被查
                #   interfered_yet —— 我想花的是本轮举报/攻击抄来的钱，还没到账
                #   corrupted_yet  —— 这一轮刚贪的钱，当轮花不出去
                # 走政绩的晋升当场结算，玩家自选的出牌顺序完整保留。
                if may_need_money and (targeted or interfered_yet or corrupted_yet):
                    if has_money_for_promotion(p, cfg):
                        # 礼已经备好了：查实的话钱照样没了，官却升不成
                        o.pending_bribe = max(
                            o.pending_bribe, cfg.money_cost(p.rank) or 0
                        )
                    deferred_promotions.append((p, card))
                    continue
                _resolve_promotion_card(
                    p, card, o, cfg, names, promo_msgs, already=o.promotion
                )
                if o.promotion is not PromotionKind.NONE and not produced_yet:
                    o.promoted_before_production = True


    # ---- 3. 政治攻击（结算效果；目标在第 1 步就已经记下了）-----------------

    # 伤害一律基于"攻击结算之前"的政绩快照，这样互相攻击时两边完全对称，
    # 不会因为谁先结算而占便宜。
    merit_before_attacks = {p.id: p.merit for p in ordered}

    for target_id, attacker_ids in sorted(attackers_of.items()):
        target = by_id[target_id]
        o_target = outcome.outcomes[target_id]
        target_worked = has_card(target_id, Card.WORK) or has_card(target_id, Card.GRAFT)

        if cfg.attack_resets_tenure and target.tenure > 0:
            # 资历被搅黄：工龄清零，本轮之后重新从 0 熬起
            o_target.tenure_reset_by_attack = target.tenure
            target.tenure = 0
            attack_msgs.append(f"{names[target_id]} 被人告了黑状，多年资历一朝清零。")

        if cfg.attack_mode == "denial":
            # 纯破坏型。政绩清零放到晋升阶段处理（那时才知道有没有真的拦下晋升），
            # 这里只处理"没打 WORK 的政绩处罚"和"撞上贪污的赃款没收"。
            #
            # "未受政绩处罚"这句要**等搜完账再决定发不发**：先播它、再播
            # "经济问题被人揭发"，公报就自相矛盾了（这一刀明明打中了）。
            penalised = False
            if not target_worked and cfg.attack_merit_penalty > 0:
                loss = min(cfg.attack_merit_penalty, target.merit)
                target.merit -= loss
                o_target.attack_merit_loss += loss
                if loss > 0:
                    attack_msgs.append(
                        f"{names[target_id]} 遭到政治攻击，损失 {loss} 点政绩。"
                    )
                    penalised = True

            # 撞上对方本轮在贪污
            haul = o_target.corrupt_amount
            if haul <= 0:
                for aid in attacker_ids:
                    outcome.outcomes[aid].private_notes.append(
                        f"{names[target_id]} 本轮手脚干净，没搜出问题。"
                    )
                if o_target.merit_promotion_blocked:
                    # 这一刀已经把人家的晋升按住了，再说"未受处罚"就是自相矛盾
                    pass
                elif not penalised:
                    attack_msgs.append(
                        f"{names[target_id]} 遭到政治攻击，但本轮埋头工作，未受政绩处罚。"
                        if target_worked
                        else f"{names[target_id]} 遭到政治攻击，但本来就没什么政绩可丢。"
                    )
                continue

            if cfg.attack_on_corruption == "confiscate_to_attacker":
                taken = min(haul, target.money)
                target.money -= taken
                o_target.money_confiscated += taken
                share = taken // len(attacker_ids)
                for aid in attacker_ids:
                    by_id[aid].money += share
                    outcome.outcomes[aid].money_from_reports += share
                    outcome.outcomes[aid].private_notes.append(
                        f"你撞上 {names[target_id]} 正在捞钱，起获赃款 {share}。"
                    )
                attack_msgs.append(f"{names[target_id]} 被当场起获赃款。")
            elif cfg.attack_on_corruption == "merit_to_attacker":
                # 抓贪腐立功：抄家是举报的活，这里攻击者只记一笔政绩；
                # 但被抓住把柄的人得自己掏钱上下打点、把事压下去。
                hush = math.floor(Fraction(haul) * cfg.attack_hush_money_ratio)
                hush = min(hush, target.money)
                if hush > 0:
                    target.money -= hush
                    o_target.hush_money_paid += hush
                    o_target.private_notes.append(
                        f"把柄被人攥住，上下打点花掉 {hush}，这事才算压下去。"
                    )
                    attack_msgs.append(f"{names[target_id]} 为了压事，破了一笔财。")
                pool = math.floor(Fraction(haul) * cfg.attack_corruption_merit_ratio)
                for aid in attacker_ids:
                    gain = math.floor(
                        Fraction(pool, len(attacker_ids))
                        * cfg.work_multiplier(by_id[aid].rank)
                        / cfg.work_multiplier(target.rank)
                    )
                    if gain <= 0:
                        continue
                    by_id[aid].merit += gain
                    outcome.outcomes[aid].merit_from_attacks += gain
                    outcome.outcomes[aid].private_notes.append(
                        f"你揭发 {names[target_id]} 的经济问题，记功 {gain} 点政绩。"
                    )
                if pool > 0:
                    attack_msgs.append(f"{names[target_id]} 的经济问题被人揭发。")
                if not penalised and hush <= 0 and pool <= 0:
                    # 搜出来的数目太小，什么也没抖落出来
                    attack_msgs.append(f"{names[target_id]} 遭到政治攻击。")
            continue

        if cfg.attack_mode == "negative_sum":
            # 负和：目标掉一大块，攻击者只拿回其中一部分，差额凭空蒸发。
            # 两人互攻则双方照吃伤害、谁都拿不到好处 —— 便宜的是没参战的第三家。
            # 伤害按"回合当量"算：2 个回合 = 目标靠 WORK 干两轮的产出
            full_damage = math.floor(
                Fraction(EXPECTED_CARD_VALUE)
                * cfg.attack_damage_turns
                * cfg.work_multiplier(target.rank)
            )
            damage = min(full_damage, merit_before_attacks[target_id], target.merit)
            if damage <= 0:
                for aid in attacker_ids:
                    outcome.outcomes[aid].private_notes.append(
                        f"{names[target_id]} 政绩本来就没多少，你的攻击扑了个空。"
                    )
                attack_msgs.append(f"{names[target_id]} 遭到政治攻击，但没什么可损失的。")
                continue
            target.merit = max(0, target.merit - damage)
            o_target.attack_merit_loss += damage
            o_target.merit_stolen_by_attackers += damage

            mutual_names: list[str] = []
            for aid in attacker_ids:
                fighting_back = target_id in attackers_of.get(aid, [])
                if fighting_back and cfg.attack_mutual_cancels_gain:
                    mutual_names.append(names[aid])
                    outcome.outcomes[aid].private_notes.append(
                        f"你和 {names[target_id]} 互相开火，两败俱伤，谁也没占到便宜。"
                    )
                    continue
                # 收益按"实际打掉多少"折算，再换成攻击者自己官职下的当量。
                # 打在没有政绩的人身上 => damage 很小 => 基本拿不到东西。
                share = math.floor(
                    Fraction(damage)
                    * cfg.attack_gain_ratio
                    / len(attacker_ids)
                    * cfg.work_multiplier(by_id[aid].rank)
                    / cfg.work_multiplier(target.rank)
                )
                by_id[aid].merit += share
                outcome.outcomes[aid].merit_from_attacks += share
                outcome.outcomes[aid].private_notes.append(
                    f"你把 {names[target_id]} 打掉了 {damage} 点政绩，自己拿到 {share} 点。"
                )
            if mutual_names:
                attack_msgs.append(
                    f"{names[target_id]} 与人互相攻讦，损失 {damage} 点政绩，两败俱伤。"
                )
            else:
                attack_msgs.append(
                    f"{names[target_id]} 遭到政治攻击，损失 {damage} 点政绩。"
                )
            continue

        if cfg.attack_mode == "steal_merit":
            # 抢走目标当前政绩存量的一部分。政绩少的人（也就是在闷声捞钱的人）
            # 本来就没什么可抢，攻击自然扑空——不需要额外写"命中条件"。
            pool = math.floor(Fraction(target.merit) * cfg.attack_steal_fraction)
            share = pool // len(attacker_ids)
            taken = share * len(attacker_ids)
            if taken <= 0:
                for aid in attacker_ids:
                    outcome.outcomes[aid].private_notes.append(
                        f"{names[target_id]} 政绩本来就没多少，你的攻击扑了个空。"
                    )
                # 这里是 steal_merit 模式，没有"戴帽子"那套罚款（penalty 是
                # steal_work 分支的局部变量，引用它会直接 UnboundLocalError）。
                if cfg.attack_announces_attacker:
                    who = _join_names([names[aid] for aid in attacker_ids], cfg)
                    attack_msgs.append(
                        f"{who} 想抢 {names[target_id]} 的功劳，"
                        f"可他政绩本来就没多少，扑了个空。"
                    )
                else:
                    attack_msgs.append(
                        f"{names[target_id]} 遭到政治攻击，但政绩本来就没多少，"
                        f"对方扑了个空。"
                    )
                continue
            target.merit = max(0, target.merit - taken)
            o_target.attack_merit_loss += taken
            o_target.merit_stolen_by_attackers += taken
            for aid in attacker_ids:
                by_id[aid].merit += share
                outcome.outcomes[aid].merit_from_attacks += share
                outcome.outcomes[aid].private_notes.append(
                    f"你从 {names[target_id]} 手里抢到了 {share} 点政绩。"
                )
            # 明攻击：把抢功的人指名道姓写进公报，被抢的人才知道该报复谁。
            # （匿名举报保持暗箭，两张牌形成明/暗对照。）
            if cfg.attack_announces_attacker:
                who = _join_names([names[aid] for aid in attacker_ids], cfg)
                verb = "把功劳揽了过去" if len(attacker_ids) == 1 else "一起把功劳分了"
                attack_msgs.append(
                    f"【抢功】{names[target_id]} 埋头苦干，"
                    f"{who} 在上级面前{verb}，抢走 {taken} 点政绩。"
                )
            else:
                attack_msgs.append(
                    f"【抢功】{names[target_id]} 埋头苦干，功劳却被人在上级面前揽了去，"
                    f"损失 {taken} 点政绩。"
                )
            continue

        if cfg.attack_mode == "steal_work":
            # 抢功：只有目标本轮真的在干活才抢得到东西，否则整个回合落空。
            # 这一条和"举报只对本轮真的贪了的人有效"完全对称。
            if not target_worked or o_target.merit_gained <= 0:
                # 没功劳可抢，但这一刀不是完全落空：他本轮**无所事事**，
                # 就按官职扣一笔政绩（官越大扣越多）。攻击者拿不到这一份——
                # 这是处罚不是转移，所以"互相攻击"仍然是两败俱伤。
                #
                # "干正事"= 埋头工作，或者凭政绩升职。
                # 凭政绩升职的人已经被这一刀拦下了（说好的"暂缓、政绩不掉"），
                # 不能转头再用这笔罚款把清零那套偷偷加回来。
                # 反过来，拿钱买官的不算干正事，照罚 —— 那是花钱不是干活。
                # 用 tried_merit_promotion 而不是 merit_promotion_blocked：
                # 后者在晋升卡被延后时（他同时还被举报）要等到第 5a 步才置位，
                # 这会儿还是 False，判据就漏了。
                did_honest_work = o_target.tried_merit_promotion
                penalty = 0 if did_honest_work else min(
                    work_merit(cfg.attack_merit_penalty, target.rank, event, cfg),
                    target.merit,
                )
                if penalty > 0:
                    target.merit -= penalty
                    o_target.attack_merit_loss += penalty

                # penalty == 0 有三种完全不同的原因，不能都说成"他政绩本来就是 0"：
                #   a) 他在忙着凭政绩升职 -> 豁免帽子，但穿小鞋那一下是命中的
                #   b) 他政绩真的是 0     -> 这一刀什么也没捞着
                #   c) 其余               -> 没在攒政绩，没功劳可抢
                if penalty > 0:
                    note = f"给他扣了顶不务正业的帽子，扣掉 {penalty} 点政绩。"
                    public = f"扣掉 {penalty} 点政绩。"
                elif did_honest_work:
                    note = "他这轮在忙着凭政绩升职，扣不了帽子——但你把他的升职按住了。"
                    public = None  # "暂缓升职"那句已经说明白了，别再补一句"白打"
                elif target.merit <= 0:
                    note = "他政绩本来就是 0，这一刀彻底落空。"
                    public = "可他政绩本来就是 0，这顶帽子扣了个空。"
                else:
                    note = "他这轮没在攒政绩，没功劳可抢。"
                    public = "但他本轮没什么功劳可抢。"

                for aid in attacker_ids:
                    outcome.outcomes[aid].private_notes.append(
                        f"{names[target_id]} 本轮没在攒政绩；{note}"
                    )
                if public is not None:
                    if cfg.attack_announces_attacker:
                        who = _join_names([names[aid] for aid in attacker_ids], cfg)
                        attack_msgs.append(
                            f"【戴帽子】{names[target_id]} 这一轮没干正事，"
                            f"{who} 参了他一本不务正业，{public}"
                        )
                    else:
                        attack_msgs.append(
                            f"【戴帽子】{names[target_id]} 这一轮没干正事，"
                            f"被人参了一本不务正业，{public}"
                        )
                continue
            pool = math.floor(Fraction(o_target.merit_gained) * cfg.attack_steal_fraction)
            pool = min(pool, target.merit)
            share = pool // len(attacker_ids)
            taken = share * len(attacker_ids)
            target.merit = max(0, target.merit - taken)
            o_target.attack_merit_loss += taken
            o_target.merit_stolen_by_attackers += taken
            for aid in attacker_ids:
                # 抢到的"功劳"按攻击者自己的官职折算，而不是原封不动搬过来：
                # 一来符合直觉（同一份功劳，官越大写进履历越漂亮），
                # 二来让两边的政绩变动数字对不上，别人没那么容易认出谁动的手。
                if cfg.attack_steal_scaled_by_rank:
                    gain = math.floor(
                        Fraction(share)
                        * cfg.work_multiplier(by_id[aid].rank)
                        / cfg.work_multiplier(target.rank)
                    )
                else:
                    gain = share
                # 往上打的额外加成（默认 0）。抢上位者本来就更值——
                # 高官一张 WORK 的绝对产出就大——这里只是再给一个显式旋钮。
                gap = target.rank - by_id[aid].rank
                if gap > 0 and cfg.attack_steal_rank_bonus:
                    gain += math.floor(
                        Fraction(gain) * cfg.attack_steal_rank_bonus * gap
                    )
                by_id[aid].merit += gain
                outcome.outcomes[aid].merit_from_attacks += gain
                outcome.outcomes[aid].private_notes.append(
                    f"你从 {names[target_id]} 手里抢到了 {gain} 点政绩。"
                )
            # 明攻击：把抢功的人指名道姓写进公报，被抢的人才知道该报复谁。
            # （匿名举报保持暗箭，两张牌形成明/暗对照。）
            if cfg.attack_announces_attacker:
                who = _join_names([names[aid] for aid in attacker_ids], cfg)
                verb = "把功劳揽了过去" if len(attacker_ids) == 1 else "一起把功劳分了"
                attack_msgs.append(
                    f"【抢功】{names[target_id]} 埋头苦干，"
                    f"{who} 在上级面前{verb}，抢走 {taken} 点政绩。"
                )
            else:
                attack_msgs.append(
                    f"【抢功】{names[target_id]} 埋头苦干，功劳却被人在上级面前揽了去，"
                    f"损失 {taken} 点政绩。"
                )
            continue

        # 规则书原版：固定扣政绩，且 WORK 玩家免疫。多人攻击按人次叠加。
        if not (target_worked and cfg.attack_spares_workers):
            penalty = cfg.attack_merit_penalty * len(attacker_ids)
            loss = min(penalty, target.merit)
            target.merit = max(0, target.merit - penalty)
            o_target.attack_merit_loss += loss
            if loss > 0:
                attack_msgs.append(
                    f"{names[target_id]} 遭到政治攻击，损失 {loss} 点政绩。"
                )
            else:
                attack_msgs.append(f"{names[target_id]} 遭到政治攻击。")
        else:
            attack_msgs.append(
                f"{names[target_id]} 遭到政治攻击，但本轮埋头工作，未受政绩处罚。"
            )

    # ---- 4. 匿名举报：判定是否查实 ----------------------------------------
    # 事件产生的举报没有举报人，查实的赃款直接充公。
    # 这里要把"真人举报"和"事件查办"分开记：后果一样，但公报得说清楚是哪一种，
    # 否则被反腐风暴扫到的人会以为桌上有人在专门盯着自己。
    for p in ordered:
        outcome.outcomes[p.id].report_count_players = len(report_actors.get(p.id, []))

    if event.flag("mass_report"):
        for p in ordered:
            report_counts[p.id] += 1
            outcome.outcomes[p.id].report_from_event = event.name
    if event.flag("storm_report"):
        for pid in storm_targets(
            {p.id: outcome.outcomes[p.id].corrupt_amount for p in ordered}, cfg
        ):
            report_counts[pid] += 1
            outcome.outcomes[pid].report_from_event = event.name

    for p in ordered:
        o = outcome.outcomes[p.id]
        count = report_counts.get(p.id, 0)
        if count <= 0:
            continue
        o.reported = True
        o.report_count = count
        kind = classify_report(o.corrupt_amount, cfg, bribe=o.pending_bribe)
        if kind is DemotionKind.NONE:
            # 目标本轮没有贪污 -> 举报无效
            if o.report_count_players:
                o.private_notes.append("有人举报了你，但查无实据。")
            else:
                o.private_notes.append(f"{o.report_from_event}扫到了你，但查无实据。")
            continue
        o.report_effective = True
        o.demotion = kind
        # 被查办的人自己也要知道是"有人盯上我"还是"运气不好撞上风暴"——
        # 搞错了会去报复一个根本不存在的敌人
        if o.report_count_players and o.report_from_event:
            o.private_notes.append(
                f"有人举报了你，偏偏又赶上{o.report_from_event}，两头都没躲过。"
            )
        elif o.report_count_players:
            o.private_notes.append("有人举报了你，查实了。")
        else:
            o.private_notes.append(
                f"没人举报你，是{o.report_from_event}把你扫进去了——贪得太扎眼。"
            )

    # 记录"没收与降级之前"的晋升资格。规则书第 12 节（攻击阻断政绩晋升）和
    # 第 16 节（攻击 + 举报）都是按这个时点判定的。
    for p in ordered:
        o = outcome.outcomes[p.id]
        elig_money_pre = can_promote_by_money(p, cfg)
        elig_merit_pre = can_promote_by_merit(p, cfg)
        if o.attacked and elig_merit_pre:
            o.merit_promotion_blocked = True
        if o.attacked and o.report_effective and elig_money_pre and elig_merit_pre:
            o.promotion_blocked_by_attack_report = True

    # ---- 5. 没收 + 降职警告 --------------------------------------------------
    # 查实的后果不再是"一次被抓就打回基层"那种断崖，而是一张分期账单：
    #   * 本轮贪污所得全部没收
    #   * 拿钱买的官作废，而且**钱照样没了**——礼送出去了，事没办成
    #   * 记一次降职警告，工龄清零
    #   * 警告攒够 WARNINGS_BEFORE_DEMOTION 次才降一级，然后警告清空
    # 先把所有没收金额算完再统一扣，这样"甲举报乙、乙同时举报甲"能同时结算，
    # 不会因为先后顺序让某一方多拿或少拿。
    confiscated: dict[int, int] = {}
    for p in ordered:
        o = outcome.outcomes[p.id]
        if not o.report_effective or not cfg.report_reward_enabled:
            continue
        # 本轮贪的全部吐出来（存款不动——那是以前"重大贪腐抄家"的活）
        confiscated[p.id] = min(o.corrupt_amount, p.money)

    for p in ordered:
        o = outcome.outcomes[p.id]
        if not o.report_effective:
            continue
        before = p.rank

        taken = confiscated.get(p.id, 0)
        if taken:
            p.money = max(0, p.money - taken)
            o.money_confiscated = taken

        # 行贿的钱打水漂：官没升成，钱也要不回来。
        # 这笔钱同样是查获的赃款，要并进分赃池——
        # 不然"抓到一个光买官没贪钱的"，举报人查实了却一分钱拿不到。
        bribe = min(o.pending_bribe, p.money)
        if bribe:
            p.money -= bribe
            o.bribe_lost = bribe
            confiscated[p.id] = confiscated.get(p.id, 0) + bribe

        issued = warnings_for(o.demotion, cfg)
        o.warnings_issued = issued
        p.warnings += issued
        p.tenure = 0  # 任何一次警告都把工龄清零

        demoted = False
        while p.warnings >= cfg.warnings_before_demotion:
            p.warnings -= cfg.warnings_before_demotion
            apply_demotion(p, DemotionKind.MINOR, cfg)
            demoted = True
        # 已经在基层的人降无可降：官职没动，就别播"由基层降为基层"
        hit_the_floor = demoted and p.rank == before
        o.warnings_after = p.warnings
        o.demotion = DemotionKind.MINOR if demoted else DemotionKind.NONE

        how = _report_source_phrase(o)
        bits = []
        if taken:
            bits.append(f"赃款 {taken} 全部没收")
        if bribe:
            bits.append(f"行贿的 {bribe} 打了水漂、官也没升成")
        detail = "，".join(bits)
        if demoted and hit_the_floor:
            report_msgs.append(
                f"{names[p.id]} 因经济问题{how}，{detail + '，' if detail else ''}"
                f"警告记满——但已经在{cfg.rank_name(p.rank)}，再降无可降。"
            )
        elif demoted:
            report_msgs.append(
                f"{names[p.id]} 因经济问题{how}，{detail + '，' if detail else ''}"
                f"警告记满，由{cfg.rank_name(before)}降为{cfg.rank_name(p.rank)}。"
            )
        else:
            left = cfg.warnings_before_demotion - p.warnings
            report_msgs.append(
                f"{names[p.id]} 因经济问题{how}，{detail + '，' if detail else ''}"
                f"记降职警告一次（再记 {left} 次就要降级）。"
            )

    # 分赃：没收的赃款里只有 report_reward_ratio 归举报人，其余充公；
    # 多人举报再平分，向下取整，零头也充公。
    # 举报本身已经能降级 + 没收 + 冻结晋升了，拿钱这块要收着点。
    for victim_id, taken in confiscated.items():
        actors = report_actors.get(victim_id, [])
        if taken <= 0 or not actors:
            continue
        payout = math.floor(Fraction(taken) * cfg.report_reward_ratio)
        share = payout // len(actors) if cfg.report_reward_split_evenly else payout
        if share <= 0:
            continue
        for actor_id in actors:
            by_id[actor_id].money += share
            outcome.outcomes[actor_id].money_from_reports += share

    # 告知举报者结果（只说查实与否 + 自己分到多少，不泄露对方的家底）
    for p in ordered:
        o_actor = outcome.outcomes[p.id]
        for target_id, actors in sorted(report_actors.items()):
            if p.id not in actors:
                continue
            if not outcome.outcomes[target_id].report_effective:
                o_actor.private_notes.append(
                    f"你对 {names[target_id]} 的举报没有查到问题。"
                )
            elif o_actor.money_from_reports > 0:
                o_actor.private_notes.append(
                    f"你的举报查实了，分得赃款 {o_actor.money_from_reports}。"
                )
            else:
                o_actor.private_notes.append("你的举报查实了，但没有分到赃款。")

    # ---- 5a. 补结算：排在举报/攻击后面的晋升卡 ------------------------------
    for p, card in deferred_promotions:
        o = outcome.outcomes[p.id]
        if o.report_effective:
            # 查实了就升不了：光记警告没降级也一样，钱还白花了
            o.promotion_card_played = True
            o.promotion_frozen_by_report = True
            o.private_notes.append(
                "举报查实，这一轮的晋升作废了"
                + ("——行贿的钱也没能要回来。" if o.bribe_lost else "。")
            )
            continue
        if o.demotion is not DemotionKind.NONE and cfg.demoted_cannot_promote_same_round:
            o.promotion_card_played = True
            o.private_notes.append("刚被打下来，这一轮的晋升卡没法用了。")
            continue
        _resolve_promotion_card(
            p, card, o, cfg, names, promo_msgs, already=o.promotion
        )

    # ---- 5b. 自动晋升（仅 promotion_requires_card=False 的规则书原版）---------
    # 没有晋升卡就没有"玩家选的位置"，所以沿用老规矩：所有事情尘埃落定后再结算。
    if not cfg.promotion_requires_card:
        for p in ordered:
            o = outcome.outcomes[p.id]
            if o.promotion is not PromotionKind.NONE:
                continue
            if o.report_effective:
                continue
            if o.demotion is not DemotionKind.NONE and cfg.demoted_cannot_promote_same_round:
                continue
            _resolve_promotion_card(
                p, Card.PROMOTE_ANY, o, cfg, names, promo_msgs,
                already=PromotionKind.NONE, auto=True,
            )

    # ---- 5b2. 统计每个举报者有没有举报中 -------------------------------------
    # 和攻击同理："没分到钱"不等于"白打"：把人从省级打回基层本身就是战果，
    # 而对方可能早就把钱花光了，抄无可抄。
    for victim_id, actors in report_actors.items():
        if not outcome.outcomes[victim_id].report_effective:
            continue
        for aid in actors:
            outcome.outcomes[aid].reports_landed += 1

    # ---- 5c. 统计每个攻击者到底有没有打中 ------------------------------------
    # "拿到政绩"不等于"打中"：denial 模式下最有价值的一刀是拦下别人的晋升，
    # 而那一刀攻击者一分政绩都拿不到。只看收益的话复盘会说"你的攻击白费了"。
    for target_id, attacker_ids in attackers_of.items():
        ot = outcome.outcomes[target_id]
        landed = (
            ot.merit_promotion_blocked
            or ot.merit_wiped_by_attack > 0
            or ot.attack_merit_loss > 0
            or ot.hush_money_paid > 0
            or ot.tenure_reset_by_attack > 0
            or ot.merit_stolen_by_attackers > 0
        )
        if not landed:
            continue
        for aid in attacker_ids:
            outcome.outcomes[aid].attacks_landed += 1

    # ---- 6. 晋升相关的提示 --------------------------------------------------
    # 晋升本身已经在第 2 步按玩家自选的出牌顺序结算完了，这里只补几句说明。
    for p in ordered:
        o = outcome.outcomes[p.id]
        if o.promotion is PromotionKind.NONE:
            attempted = o.promotion_card_played or not cfg.promotion_requires_card
            if o.promotion_blocked_by_attack_report and attempted:
                # 规则书第 16 节：攻击堵死政绩那条路，举报堵死金钱那条路。
                # 没打晋升卡就没有"泡汤"这回事——本来也没在升。
                promo_msgs.append(
                    f"【穿小鞋】{names[p.id]} 政绩这条路被人放黑料挡住，"
                    f"转走门路又被匿名举报，本轮晋升泡汤。"
                )
            elif (
                o.merit_promotion_blocked
                # 没打晋升卡就没有"被拦下"这回事——本来也升不了，
                # 播这句会让人以为自己挨了一刀，其实那一刀打空了
                and attempted
                # "眼看就要上位被一状告倒"那句已经把这件事说透了，别再补一句弱的
                and not o.merit_wiped_by_attack
            ):
                smear = (
                    rng.choice(cfg.attack_smear_rumors)
                    if cfg.attack_smear_rumors else "放出黑料"
                )
                if cfg.attack_announces_attacker and o.attacked_by:
                    who = _join_names([names[a] for a in o.attacked_by], cfg)
                    promo_msgs.append(
                        f"【穿小鞋】{names[p.id]} 本要凭政绩升职，"
                        f"{who} 放出黑料，{smear}——升职暂缓。"
                    )
                else:
                    promo_msgs.append(
                        f"【穿小鞋】{names[p.id]} 本要凭政绩升职，"
                        f"有人放出黑料，{smear}——升职暂缓。"
                    )


        # ---- 7. 工龄 ----
        if o.promotion is not PromotionKind.NONE:
            p.tenure = 0
        elif o.report_effective:
            p.tenure = 0  # 任何一次降职警告都把工龄清零（降级与否都一样）
        else:
            p.tenure += 1
            tenure_ceiling = (
                cfg.president_rank
                if cfg.tenure_can_reach_president
                else cfg.president_rank - 1
            )
            if p.tenure >= cfg.tenure_required and p.rank < tenure_ceiling:
                # 熬上去的也是升职，政绩照样 /5——不然"不打晋升卡光攒政绩"
                # 反而能靠工龄白拿一级又不掉政绩
                merit_before = p.merit
                apply_promotion_costs(p, cfg)
                o.merit_before_promotion = merit_before
                o.promotion_merit_decay = merit_before - p.merit
                p.tenure = 0
                o.promotion = PromotionKind.TENURE
                promo_msgs.append(
                    f"{names[p.id]} 资历深厚，按工龄晋升为{cfg.rank_name(p.rank)}。"
                )

        o.rank_after = p.rank
        o.tenure_after = p.tenure
        o.money_after = p.money
        o.merit_after = p.merit

    # ---- 财富广播 --------------------------------------------------------
    # 用"落袋"的净额，不是毛收入：当场被没收/抄家的人不该还被传"住上洋房"
    msgs, top_names, tier = wealth_broadcast_detail(
        ((names[pid], o.net_corrupt_gain) for pid, o in outcome.outcomes.items()),
        cfg, rng,
    )
    outcome.wealth_broadcast = msgs
    outcome.wealth_top_ids = [pid for pid in sorted(names) if names[pid] in top_names]
    outcome.wealth_tier = tier

    outcome.public_messages = attack_msgs + report_msgs + promo_msgs
    outcome.presidents = [p.id for p in ordered if p.rank >= cfg.president_rank]
    return outcome
