"""会推理的 AI 玩家。

和 analysis.py 里那些启发式 bot 最大的区别：

* 它**只吃 `game.public_state()` 和自己的 `game.private_state(pid)`**，
  也就是浏览器里那个玩家能看到的一模一样的 JSON。作弊在结构上就不可能。
* 它维护一份对每个对手**金钱的估计**。金钱是隐藏的，但可以从公开信息推断：
    - 坊间传闻点名了谁（只说"这轮他到手的钱最多"，不报金额）。
      工资是公开可算的，所以这条给出一个**硬下界**：
      被点名的人，脏钱至少 = 全场最高工资 - 他自己的工资
    - 某人政绩没涨 => 他本轮没打 WORK => 有可能在贪
    - 政绩涨了也**不能**判他清白：一轮打两张牌，可以工作+贪污，
      而以权谋私本身就同时给政绩和钱，看起来和埋头干活一模一样
    - "四处打点"晋升 => 他的钱至少够门槛，晋升后被 /5 砍掉
    - 被举报打回基层 => 他的钱清零
* 政绩是**公开**的，所以它知道谁快要靠政绩晋升了。
  ATTACK 只在"目标本轮真的可能靠政绩升官"的时候才打——这是攻击唯一的实质价值。

决策方式是显式的期望收益比较，单位统一成"官职"：
1.0 = 升一级。所有权重都在 Weights 里，方便调。
"""

from __future__ import annotations

import math
import random
from collections import Counter
from itertools import combinations
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import rules
from config import Config, DEFAULT_CONFIG
from models import Card

EXPECTED_CARD_VALUE = 10  # 兜底用的牌面期望（正常情况下用发牌时摇好的真实点数）

# "这桌最近的风气"的起始值和衰减（见 SmartAgent._update_table）。
# 起始值取 AI 混战里的平均水平；每轮新观察占 1 - TABLE_DECAY，越近的轮次分量越重。
TABLE_DECAY = 0.6
TABLE_PRIOR = {
    "rep": 0.25,      # 每轮有多大比例的人被玩家举报（没查实也公开）
    "atk": 0.25,      # 每轮有多大比例的人挨了攻击
    "corrupt": 0.3,   # 有人贪了没被抓（传闻响了）/ 有人被查实
    "work": 0.6,      # 每轮有多大比例的人政绩在涨（攻击能抢到东西的人）
    "me_atk": 0.2,    # 我自己最近挨打的频率
    "me_rep": 0.2,    # 我自己最近被举报的频率
}

# 「他这一轮在贪 / 在掏钱升职」的估计：按数据拟合，不再拍脑袋。
# 来源：400 局六 AI 混战，每轮每一对（观察者, 对手）记一条——AI 看得到的特征
# vs 对手这一轮实际干了什么（`python3 audit.py --features 400`，混脚本打法加 --bots）。
# 每个因子是"这一档的发生率 ÷ 总体发生率"。几个特征彼此相关（轮次早的人也穷），
# 直接连乘会重复计数，所以乘完再开 DAMPING 次方。
# 拟合之前的老规则估"在贪"平均 52%、实际 21%，而且估高估低实际都在 20% 上下——几乎没有信息量；
# 还有一条反过来了：老规则认为"钱快够了就会去贪"，实际上钱够门槛的人只有 13% 在贪，最穷的 38%。
READ_DAMPING = 0.65
# 第三次拟合：对手换成学习型 AI 之后（audit.py --features 480 --audit-policy policies/best.json）。
# 学习型 AI 60% 的轮次在贪、47% 在掏钱升职——上一版表（按手写 AI 拟合）以为只有两成，差了三倍，
# 好几条还反过来了：经常被点名的人反而更收敛（学会了"被盯上就收手"）。
# 真人和 AI 打法不一样，所以这张表只是起点，对局里还会按"被举报的人实际查实了多少"自动校准
# （见 SmartAgent._read_calibration）。
CORRUPT_BASE = 0.60
BRIBE_BASE = 0.47
CORRUPT_BY_MONEY = [(0.3, 1.07), (0.6, 1.03), (1.0, 0.96), (float("inf"), 0.82)]  # 估钱 ÷ 门槛
BRIBE_BY_MONEY = [(float("inf"), 1.0)]
CORRUPT_BY_ROUND = [(4, 1.16), (8, 0.87), (99, 0.70)]
BRIBE_BY_ROUND = [(4, 1.15), (8, 0.89), (99, 0.75)]
CORRUPT_BY_QUIET = [(9.0, 1.0)]  # 不干活的比例：学习型 AI 身上没区分度
BRIBE_BY_QUIET = [(0.2, 1.02), (0.4, 0.87), (0.6, 0.99), (9.0, 1.16)]
CORRUPT_BY_RANK = [1.17, 1.13, 0.77, 0.84]  # 基层 / 县级 / 市级 / 省级
BRIBE_BY_RANK = [1.03, 1.24, 1.19, 0.5]
# 出身：学习型 AI 会按出身打（会计敢贪、卷王爱干活、富二代爱买官），读别人时也要看出身
CORRUPT_BY_ORIGIN = {"ACCOUNTANT": 1.2, "GRINDER": 0.73, "OFFICIAL": 0.96,
                     "PEASANT": 1.06, "RED": 1.01, "RICH": 0.98}
BRIBE_BY_ORIGIN = {"ACCOUNTANT": 1.06, "GRINDER": 0.81, "OFFICIAL": 0.92,
                   "PEASANT": 0.98, "RED": 1.01, "RICH": 1.22}
# 个人档案 = (被传闻点名或被查实的轮数 + 1) / (观察轮数 + 4)。
# 学习型 AI 被盯上就收手，所以档案高的人反而少贪；但真人里有"一路贪到底"的
# （混脚本打法拟合时档案 0.7 的人 77% 在贪），所以最高那档留一个往上翘的尾巴。
CORRUPT_BY_RECORD = [(0.3, 1.0), (0.5, 0.88), (0.7, 0.75), (9.0, 1.5)]
BRIBE_BY_RECORD = [(0.7, 1.0), (9.0, 1.5)]


# 按出身各复制一份的特征（见 _combo_features）
ORIGIN_CROSSED = ("dirty", "money_gain", "merit_gain", "promote_money", "promote_merit",
                  "n_attack", "n_report", "risk", "family", "econ", "dirty_x_rep", "intf_x_contender")


def _band(table, x):
    for upper, factor in table:
        if x < upper:
            return factor
    return table[-1][1]


def _early_promotion_rank(cfg, rank, money, merit, cards, origin=None):
    """如果这组牌能在"动手之前"就把官升了，返回 (新官职, 升职后的钱, 升职后的政绩)。

    要和 rules.apply_promotion_costs 保持一致；AI 不这么算就会误判组合技的价值。
    """
    if not any(c.is_promotion for c in cards) and cfg.promotion_requires_card:
        return None
    mc = rules.money_cost_at(rank, origin, cfg)
    tc = rules.merit_cost_at(rank, origin, cfg)
    if mc is None or tc is None:
        return None
    merit_card = any(c.can_use_merit for c in cards) or not cfg.promotion_requires_card
    money_card = any(c.can_use_money for c in cards) or not cfg.promotion_requires_card

    def shrink(left, divisor):
        return 0 if left <= 0 else -(-left // divisor)

    def merit_after(left):
        """政绩的衰减：开关打开时，怎么升上去的都要 /5。"""
        if cfg.promotion_always_decays_merit:
            return shrink(left, cfg.merit_overflow_divisor)
        return max(0, left)

    if cfg.needs_both(rank):
        if merit >= tc and money >= mc:
            return (rank + 1,
                    shrink(money - mc, cfg.money_overflow_divisor),
                    shrink(merit - tc, cfg.merit_overflow_divisor))
        return None
    if merit_card and merit >= tc:
        return rank + 1, money, shrink(merit - tc, cfg.merit_overflow_divisor)
    if money_card and money >= mc:
        # 贿赂上位也会把政绩打掉，AI 不算这一笔就会高估"花钱升职"
        return rank + 1, shrink(money - mc, cfg.money_overflow_divisor), merit_after(merit)
    return None


@dataclass
class Weights:
    """AI 的性格参数。调这些可以造出不同风格的对手。"""

    promotion_bonus: float = 0.35  # 真的升上去（官职是永久的）额外值多少
    # 干扰收益的整体系数。0.2 是**校准**出来的：把权重从 1.0 往下扫，
    # 直到"禁用这张牌的 AI"不再打得更好（消融差归零）。1.0 时 AI 会高估干扰约 5 倍，
    # 出牌率虚高到 114%，但实际是净负收益。
    # 0.2 -> 0.15：给"拦住领跑者"加权之后（见 _share），攻击的估值整体抬高了，
    # 消融差从 +1.20 恶化到 +2.20（4 个种子 x 1500 局的均值，>0 = 负收益）。
    # 选靶子变准是好事，但"更常出手"不是——这里把频率旋钮调回去，
    # 重测 +0.90，比改之前还准一点。0.12 会矫枉过正（-1.02）。
    interfere_attack: float = 0.15
    # 0.2 -> 0.16：同上，给"拦领跑者"加权后举报也被整体高估，
    # 消融差从 -0.54 恶化到 +1.55（6 种子 x 1500 局），出牌率虚高到 63%。
    # 拧到 0.16 重测 +0.04，出牌率回到 53%。
    interfere_report: float = 0.16
    caught_dread: float = 1.0  # 对"贪污被抓"的恐惧程度
    # 我贪了之后被查实的先验概率（含反腐风暴）。体检数据：AI 互打时实际约 35%。
    # 以前是 0.22，分母（估计全场贪了几次）又被老的瞎猜规则吹大，
    # AI 自以为只有 7%~25% 风险的那些贪污，实际 35% 被查实——贪得太放心。
    base_report_pressure: float = 0.35
    # 我是明面领先者时的放大倍数。体检：领先者被查实 52%、其他人 35%，约 1.4 倍（以前拍的 2.0）
    leader_suspicion: float = 1.4
    # 威胁评估：干扰一个离赢还很远的人值几折。
    # 0.45 太平了（基层 0.45 vs 省级 1.0，只差 2.2 倍），AI 会无差别攻击。
    # 配合 threat_exponent=2 之后差距拉到约 12 倍，攻击自然集中到快赢的人身上。
    threat_floor: float = 0.08
    threat_exponent: float = 2.0
    # 拦住一个下一级就是国家主席的人，等于直接阻止别人赢下整局，
    # 这份好处不该按"全桌平分"打折——他赢了我就全输。
    endgame_block_weight: float = 4.0
    # 拦登顶那份价值的量纲。以前直接借用 interfere_attack，于是"平时少攻击一点"
    # 会连带"终局少拦一点"——两件事要分开调。0.15 = 借用时的取值，终局行为不变。
    endgame_scale: float = 0.15
    # 拦登顶的人值多少，按"拦下之后我能不能接着赢"打折：比剩下的人里最近的那个每落后一级，
    # 打 exp(-contender_decay) 折（见 _contender）。0 = 不分名次人人一样拼命拦
    contender_decay: float = 1.5
    # "别人眼里的我有多可疑"对自己被查实风险的影响力度（0 = 关，1 = 全量，见 _self_suspicion）
    self_suspicion: float = 1.0
    # "他在贪"的读数往全桌平均（CORRUPT_BASE）收多少（0 = 原样，1 = 完全不看读数）。
    # 人人都知道自己看上去可不可疑之后（_self_suspicion），读数最高的那些人反而最不敢贪：
    # 体检里读数 60%~80% 的人实际只有 3% 在贪，基层被举报的预估命中 77%、实际 7%。
    # 这是捉迷藏的均衡——越好猜的人越会收手，所以高读数不能全信。
    read_shrink: float = 0.0
    # 判断"他下一步就夺冠"时，钱这一项按门槛的几折算。
    # 钱是暗的，money_est 是下界（贪污看不见、以权谋私伪装成干活），
    # 1.0 意味着只信估计值，结果就是终局刹车形同虚设。
    endgame_money_doubt: float = 0.6
    # 有人这一轮就可能登顶、而我自己这轮当不上主席时，我自己的生产/晋升值几折。
    # 他一赢游戏就结束，我这轮捞的钱、升的官全都兑现不了。
    # 和 _cash_horizon 给赃款打的折一致。不打这个折，复盘 QN78 第 10 轮：
    # 老张手里有举报牌、明知真人要登顶，还是选了"贪污 + 基层升县级"（0.85 分）
    # 而不是举报他（0.32 分）——那一举报本来能拦下整局。
    endgame_economy_discount: float = 0.15
    # 买官被举报查实时，打水漂的那笔钱值多少（单位：官职进度）。
    # 正好是一级的金钱门槛 = 1.0。
    bribe_caught_loss: float = 1.0
    # 我去攻击贫农时，估计还有别人也在打他的概率（贫农只挡得住一个人）。
    # 看不见别人出什么牌，取个中间值；快登顶的人往往被好几个人一起打，偏高一点也合理。
    peasant_second_attacker: float = 0.5
    # 官二代「透风」知道这轮是反腐风暴时，贪污被查的概率按多少算
    tipoff_storm_caught: float = 0.7
    # 红二代「一纸调令」：每局一次的卡，打出去的机会成本；用钱升被查实时记警告的代价
    family_card_reserve: float = 0.2
    # 干扰牌"自己拿到的好处"（抢功、分赃）按目标威胁加权时的下限：
    # 打桌上威胁最大的人拿满，威胁最小的也至少算这么多
    target_focus_floor: float = 0.15
    # 威胁值里"本级进度"占多重（1 = 进度和官职一样算）。0.5 = 官职为主
    threat_progress_weight: float = 0.5
    family_warning_cost: float = 0.15
    # 拦一个这一轮就可能登顶的人，单靠一张牌能拦下的把握。
    # 省级 -> 主席那一级"走哪条路由卡决定"：攻击只按得住政绩升职，举报只抓得住贿赂升职，
    # 通用升职被攻击会改走金钱、还得再挨一张举报。他手里是哪张看不见，
    # 而真人在门口往往两张晋升卡一起打（复盘 QN78 第 10 轮就是）——
    # 单张牌只能赌中一半，攻击 + 举报一起压上去才稳。
    endgame_cover_single: float = 0.4
    endgame_cover_both: float = 0.95
    # 他钱还不够、这一轮得现贪才凑得齐时，举报单张就几乎稳拦：
    # 贪了 -> 查实 -> 排在贪污后面的晋升卡全部冻结。
    endgame_cover_report_must_corrupt: float = 0.8
    noise: float = 0.02  # 决策噪声，避免完全可预测
    # 目标评分差在**最高分的这个比例**之内视为打平，打平就随机挑。
    # 必须用相对值：绝对阈值（原来是 0.02）在开局能用，到了中局就坏了——
    # 那时各目标的攻击分普遍只有 0.006~0.016，整个区间比阈值还小，
    # 于是"谁快升职了"这套威胁模型算完就被扔掉，变成纯随机挑人。
    tie_epsilon: float = 0.05
    endgame_round: int = 8  # 从第几轮开始按"打满 10 轮比钱"的口径考虑


@dataclass
class OpponentModel:
    money_est: float = 0.0
    work_rounds: int = 0  # 观测到他打 WORK 的轮数
    quiet_rounds: int = 0  # 观测到他政绩没涨的轮数（在贪 / 在举报 / 在攻击）
    observed_rounds: int = 0
    last_seen_corrupting: int = -99  # 最后一次被财富广播点名的轮次
    # 他贪污的证据量：被广播点名记 1，全场无广播记 0，说不清的记部分
    corrupt_evidence: float = 0.0
    dirty_rounds: int = 0  # 被传闻点名、或者被查实的轮数（个人档案，见 CORRUPT_BY_RECORD）

    @property
    def record(self) -> float:
        return (self.dirty_rounds + 1.0) / (self.observed_rounds + 4.0)

    @property
    def work_rate(self) -> float:
        """他打 WORK 的频率。拉普拉斯平滑，避免看一轮就下死结论。"""
        return (self.work_rounds + 1.0) / (self.observed_rounds + 2.0)

    @property
    def quiet_rate(self) -> float:
        """他没在积累政绩的频率（在贪 / 在举报 / 在攻击）。"""
        return 1.0 - self.work_rate

    @property
    def corrupt_rate(self) -> float:
        """他每轮贪污的频率。这是举报该不该打的核心依据。

        关键证据来自"财富广播完全没响"——那等于全场确认本轮无人贪污。
        """
        return (self.corrupt_evidence + 0.4) / (self.observed_rounds + 2.0)


class SmartAgent:
    """一个只看公开信息 + 自己私密状态的思考型玩家。"""

    def __init__(
        self,
        player_id: int,
        cfg: Config = DEFAULT_CONFIG,
        weights: Weights | None = None,
        rng: random.Random | None = None,
        allow_attack: bool = True,
        allow_report: bool = True,
        allow_corrupt: bool = True,
    ) -> None:
        self.id = player_id
        self.cfg = cfg
        self.w = weights or Weights()
        self.rng = rng or random.Random()
        self.allow_attack = allow_attack
        self.allow_report = allow_report
        self.allow_corrupt = allow_corrupt

        self.models: dict[int, OpponentModel] = {}
        self._prev_players: dict[int, dict[str, Any]] = {}
        self._last_round_seen = 0
        self.decision_log: list[str] = []  # 调试用
        # 上一次 decide 的不含噪声打分 [(分数, 牌), ...]，复盘工具读它
        self.last_scores: list[tuple[float, list[Card]]] = []
        self.last_prediction: dict[str, Any] = {}  # audit.py 对账用（见 _predictions）
        # 全场层面的自适应估计：这桌人到底抓得有多凶
        self.est_corrupt_attempts = 0.0  # 估计发生过多少次贪污
        # 别人眼里的我：只用公开信息、按和对手一样的规则更新
        self.self_model = OpponentModel()
        self.table: dict[str, float] = dict(TABLE_PRIOR)
        self.want_features = False  # LearnedAgent 打开：score_combos 顺便产出每个组合的特征
        self.last_features: list[dict[str, float]] = []
        self._econ_detail: dict[str, float] = {}
        self._seen_public: dict[str, Any] | None = None
        self._seen_private: dict[str, Any] | None = None
        # 上一轮出牌时，我估的每个对手"这一轮在贪"的概率；下一轮看到结算后累加进上面那个数
        self._pending_corrupt_expect: dict[int, float] = {}
        # 读法的对局内校准（见 _read_calibration）：被举报的人实际查实几个 vs 我当初估了多少
        self._pending_hit_expect: dict[int, float] = {}
        self._cal_hits = 0.0
        self._cal_expect = 0.0
        self._times_attacked = 0  # 我自己被攻击过几次
        self._last_attackers: dict[int, int] = {}  # 谁攻击过我 -> 最近一次的轮次（明攻击才公开）
        self._rounds_seen = 0
        self._my_rank = 0
        self.observed_demotions = 0  # 公开可见的"被查实"次数（举报压力的证据）

    # ------------------------------------------------------------------
    # 观察：把公开 payload 变成对隐藏金钱的估计
    # ------------------------------------------------------------------

    def observe(
        self, public: dict[str, Any], private: dict[str, Any] | None = None
    ) -> None:
        """把公开结算翻译成对每个对手的估计。

        `private` 是自己的私密 payload，可以不给。给了的话多一条很强的信息：
        **我对自己这轮挣了多少是完全知情的**，而传闻排的是总收入，
        所以"别人被点名"就等于"他的收入 >= 我的收入"——
        这条下界通常比"他的收入 >= 全场最高工资"紧得多。
        """
        players = {p["id"]: p for p in public["players"]}
        self._seen_public, self._seen_private = public, private  # 给 _contender 用
        for pid in players:
            if pid != self.id:
                self.models.setdefault(pid, OpponentModel())

        result = public.get("last_result")
        if result is None or result.get("round") == self._last_round_seen:
            self._prev_players = players
            return
        self._last_round_seen = result["round"]

        facts = {f["player_id"]: f for f in result.get("player_facts", [])}
        self._update_table(facts, players, result)
        self._rounds_seen += 1
        # 全场这轮大概贪了几次：用出牌时那份校准过的估计，而不是事后的先验证据
        # （事后证据里"没干活 = 0.45 次贪污"这种先验把分母吹大，被查实的概率就被摊薄了）
        self.est_corrupt_attempts += sum(
            p for pid, p in self._pending_corrupt_expect.items() if pid in facts
        )
        self._pending_corrupt_expect = {}
        for pid, p in self._pending_hit_expect.items():
            f = facts.get(pid)
            if f and f.get("reported_by_player"):
                self._cal_expect += p
                self._cal_hits += float(self._caught(f))
        self._pending_hit_expect = {}
        if facts.get(self.id, {}).get("attacked"):
            self._times_attacked += 1
        for a in facts.get(self.id, {}).get("attacked_by", []) or []:
            self._last_attackers[a] = result["round"]
        top_ids = set(result.get("wealth_top_ids") or [])
        # 全场最高工资。传闻排的是"工资 + 没被查实的贪污款项（毛额，打点不扣；
        # 被查实的贪污记 0）+ 举报分到的赃款"，
        # 而工资人人算得出来，所以这个数是下面两条边界的基准。
        top_salary = max(
            (self.cfg.salary(f["rank_before"]) for f in facts.values()), default=0
        )
        # 我自己这轮到手多少？完全知情，作为"可证实的最高收入"的候选之一。
        my_income = 0
        mine = (private or {}).get("private_result") or {}
        if mine:
            my_income = int(mine.get("salary", 0)) + (
                0 if mine.get("report_effective") else int(mine.get("corrupt_amount", 0))
            ) + int(mine.get("money_from_reports", 0))
        # 能证实的最高收入：每个人至少拿到自己那份工资，而我自己的是精确值
        known_income = max(top_salary, my_income)

        # 榜首这轮脏钱有多少？先估出来，因为它同时是**所有其他人的上界**（见下）。
        # 被查实的榜首是光凭工资上的榜，他计入传闻的脏钱是 0。
        top_dirty = max(
            (
                self._dirty_prior(f, players.get(pid), self._prev_players.get(pid))[1]
                for pid, f in facts.items()
                if pid in top_ids and not self._caught(f)
                and players.get(pid) and self._prev_players.get(pid)
            ),
            default=0.0,
        )
        top_dirty = max(
            top_dirty,
            max(
                (
                    float(known_income - self.cfg.salary(f["rank_before"]))
                    for pid, f in facts.items()
                    if pid in top_ids
                ),
                default=0.0,
            ),
        )

        # 自己也按同一套规则建一份"别人眼里的我"（见 _self_suspicion）
        for pid, model in [*self.models.items(), (self.id, self.self_model)]:
            cur, prev = players.get(pid), self._prev_players.get(pid)
            fact = facts.get(pid)
            if cur is None or prev is None or fact is None:
                continue
            model.observed_rounds += 1
            model.dirty_rounds += (pid in top_ids) or self._caught(fact)
            rank_before = fact["rank_before"]

            # --- 他本轮打的是不是 WORK？政绩是公开的，直接看涨了没 ---
            merit_gain = cur["merit"] - prev["merit"] + fact["attack_merit_loss"]
            if fact["promotion"] == "MERIT":
                worked = True  # 政绩晋升会把池子清空，涨幅看不出来，但必然在积累政绩
            else:
                worked = merit_gain > 0
            if worked:
                model.work_rounds += 1
            else:
                model.quiet_rounds += 1

            # --- 他这轮贪了多少？ ---
            evidence, gained = self._dirty_prior(fact, cur, prev)

            # 再看坊间传闻。文案不再报金额（以前分四档，等于把区间念出来），
            # 所以点名本身只说明"他这轮到手的钱全场最多"。
            # 但**工资是公开可算的**，于是能挤出两条边界。
            my_salary = self.cfg.salary(rank_before)
            if self._caught(fact):
                # 被查实的人贪污那项在传闻里记 0：点没点他的名都和脏钱无关，两条边界都不适用
                pass
            elif pid in top_ids:
                # 下界：他的总收入 >= 任何我能证实的收入，取最大的那个
                #   —— 每个人至少有自己那份工资，而**我自己这轮挣了多少我最清楚**。
                #    => 他的脏钱 >= 那个数 - 他自己的工资
                # 光看工资的话，官最大的那个被点名说明不了什么；
                # 但只要我自己这轮捞了一笔而他还是压过我，他就一定也捞了。
                floor_dirty = float(max(0, known_income - my_salary))
                if floor_dirty > 0:
                    gained = max(gained, floor_dirty, self._expected_corrupt(rank_before))
                    evidence = 1.0
                    model.last_seen_corrupting = result["round"]
                else:
                    # 证明不了，但有传闻就说明这轮**确实有人**捞了钱没被抓（光拿工资不传），
                    # 而被点名的就是到手最多的那个——真人一般就认定是他了。
                    # 以前这里什么都不做，只留先验里那 0.45 × 期望：复盘 5A9J 里
                    # 真人以权谋私连着被点名，AI 每次只给他记三四块，存款越估越少。
                    gained = max(gained, self._expected_graft_money(rank_before))
                    evidence = max(evidence, 0.7)
            else:
                # 上界：他没被点名，所以 他的工资+脏钱 <= 榜首的工资+脏钱
                #    => 他的脏钱 <= 榜首脏钱 + (榜首工资 - 他的工资)
                # 老代码里这条上界是拿档位上限做的（cap = tier_high），
                # 档位去掉之后差点跟着丢了。它很重要：没有上界，
                # 一个闷头不动的人也会被每轮加一份先验，估计越飘越高。
                # 没人被点名 = 没人贪了钱还没被查（只有工资可比时不广播），上界直接是 0
                cap = (
                    top_dirty + float(max(top_salary, my_salary) - my_salary)
                    if top_ids
                    else 0.0
                )
                if gained > cap:
                    evidence *= cap / gained if gained else 0.0
                    gained = max(0.0, cap)

            model.corrupt_evidence += evidence
            model.money_est += gained + self.cfg.salary(rank_before)  # 工资是公开可算的
            # 查实的证据现在看"记没记警告"，不能再看降级：
            # 攒够两次才降一级，只数降级会把举报压力低估一半。
            if self._caught(fact):
                if pid != self.id:  # 分母只数对手，分子也只数对手
                    self.observed_demotions += 1

            # --- 被举报没收：只没收本轮那一笔（按比例），存款不再被抄 ---
            if self._caught(fact):
                model.money_est = max(
                    0.0, model.money_est - gained * self._exposed_share(cur.get("origin"))
                )

            # --- 晋升对钱的影响 ---
            if fact["promotion"] == "MONEY":
                # 买得起说明他手里至少有门槛那么多；花掉门槛之后余额**全留**
                # （money_overflow_divisor 默认 1）。以前误用了老规则书的 overflow_divisor=5，
                # 每次有人花钱升职，AI 就把他的存款估成五分之一——复盘 5A9J：
                # 真人两次买官后实有 31，AI 估 2，到他凑齐钱登顶那轮谁也没拦。
                cost = self._costs(rank_before, cur.get("origin"))[0] or 0
                model.money_est = max(model.money_est, float(cost))
                model.money_est = math.ceil(
                    (model.money_est - cost) / self.cfg.money_overflow_divisor
                )
                model.money_est = max(0.0, model.money_est)

        self._prev_players = players

    def _update_table(self, facts, players, result) -> None:
        """这桌最近的风气：举报多凶、攻击多凶、有多少人在贪、多少人在干活、我自己挨了多少。"""
        if not facts:
            return
        n = len(facts)
        caught = sum(1 for f in facts.values() if self._caught(f))
        working = 0
        for pid, f in facts.items():
            cur, prev = players.get(pid), self._prev_players.get(pid)
            if cur and prev and cur["merit"] - prev["merit"] + f["attack_merit_loss"] > 0:
                working += 1
        mine = facts.get(self.id, {})
        obs = {
            "rep": sum(1 for f in facts.values() if f.get("reported_by_player")) / n,
            "atk": sum(1 for f in facts.values() if f.get("attacked")) / n,
            "corrupt": min(1.0, 0.5 * bool(result.get("wealth_top_ids")) + caught / n),
            "work": working / n,
            "me_atk": float(bool(mine.get("attacked"))),
            "me_rep": float(bool(mine.get("reported_by_player"))),
        }
        for k, x in obs.items():
            self.table[k] = TABLE_DECAY * self.table[k] + (1 - TABLE_DECAY) * x

    @staticmethod
    def _caught(fact) -> bool:
        """本轮被举报/事件查实了没有（公开事实：记了警告或者降了级）。"""
        return fact.get("warnings_issued", 0) > 0 or fact["demotion"] != "NONE"

    def _hand_is_weak(self, private: dict[str, Any]) -> bool:
        """这手牌烂不烂——只在换牌**免费**时用（富二代），所以判据可以宽一点。

        真人会换的两种：
          * 够门槛了，手里却没有用得上的晋升卡 —— 白白耽误一轮
          * 一张生产牌都没有、也没有能用的晋升卡 —— 这一轮什么都推进不了
        不碰 rng：复盘工具重演时不会调这里，碰了 rng 后面的决策就全错位。
        """
        cfg = self.cfg
        cards = [Card(d["card"]) for d in private["hand"]]
        rank, money, merit = private["rank"], private["money"], private["merit"]
        mc, tc = self._costs(rank, private.get("origin"))
        merit_ok = tc is not None and merit >= tc
        money_ok = mc is not None and money >= mc
        if cfg.needs_both(rank):
            usable = merit_ok and money_ok and any(c.is_promotion for c in cards)
            ready = merit_ok and money_ok
        else:
            usable = any(
                (c.can_use_merit and merit_ok) or (c.can_use_money and money_ok)
                for c in cards
            )
            ready = merit_ok or money_ok
        if ready and not usable:
            return True
        return not usable and not any(c.is_production for c in cards)

    def _dirty_prior(self, fact, cur, prev) -> tuple[float, float]:
        """光看公开信息，他这轮捞钱的概率和金额先验 -> (证据量, 估计金额)。

        「政绩涨了」**不等于**「没捞钱」——每轮能打 2 张牌，完全可以工作 + 贪污；
        而以权谋私更是一张牌同时给政绩和钱，打出来看上去和埋头工作一模一样。
        以前这里对"政绩涨了"的人直接记 0，于是闷声发财的领先者被系统性低估：
        复盘 F7KF 第 7 轮，老张实际有 45 块（门槛 37），AI 只估到 26，
        「他下一步就夺冠」那道刹车因此一次都没踩。
        """
        rank_before = fact["rank_before"]
        if fact["promotion"] == "MERIT":
            worked = True  # 政绩晋升会把池子清空，涨幅看不出来，但必然在积累政绩
        else:
            worked = (cur["merit"] - prev["merit"] + fact["attack_merit_loss"]) > 0
        if worked:
            p_dirty = 0.45 if int(self.cfg.picks_per_round) > 1 else 0.0
            return p_dirty, p_dirty * self._expected_graft_money(rank_before)
        # 他没打 WORK，可能在贪、在举报、在攻击
        return 0.45, 0.45 * self._expected_corrupt(rank_before)

    # ------------------------------------------------------------------
    # 小工具
    # ------------------------------------------------------------------

    def _mult(self, rank: int) -> float:
        return float(self.cfg.rank_multiplier(rank))

    def _expected_work(self, rank: int) -> float:
        return self._mean(self.cfg.work_card_distribution) * float(
            self.cfg.work_multiplier(rank)
        )

    def _expected_corrupt(self, rank: int) -> float:
        return self._mean(self.cfg.corrupt_card_distribution) * float(
            self.cfg.money_multiplier(rank)
        )

    def _expected_graft_money(self, rank: int) -> float:
        return self._mean(self.cfg.graft_card_distribution) * float(
            self.cfg.money_multiplier(rank)
        )

    def _expected_graft_merit(self, rank: int) -> float:
        return (
            self._mean(self.cfg.graft_card_distribution)
            * float(self.cfg.graft_merit_ratio)
            * float(self.cfg.work_multiplier(rank))
        )

    @staticmethod
    def _mean(dist) -> float:
        return sum(v * w for v, w in dist) / sum(w for _, w in dist)

    def _exposed_share(self, origin: str | None) -> float:
        """我捞的钱里，被举报时真会被抄走的比例。

        小镇做题家·会计「做账」能把一半做成合法收入，所以他敢贪得多一些。
        （这只影响"我怕不怕"，不影响"别人抓不抓得到我"——做账挡的是钱不是罪。）
        """
        share = float(self.cfg.report_seize_ratio)
        if (self.cfg.origin(origin) or {}).get("id") == "ACCOUNTANT":
            share = min(share, 1.0 - float(self.cfg.origin_accountant_launder_ratio))
        # 举报人那份一定照抄（见 rules 第 5 步）
        return max(float(self.cfg.report_reward_ratio), share)

    def _own_work_merit(
        self, value: int, rank: int, origin: str | None, event=None
    ) -> float:
        """我打这张 WORK 能拿多少政绩。出身不影响单张（卷王的加班看的是"连干两张"）。
        `event` 只有官二代「透风」知道本轮事件时才传。"""
        if value > 0:
            return float(rules.work_merit(value, rank, event, self.cfg))
        if event is None:
            return self._expected_work(rank)
        dist = self.cfg.work_card_distribution
        return sum(w * rules.work_merit(v, rank, event, self.cfg) for v, w in dist) / sum(
            w for _, w in dist
        )

    def _overtime_merit(self, values: list[int], rank: int, event=None) -> float:
        """卷王「加班」：两张埋头工作的政绩最后 ×倍数，多出来的 = (倍数 − 1) × 两张的政绩。
        不扣可能被抢走的那份——出牌时不知道会不会挨打。"""
        two = sum(self._own_work_merit(v, rank, None, event) for v in values[:2])
        return (self.cfg.origin_grinder_overtime_multiplier - 1) * two

    def _costs(self, rank: int, origin: str | None) -> tuple[int | None, int | None]:
        """这个(官职, 出身)升下一级要多少钱和政绩。

        必须走 rules 里那对函数，不能直接读 cfg：官二代的政绩门槛打了八折，
        AI 照原价算就会低估他的进度，"他下一步就夺冠"那道刹车会失灵——
        这正是上一轮花了很大力气才修好的一类 bug。
        """
        return (
            rules.money_cost_at(rank, origin, self.cfg),
            rules.merit_cost_at(rank, origin, self.cfg),
        )

    def _progress(
        self, rank: int, money: float, merit: float, origin: str | None = None
    ) -> float:
        """离下一级还剩多少，0~1。

        双条件台阶上要两样都够，所以看的是**短板**那一项，不是强项。
        """
        cfg = self.cfg
        mc, tc = self._costs(rank, origin)
        if mc is None or tc is None:
            return 1.0
        if cfg.needs_both(rank):
            return min(min(money / mc, 1.0), min(merit / tc, 1.0))
        return max(min(money / mc, 1.0), min(merit / tc, 1.0))

    def _own_progress(self, public, private, rank: int, money: float, merit: float) -> float:
        """我自己离下一级还剩多少。和 _progress 的区别：钱那条路要打风险折扣。

        钱够了不等于能升——买官会被举报查实（官作废、钱打水漂）。不打折的话，
        进度取的是 max(钱, 政绩)，钱一过线进度就是 1，**埋头工作被估成 0 分**；
        而 AI 又嫌买官太危险不肯买，结果两头落空。
        """
        cfg = self.cfg
        origin = private.get("origin")
        mc, tc = self._costs(rank, origin)
        if mc is None or tc is None:
            return 1.0
        if cfg.needs_both(rank) or not cfg.report_catches_bribery:
            return self._progress(rank, money, merit, origin)
        money_part = min(money / mc, 1.0) * (1.0 - self._report_pressure(public, private))
        merit_part = min(merit / tc, 1.0)
        # 两条路"有一条走得通就行"：不能取 max——钱那条打了折还是比政绩大的时候，
        # 干活又会被估成 0 分
        return 1.0 - (1.0 - money_part) * (1.0 - merit_part)

    def _threat(self, rank: int, progress: float) -> float:
        """这个对手有多值得我花一个回合去按住 —— 看他离夺冠还有多远。

        用"还差几级"而不是"现在几级"，并且是加速的：差 3 级的人基本不值得理，
        差半级的人必须按住。
        """
        top = self.cfg.president_rank
        # 本级进度只算一半：威胁主要看官职。不然一个快升市级的县级，
        # 会和刚到省级的人差不多吓人——真人看的是"谁官最大"。
        steps_left = (top - rank) - progress * self.w.threat_progress_weight
        if steps_left <= 0:
            return 1.0
        closeness = max(0.0, 1.0 - steps_left / top)
        return self.w.threat_floor + (1.0 - self.w.threat_floor) * (
            closeness ** self.w.threat_exponent
        )

    def _about_to_win(self, opp: dict[str, Any], model: "OpponentModel") -> bool:
        """他是不是**这一轮**就可能直接当上主席。

        政绩是公开的，按实数比。**钱是暗的**，`money_est` 只能算个下界——
        贪污看不见，以权谋私看起来又像在老实干活。所以钱这一项留出
        `endgame_money_doubt` 的余量：宁可多拦一次，也不要在他登顶那轮
        才发现自己估少了。少拦一次的代价是整局输掉，多拦一次只亏一个回合。

        还要往前看一轮：一轮能打两张牌，「生产牌 + 晋升卡」当轮就能把差的
        那点补上再升。以前只看"现在够不够"，复盘 QN78 第 9 轮：真人省级、
        政绩 30（门槛 43），手里一张埋头工作就是 +15，当轮就能登顶；
        AI 判"还差 13，不危险"，两个人手里都有攻击牌，谁也没动。
        人类玩家一眼就看得出"他再干一轮就够了"。
        所以：差的那一项只要一张生产牌（期望值）补得上，就算。
        两项都差就补不过来——生产牌只有一个位子，另一张得留给晋升卡。
        """
        cfg = self.cfg
        rank = opp["rank"]
        if rank != cfg.president_rank - 1:
            return False
        origin = opp.get("origin")
        mc, tc = self._costs(rank, origin)
        money_bar = mc * self.w.endgame_money_doubt if mc is not None else None
        merit_ready = tc is not None and opp["merit"] >= tc
        money_ready = money_bar is not None and model.money_est >= money_bar
        lookahead = int(cfg.picks_per_round) > 1
        merit_reach = merit_ready or (
            lookahead and tc is not None
            and opp["merit"] + self._own_work_merit(0, rank, origin) >= tc
        )
        money_reach = money_ready or (
            lookahead and money_bar is not None
            and model.money_est + self._expected_corrupt(rank) >= money_bar
        )
        if cfg.needs_both(rank):
            return (merit_ready and money_reach) or (money_ready and merit_reach)
        return merit_reach or money_reach

    def _target_focus(self, public: dict[str, Any], opp: dict[str, Any]) -> float:
        """这个对手值不值得我花一张干扰牌：他的威胁 ÷ 桌上最大的威胁。

        威胁主要看官职（_threat 用的是"离主席还差几级"），所以一个政绩一万的基层
        也排不到前面。以前"抢功 / 分赃"那份自己的好处不看对象是谁，AI 就专挑
        干活最多的人（卷王几乎每轮两张工作）去抢，放着省级的领跑者不管——
        真人玩的时候也明显感觉到：别人都当上省级第一了，AI 还在打我。
        """
        def threat(o: dict[str, Any]) -> float:
            m = self.models.get(o["id"], OpponentModel())
            return self._threat(
                o["rank"], self._progress(o["rank"], m.money_est, o["merit"], o.get("origin"))
            )

        others = [o for o in public["players"] if o["id"] != self.id]
        top = max((threat(o) for o in others), default=0.0)
        if top <= 0:
            return 1.0
        floor = self.w.target_focus_floor
        return floor + (1.0 - floor) * threat(opp) / top

    def _endgame_stop_value(self, opp: dict[str, Any], model: "OpponentModel") -> float:
        """把一个这一轮就可能登顶的人**确定**拦下来值多少。

        量纲和攻击分一致：拦下一级（1 + promotion_bonus）× 威胁 × 终局权重。
        单张牌、两张牌各能拦下几成，乘 _endgame_cover 给的把握。
        """
        progress = self._progress(
            opp["rank"], model.money_est, opp["merit"], opp.get("origin")
        )
        return (
            self.w.endgame_scale
            * (1.0 + self.w.promotion_bonus)
            * self._threat(opp["rank"], progress)
            * self.w.endgame_block_weight
            * self._p_reach(opp, model)
            * self._contender(opp["id"])
        )

    def _contender(self, leader_id: int) -> float:
        """把他拦下来之后，我自己有多大机会接着赢（相对剩下的人里最强的那个，0~1）。

        拦住要登顶的人，好处不是全桌平分的：他赢了每个人都是 0，
        可拦下来之后能反超的只有紧跟着的第二、第三名——对他们来说这一刀值一整局。
        远远落后的人就算拦下来了自己也赢不了，真人会想"第二第三会去拦，我先发展"。
        以前每个 AI 不管自己排第几都一样拼命拦，结果落后的人把回合全花在拦人上
        （对决里把拦人力度整体调低，单个 AI 反而赢得更多）。
        """
        public, private = self._seen_public, self._seen_private
        if not public or not private:
            return 1.0

        top = self.cfg.president_rank

        def steps(o: dict[str, Any]) -> float:
            """离主席还差几级（本级进度算进去）"""
            if o["id"] == self.id:
                prog = self._progress(private["rank"], private["money"], private["merit"],
                                      private.get("origin"))
                return (top - private["rank"]) - prog
            m = self.models.get(o["id"], OpponentModel())
            return (top - o["rank"]) - self._progress(o["rank"], m.money_est, o["merit"],
                                                      o.get("origin"))

        # 不能用威胁值比：到了后期好几个人都在省级，威胁值全挤在 1 附近，分不出谁是第二谁是第五
        rest = [o for o in public["players"] if o["id"] != leader_id]
        if not rest:
            return 1.0
        mine = next((steps(o) for o in rest if o["id"] == self.id), float(top))
        best = min(steps(o) for o in rest)
        # 落后半级 ≈ 0.47，落后一级 ≈ 0.22，落后两级基本不拦
        return math.exp(-self.w.contender_decay * max(0.0, mine - best))

    def _p_reach(self, opp: dict[str, Any], model: "OpponentModel") -> float:
        """他这一轮真凑得齐登顶条件的把握（0.2~1）。

        _about_to_win 只给"是 / 否"，几个人同时被判要登顶时，AI 会在他们中间随便挑一个打。
        体检：有人当轮登顶、旁观者手里有干扰牌却没打他的情况里，八成是打了另一个"要登顶"的人。
        差的那一项越要靠一张好牌才补得上，把握越低；两样都已经够了的人最该先按住。
        """
        rank = opp["rank"]
        mc, tc = self._costs(rank, opp.get("origin"))
        if mc is None or tc is None:
            return 0.0

        def reach(gap: float, per_card: float) -> float:
            if gap <= 0:
                return 1.0
            if per_card <= 0:
                return 0.2
            return max(0.2, min(1.0, 1.5 - gap / per_card))

        p_merit = reach(tc - opp["merit"], self._expected_work(rank))
        p_money = reach(mc * self.w.endgame_money_doubt - model.money_est,
                        self._expected_corrupt(rank))
        # 生产牌只有一个位子：两样都差的话只能补一样
        return min(p_merit, p_money) if p_merit < 1 and p_money < 1 else p_merit * p_money

    def _endgame_cover(
        self, opp: dict[str, Any], model: "OpponentModel"
    ) -> tuple[float, float, float]:
        """(只攻击, 只举报, 攻击 + 举报都压上) 各有几成把握拦下他这一轮登顶。

        主席那一级走哪条路由卡决定：
          * 政绩升职      -> 只怕攻击
          * 贿赂升职      -> 只怕举报
          * 通用升职      -> 被攻击就改走金钱，还得再挨一张举报
          * 两张一起打    -> 两张都得挨
        他手里是哪张看不见，单张牌只能赌中一部分；两张都压上才稳。
        这正是人类玩家在门口会做的事：手里攻击、举报都有，就一起砸过去。

        例外：
          * 他的钱还不够、这一轮得现贪才凑得齐 —— 一贪就能被举报查实，
            排在贪污后面的晋升卡全部冻结，举报单张就几乎稳拦
          * 贫农「政治正确」攻击挡不住他，政绩那条路谁也按不住
        """
        mc, _ = self._costs(opp["rank"], opp.get("origin"))
        single = self.w.endgame_cover_single
        report = single
        if mc is not None and model.money_est < mc * self.w.endgame_money_doubt:
            report = max(report, self.w.endgame_cover_report_must_corrupt)
        both = max(self.w.endgame_cover_both, single, report)
        if (self.cfg.origin(opp.get("origin")) or {}).get("id") == "PEASANT":
            # 贫农只挡得住一个人：有别人也打他（概率 k）时攻击照常管用，否则白打；
            # 开关设成"几个人都拦不住"时攻击对他的政绩升职完全没用
            k = self._peasant_block_chance()
            return k * single, report, k * both + (1 - k) * report
        return single, report, both

    def _dogpile_target(self, public: dict[str, Any]) -> dict[str, Any] | None:
        """围堵默契的靶子：在省级、政绩差一张埋头工作就够门槛（或已判要登顶）的人里进度最快的。"""
        if not self.cfg.ai_dogpile:
            return None
        top = self.cfg.president_rank - 1
        best, best_p = None, -1.0
        for o in public["players"]:
            if o["id"] == self.id or o["rank"] != top:
                continue
            _, tc = self._costs(o["rank"], o.get("origin"))
            if tc is None:
                continue
            close = (o["merit"] + self._expected_work(o["rank"]) >= tc
                     or self._about_to_win(o, self.models.get(o["id"], OpponentModel())))
            if close and o["merit"] / tc > best_p:
                best, best_p = o, o["merit"] / tc
        return best

    def _apply_dogpile(self, public, hand: list[Card], scored, feats):
        """围堵默契：手里有攻击就必须打（他政绩够了的话举报也要打），除非我这轮能当主席。"""
        target = self._dogpile_target(public)
        if target is None or getattr(self, "_can_win_now", False):
            return scored, feats
        need = []
        if Card.ATTACK in hand and self.allow_attack:
            need.append(Card.ATTACK)
        _, tc = self._costs(target["rank"], target.get("origin"))
        if Card.REPORT in hand and self.allow_report and tc is not None and target["merit"] >= tc:
            need.append(Card.REPORT)
        if not need:
            return scored, feats
        keep = [i for i, (_, cards) in enumerate(scored) if all(c in cards for c in need)] \
            or [i for i, (_, cards) in enumerate(scored) if any(c in cards for c in need)]
        if not keep:
            return scored, feats
        return [scored[i] for i in keep], ([feats[i] for i in keep] if feats else feats)

    def _peasant_block_chance(self) -> float:
        """我去攻击贫农，能按住他政绩升职的把握。"""
        n = self.cfg.origin_peasant_max_attackers
        if n is None:
            return 0.0
        return self.w.peasant_second_attacker if n <= 1 else self.w.peasant_second_attacker ** n

    def _cash_horizon(self, public: dict[str, Any]) -> float:
        """抄到手的一笔钱，现在还值几折。

        钱本身不算分，它得先换成官职才算数。所以两件事会让它贬值：
          * **有人下一步就夺冠** —— 他一登顶游戏立刻结束，我口袋里的钱
            一分也来不及花。这时候该比的是"能不能拦住他"，不是"能抄多少"。
          * **回合快打完了** —— 剩的轮数不够我再升一级，同理。

        不加这一折，举报的目标就会被"谁最有钱"带着走：复盘 F7KF，
        老张刚砸钱升到省级、身上估着 0 块，于是"举报老张"看起来一文不值，
        AI 转头去抄手里有四十块的真人——哪怕老张的威胁值是他的两倍。
        """
        rounds_left = max(0, int(public.get("max_rounds", 0)) - int(public.get("round", 0)))
        for o in public["players"]:
            if o["id"] == self.id:
                continue
            if self._about_to_win(o, self.models.get(o["id"], OpponentModel())):
                return 0.15  # 他赢了就散场，攒钱毫无意义
        if rounds_left <= 0:
            return 0.15
        if rounds_left == 1:
            return 0.5  # 只够用在这一轮，兑不成官职就白拿
        return 1.0

    def _diffusion(self, public: dict[str, Any]) -> float:
        """纯打压行为的收益稀释系数。

        我一个人付出整个回合的代价去按住某个对手，但省下来的"他不会赢"这份好处，
        是和其余所有人平分的——人越多，单纯搞破坏越不划算。
        拿到手的赃款不适用这个折扣，那是实打实进我口袋的。
        """
        n_opp = max(1, len(public["players"]) - 1)
        return 1.0 / n_opp

    def _share(self, public: dict[str, Any], threat: float) -> float:
        """按住某个人这件事，好处里有多少是**我自己**的。

        "搭便车"那套算法只在对手赢不了的时候成立：拖慢一个离夺冠还远的人，
        省下来的便宜确实全桌平分，我凭什么单独出力。但对手越接近登顶，
        逻辑就越反过来——他赢了我也输，拦住他是在救我自己的命，
        这份好处一分都不该按人头摊薄。

        所以用威胁值在「全桌平分」和「全归我」之间插值。平方是为了让它
        只在真正的领跑者身上生效，不要把中游选手也算成生死大敌。
        """
        base = self._diffusion(public)
        return base + (1.0 - base) * threat ** 2

    def _report_pressure(self, public: dict[str, Any], private: dict[str, Any]) -> float:
        """我这轮要是贪了，被举报查实的概率大概多少。

        先验是 base_report_pressure，然后用**实际观测到的**"贪污次数 vs 降级次数"
        把它修正过来——这桌人要是根本不举报，就该放心去贪。
        """
        n_opp = len(public["players"]) - 1
        if n_opp <= 0:
            return 0.0
        prior_strength = 6.0
        empirical = (
            prior_strength * self.w.base_report_pressure + self.observed_demotions
        ) / (prior_strength + self.est_corrupt_attempts)
        p = min(0.95, max(0.02, empirical))

        ranks = [(p_["rank"], p_["merit"]) for p_ in public["players"]]
        mine = (private["rank"], private["merit"])
        if ranks and mine >= max(ranks):
            p = min(0.95, p * self.w.leader_suspicion)  # 明面上的领先者最招人举报
        return min(0.95, p * self._self_suspicion(public))

    def _self_suspicion(self, public: dict[str, Any]) -> float:
        """别人看我有多可疑，相对全桌平均（0.6~2）。

        对手的读法（_report_hit_prob）里有出身、个人档案这些因子——会计贪得多，
        大家就盯着会计举报。以前 AI 只会这样读别人，不知道别人也这样读它：
        会计 AI 照着"全桌平均风险"放心去贪，六身份混战里胜率掉到 9.3%。
        现在拿同一套读法算一遍"别人眼里的我"，比全桌平均可疑多少，被查实的风险就高多少。
        """
        me = next((o for o in public["players"] if o["id"] == self.id), None)
        others = [o for o in public["players"] if o["id"] != self.id]
        if me is None or not others:
            return 1.0
        mine = self._read(public, me, self.self_model)[0]
        avg = sum(self._read(public, o, self.models.get(o["id"], OpponentModel()))[0]
                  for o in others) / len(others)
        if avg <= 0:
            return 1.0
        return max(0.6, min(2.0, mine / avg)) ** self.w.self_suspicion

    # ------------------------------------------------------------------
    # 决策
    # ------------------------------------------------------------------

    def decide(
        self, public: dict[str, Any], private: dict[str, Any]
    ) -> list[tuple[str, int | None]]:
        """返回本轮要打的 N 张牌 [(卡名, 目标id), ...]。

        不能逐张挑最高分就完事——「WORK + 政绩升职」是个组合技，
        单独看每张牌会漏掉协同。所以先给干扰牌定好目标，再**成对**评估经济牌。
        打分在 score_combos 里；这里只加噪声挑最高、排顺序、定目标。
        """
        self.observe(public, private)

        hand = [
            Card(d["card"]) if isinstance(d, dict) else Card(d) for d in private["hand"]
        ]
        n_picks = min(int(public.get("picks_per_round", 1)), len(hand))
        if not hand or n_picks <= 0:
            self.last_scores = []
            return []

        scored = self.score_combos(public, private)
        # 留一份不含噪声的打分给复盘工具看（replay.py）。
        # 复盘不能自己再调一次 score_combos：打平挑目标会消耗 rng，
        # 多调一次后面的噪声就全错位了，同 seed 也复现不出来。
        self.last_scores = scored
        scored, _ = self._apply_dogpile(public, hand, scored, [])
        # 噪声只用来决定"打不打干扰、打几张什么干扰"；同一种干扰搭配里，
        # 经济牌永远挑不加噪声时最好的那组。以前每个组合各加各的噪声：
        # 有人要登顶时经济分全打 0.15 折，"埋头工作"和"一张根本升不了的晋升卡"
        # 只差 0.005，比噪声（±0.02）还小，AI 就会把晋升卡当废牌打出去
        # ——体检 800 局里，没生产牌、资源本来就不够的晋升卡白打了 1190 次。
        best_in_kind: dict[tuple[int, int], tuple[float, list[Card]]] = {}
        for score, cards in scored:
            kind = (cards.count(Card.ATTACK), cards.count(Card.REPORT))
            if kind not in best_in_kind or score > best_in_kind[kind][0]:
                best_in_kind[kind] = (score, cards)
        best_score, best_combo = None, None
        for score, cards in best_in_kind.values():
            score += self.rng.uniform(-self.w.noise, self.w.noise)
            if best_score is None or score > best_score:
                best_score, best_combo = score, cards

        return self._finish(public, private, list(best_combo or []))

    def _finish(self, public, private, chosen: list[Card]) -> list[tuple[str, int | None]]:
        """选好了出哪组牌之后：排出场顺序、给干扰牌挑目标、留下复盘/体检要的记录。"""
        chosen = self._order_picks(private, chosen)
        opponents = [p for p in public["players"] if p["id"] != self.id]
        picks: list[tuple[str, int | None]] = []
        used_targets: set[int] = set()
        for card in chosen:
            target = None
            if card.needs_target:
                target = self._pick_target(public, private, card, opponents, used_targets)
                if target is None:
                    continue
                used_targets.add(target)
            picks.append((card.value, target))
        self.last_prediction = self._predictions(public, private, chosen, picks)
        reads = {o["id"]: self._report_hit_prob(public, o) for o in opponents}
        self._pending_corrupt_expect = {pid: r[0] for pid, r in reads.items()}
        self._pending_hit_expect = {pid: r[2] for pid, r in reads.items()}
        self.decision_log.append(
            f"r{public.get('round')} -> " + ", ".join(
                f"{c}{'' if t is None else f'@{t}'}" for c, t in picks
            )
        )
        return picks

    def _predictions(self, public, private, chosen, picks) -> dict[str, Any]:
        """这一手出牌时 AI 心里的预测，留给 audit.py 和真实结果对账。只读，不碰 rng。"""
        players = {o["id"]: o for o in public["players"]}
        dirty = any(c in (Card.CORRUPT, Card.GRAFT) for c in chosen)
        p_caught = None
        if dirty:
            p_caught = self._report_pressure(public, private)
            tip = private.get("tipoff_event")
            event = rules.event_by_id(tip["id"], self.cfg) if tip else None
            if event is not None and event.flag("storm_report"):
                p_caught = max(p_caught, self.w.tipoff_storm_caught)
        reports, closing = {}, {}
        for card, target in picks:
            if target is None or target not in players:
                continue
            opp = players[target]
            closing[target] = self._about_to_win(opp, self.models.get(target, OpponentModel()))
            if card == Card.REPORT.value:
                reports[target] = self._report_hit_prob(public, opp)
        return {"p_caught": p_caught, "reports": reports, "closing": closing}

    def score_combos(
        self, public: dict[str, Any], private: dict[str, Any]
    ) -> list[tuple[float, list[Card]]]:
        """给手里每一组可出的牌打分（不含噪声），按枚举顺序返回 [(分数, 牌), ...]。

        decide 和复盘工具共用这一份，打分永远同源。
        注意它会消耗 rng（干扰牌打平时随机挑目标），所以别在 decide 之外重复调用
        ——要看打分就读 decide 留下的 `last_scores`。
        """
        dealt = [
            (Card(d["card"]) if isinstance(d, dict) else Card(d),
             int(d.get("value", 0)) if isinstance(d, dict) else 0)
            for d in private["hand"]
        ]
        # 红二代「一纸调令」：不算行动卡、不占出牌位，用的话永远最先结算（每局一次）
        family_ok = bool((private.get("family_card") or {}).get("usable"))
        hand = [c for c, _ in dealt]
        values = [v for _, v in dealt]
        n_picks = min(int(public.get("picks_per_round", 1)), len(hand))
        if not hand or n_picks <= 0:
            return []

        opponents = [p for p in public["players"] if p["id"] != self.id]
        # 有人这一轮就可能登顶？那我自己的经济牌只有在"我也当轮登顶"时才算数
        killers = [
            o for o in opponents
            if self._about_to_win(o, self.models.get(o["id"], OpponentModel()))
        ]
        rival_closing = bool(killers)
        # 有人要登顶时我自己的经济牌打几折。头号挑战者照旧几乎不看（他得去拦），
        # 越落后越照常发展——"第二第三会去拦，游戏多半结束不了，我先发展"。
        # 不这么分的话，落后的人发展打 0.15 折、拦人又被 _contender 压得很低，
        # 两害相权还是去拦：体检里全场第 4 名以后的人拦人比例 96%，比第二名（84%）还高。
        closing_discount = 1.0
        if killers:
            c = max(self._contender(o["id"]) for o in killers)
            d = self.w.endgame_economy_discount
            closing_discount = d + (1.0 - d) * (1.0 - c)
        # 攻击 + 举报一起压在他身上，比两张各算各的值钱：单张只能赌中一条路，
        # 两张一起才把两条路都堵上。补上这份差额，手里两张都有的时候就会一起打。
        both_bonus = max(
            (
                self._endgame_stop_value(o, m)
                * (lambda a, r, b: b - a - r)(*self._endgame_cover(o, m))
                for o in killers
                for m in [self.models.get(o["id"], OpponentModel())]
            ),
            default=0.0,
        )

        # 1) 干扰牌：每个目标各打一遍分，从高到低排好。
        #    同一种牌打两张时，第二张会换一个人打（_pick_target 避开打过的人），
        #    所以只能算第二好的那个目标——以前两张都按最好的目标算，
        #    "举报 + 举报"被高估一倍，AI 会为了第二张举报放弃干活。
        solo: dict[Card, list[float]] = {}
        solo_targets: dict[Card, list[dict[str, Any]]] = {}  # 同样排好序的目标（特征要看打的是谁）
        # 固定顺序（不用 set）：set 的遍历顺序随进程的哈希种子变，顺序一变结果就对不上
        for card in dict.fromkeys(hand):
            scorer = (self._score_attack if card is Card.ATTACK and self.allow_attack
                      else self._score_report if card is Card.REPORT and self.allow_report
                      else None)
            if scorer is not None and opponents:
                ranked = sorted(((scorer(public, private, o)[0], i) for i, o in enumerate(opponents)),
                                reverse=True)
                solo[card] = [s for s, _ in ranked]
                solo_targets[card] = [opponents[i] for _, i in ranked]
            else:
                solo[card] = [0.0]

        # 2) 枚举所有出牌组合（按手牌下标，所以同名牌能出两张）
        scored: list[tuple[float, list[Card]]] = []
        feats: list[dict[str, float]] = []
        self._can_win_now = False
        contender = max((self._contender(o["id"]) for o in killers), default=0.0)
        closer = max(killers, key=lambda o: self._p_reach(
            o, self.models.get(o["id"], OpponentModel())), default=None)
        closer_oid = (self.cfg.origin(closer.get("origin")) or {}).get("id") if closer else None
        variants = [[]] + ([[Card.PROMOTE_FAMILY]] if family_ok else [])
        for combo, extra in (
            (combo, extra)
            for combo in combinations(range(len(hand)), n_picks)
            for extra in variants
        ):
            cards = extra + [hand[i] for i in combo]
            # 被禁用的牌（消融 / 对抗赛用）整组不考虑——只把分数记 0 不够，
            # 学习型 AI 有自己的权重，分数是 0 的组合照样可能被抽中
            if ((not self.allow_corrupt and Card.CORRUPT in cards)
                    or (not self.allow_attack and Card.ATTACK in cards)
                    or (not self.allow_report and Card.REPORT in cards)):
                continue
            score, i_win = self._score_economy(
                public, private, cards, [0] * len(extra) + [values[i] for i in combo]
            )
            econ = score
            self._can_win_now = self._can_win_now or i_win
            if rival_closing and not i_win:
                score *= closing_discount
            econ_scored = score
            seen: Counter = Counter()
            for c in cards:
                if c in (Card.ATTACK, Card.REPORT):
                    # 有人要登顶时干扰牌全都压在他身上（见 _pick_target），
                    # 同一种牌打两张不会多拦下什么：第二刀不算分
                    if rival_closing and seen[c]:
                        continue
                    ranked = solo[c]
                    score += ranked[seen[c]] if seen[c] < len(ranked) else 0.0
                    seen[c] += 1
            both = rival_closing and seen[Card.ATTACK] and seen[Card.REPORT]
            if both:
                score += both_bonus
            scored.append((score, cards))
            if self.want_features:
                f = self._combo_features(
                    cards, score, econ, score - econ_scored, both, rival_closing, contender, public
                )
                # 打的是什么出身的人：贫农单刀攻击挡不住、红二代举报了也降不了级、
                # 会计被举报一半赃款抄不走……这些让它自己学
                for c, tag in ((Card.ATTACK, "atk_on"), (Card.REPORT, "rep_on")):
                    for t in solo_targets.get(c, [])[:cards.count(c)]:
                        toid = (self.cfg.origin(t.get("origin")) or {}).get("id")
                        if toid:
                            f[f"{tag}_{toid}"] = f.get(f"{tag}_{toid}", 0.0) + 1.0
                if closer_oid:
                    f[f"intf_on_closer_{closer_oid}"] = f["n_attack"] + f["n_report"]
                feats.append(f)
        self.last_features = feats
        return scored

    def _combo_features(self, cards, total, econ, intf, both, closing, contender, public):
        """一个出牌组合的特征（给 LearnedAgent 学"怎么权衡"用）。

        状态本身的量（这桌最近举报多凶）对所有组合都一样，放进 softmax 会被约掉，
        所以都和组合的属性交叉：贪不贪 × 这桌举报强度、干活多少 × 我最近挨打的频率……
        靠这些交叉项，线性策略才学得会"没人举报就去贪，大家都在抢功就别闷头干活"。
        """
        d = self._econ_detail
        t = self.table
        n_atk = float(cards.count(Card.ATTACK))
        n_rep = float(cards.count(Card.REPORT))
        rnd = int(public.get("round", 1))
        late = rnd / max(1, int(public.get("max_rounds", 12)))
        merit_risk = self.cfg.attack_steal_fraction if self.cfg.attack_mode == "steal_work" else 0
        feats = {
            "hand": total,
            "econ": econ,
            "intf": intf,
            "promote": d["promote"],
            "promote_money": d["promote_money"],
            "promote_merit": d["promote_merit"],
            "dirty": d["dirty"],
            "merit_gain": d["merit_gain"],
            "money_gain": d["money_gain"],
            "risk": d["risk"],
            "n_attack": n_atk,
            "n_report": n_rep,
            "both": float(both),
            "family": float(Card.PROMOTE_FAMILY in cards),
            # ---- 看桌子：组合属性 × 这桌最近的风气 ----
            "dirty_x_rep": d["dirty"] * (t["rep"] - TABLE_PRIOR["rep"]),
            "dirty_x_merep": d["dirty"] * (t["me_rep"] - TABLE_PRIOR["me_rep"]),
            "bribe_x_merep": d["promote_money"] * (t["me_rep"] - TABLE_PRIOR["me_rep"]),
            "work_x_atk": d["merit_gain"] * (t["atk"] - TABLE_PRIOR["atk"]),
            "work_x_meatk": d["merit_gain"] * float(merit_risk) * t["me_atk"],
            "meritpromo_x_meatk": d["promote_merit"] * t["me_atk"],
            "report_x_corrupt": n_rep * (t["corrupt"] - TABLE_PRIOR["corrupt"]),
            "attack_x_work": n_atk * (t["work"] - TABLE_PRIOR["work"]),
            "intf_x_contender": (n_atk + n_rep) * contender,
            "intf_x_closing": (n_atk + n_rep) * float(closing),
            "econ_x_late": econ * late,
            "dirty_x_late": d["dirty"] * late,
        }
        # 每个出身学自己的打法：会计被抓了一半赃款抄不走、该更敢贪；贫农一个人打不动他、
        # 该放心干活；红二代的一纸调令……共用一套权重的话，学出来的是"平均身份"的打法，
        # 第一轮训完会计胜率只有 5%、贫农 9%。所以关键特征再按"我是什么出身"各复制一份。
        oid = (self.cfg.origin((self._seen_private or {}).get("origin")) or {}).get("id")
        if oid:
            for k in ORIGIN_CROSSED:
                feats[f"{oid}*{k}"] = feats[k]
        return feats

    def explain(self, public: dict[str, Any], private: dict[str, Any]) -> list[dict[str, Any]]:
        """我眼里的每个对手：给复盘工具看 AI 当时是怎么想的。只读，不碰 rng。"""
        rows = []
        for o in public["players"]:
            if o["id"] == self.id:
                continue
            model = self.models.get(o["id"], OpponentModel())
            origin = o.get("origin")
            mc, tc = self._costs(o["rank"], origin)
            progress = self._progress(o["rank"], model.money_est, o["merit"], origin)
            about = self._about_to_win(o, model)
            rows.append({
                "id": o["id"],
                "name": o.get("name", str(o["id"])),
                "rank": o["rank"],
                "merit": o["merit"],
                "merit_cost": tc,
                "money_est": round(model.money_est, 1),
                "money_cost": mc,
                "about_to_win": about,
                "threat": round(self._threat(o["rank"], progress), 3),
                "attack": round(self._score_attack(public, private, o)[0], 3),
                "report": round(self._score_report(public, private, o)[0], 3),
                "cover": (
                    [round(x, 2) for x in self._endgame_cover(o, model)] if about else None
                ),
            })
        return rows

    def _order_picks(self, private, cards: list[Card]) -> list[Card]:
        """给选好的牌排出场顺序——结算严格按这个顺序走，排错了要吃亏。

        已经够门槛的晋升卡排最前面：这样本轮产出吃的是新官职的倍率，
        也不会被晋升的 /5 砍掉（_score_economy 算分时就是这么假设的）。
        还不够门槛的排最后面：等这轮赚到的东西一起算，说不定就够了。
        """
        cfg = self.cfg
        early = _early_promotion_rank(
            cfg, private["rank"], private["money"], private["merit"], cards
        )

        def key(item: tuple[int, Card]) -> tuple[int, int]:
            i, c = item
            if c.is_promotion:
                # 够门槛就排最前面：既吃到新官职的倍率，也避开"这一轮的赃款
                # 当轮花不出去"那条规则（排在贪污后面的晋升卡要等举报结算）
                return (0 if early is not None else 2, i)
            return (1, i)

        return [c for _, c in sorted(enumerate(cards), key=key)]

    def _best_target(self, scored: list[tuple[float, int | None]]) -> tuple[float, int | None]:
        """从打过分的目标里挑一个。**打平时随机挑**，不能按编号取第一个。

        这条很重要：开局所有人状态一样，所有目标的分数完全相同，
        用 max() 会让全场 AI 一起选中编号最小的玩家——而房主永远是 1 号，
        结果就是真人第一轮被集体举报。
        """
        if not scored:
            return 0.0, None
        top = max(sc for sc, _ in scored)
        # 相对容差；top 接近 0 时退化成"只有精确相等才算打平"，
        # 正好覆盖开局所有人状态一样的情况
        tol = max(1e-9, self.w.tie_epsilon * abs(top))
        tied = [item for item in scored if top - item[0] <= tol]
        return self.rng.choice(tied)

    def _pick_target(self, public, private, card, opponents, used):
        """给一张需要目标的牌挑目标；一般避开本轮已经打过的人（重复打收益低）。

        例外：有人下一步就夺冠时，举报和攻击要**一起压在他身上**。
        分散开来谁也拦不住，而他赢了桌上每个人都输——这时候"别重复"是错的。
        """
        target = self._dogpile_target(public)
        if target is not None and not getattr(self, "_can_win_now", False):
            return target["id"]
        killer = [
            o for o in opponents
            if self._about_to_win(o, self.models.get(o["id"], OpponentModel()))
        ]
        if killer:
            pool = killer
        else:
            pool = [o for o in opponents if o["id"] not in used] or opponents
        scorer = self._score_attack if card is Card.ATTACK else self._score_report
        return self._best_target([scorer(public, private, o) for o in pool])[1]

    def _score_economy(
        self, public, private, cards: list[Card], values: list[int] | None = None
    ) -> tuple[float, bool]:
        """联合评估这组牌里的生产牌 + 晋升卡：先赚后升，一轮最多升一级。

        返回 (分数, 这组牌能不能让我当轮登顶)。后者给 decide 判断
        "对手要赢了，我这手经济牌还有没有意义"。
        """
        cfg = self.cfg
        rank, money, merit = private["rank"], private["money"], private["merit"]
        my_origin = private.get("origin")
        # 官二代「透风」：已经知道本轮事件，产出直接按事件算（经济大好 ×2、重点项目 +4…）
        tip = private.get("tipoff_event")
        event = rules.event_by_id(tip["id"], cfg) if tip else None
        mc, tc = self._costs(rank, my_origin)
        here = self._own_progress(public, private, rank, money, merit)
        family_needs_loot = False  # 一纸调令排在后面、要花这一轮的赃款才凑够钱

        # 点数在发牌时就摇好了，所以这里用**确定值**算，不再用期望值猜
        vals = values or [0] * len(cards)
        n_corrupt = sum(1 for c in cards if c is Card.CORRUPT)
        n_graft = sum(1 for c in cards if c is Card.GRAFT)
        gain_merit = gain_money = 0.0
        for c, v in zip(cards, vals):
            if c is Card.WORK:
                gain_merit += self._own_work_merit(v, rank, my_origin, event)
            elif c is Card.CORRUPT:
                gain_money += (
                    rules.corrupt_money(v, rank, event, cfg)
                    if v > 0
                    else self._expected_corrupt(rank)
                )
            elif c is Card.GRAFT:
                if v > 0:
                    gain_money += rules.corrupt_money(v, rank, event, cfg)
                    gain_merit += rules.graft_merit(v, rank, event, cfg)
                else:
                    gain_money += self._expected_graft_money(rank)
                    gain_merit += self._expected_graft_merit(rank)
        # 卷王「加班」第二段：两张埋头工作一起打，额外一笔政绩
        work_vals = [v for c, v in zip(cards, vals) if c is Card.WORK]
        grinder_combo = (
            (cfg.origin(my_origin) or {}).get("id") == "GRINDER" and len(work_vals) >= 2
        )
        if grinder_combo:
            gain_merit += self._overtime_merit(work_vals, rank, event)
            # 加班费：几倍工资的合法收入
            gain_money += cfg.origin_grinder_overtime_multiplier * cfg.salary(rank)
        merit_after = merit + gain_merit
        money_after = money + gain_money
        # 工资不用在这里加：它在回合开头就发了，private["money"] 里已经含着了。
        # 再加一次就是凭空多算一笔。

        # 结算顺序是动态的：本来就够门槛的话会先升官、再按新倍率干活。
        # 这里照着重算一遍，否则 AI 会低估"WORK + 晋升卡"这套组合技。
        has_merit_card = any(c.can_use_merit for c in cards)
        has_money_card = any(c.can_use_money for c in cards)
        any_promo_card = any(c.is_promotion for c in cards)
        if not cfg.promotion_requires_card:
            has_merit_card = has_money_card = any_promo_card = True

        early = _early_promotion_rank(cfg, rank, money, merit, cards, my_origin)
        if early is not None:
            new_rank, money_left, merit_left = early
            gain_merit = gain_money = 0.0
            for c, v in zip(cards, vals):
                if c is Card.WORK:
                    gain_merit += self._own_work_merit(v, new_rank, my_origin, event)
                elif c is Card.CORRUPT:
                    gain_money += (
                        rules.corrupt_money(v, new_rank, event, cfg)
                        if v > 0 else self._expected_corrupt(new_rank)
                    )
                elif c is Card.GRAFT:
                    if v > 0:
                        gain_money += rules.corrupt_money(v, new_rank, event, cfg)
                        gain_merit += rules.graft_merit(v, new_rank, event, cfg)
            if grinder_combo:
                gain_merit += self._overtime_merit(work_vals, new_rank, event)
                gain_money += cfg.origin_grinder_overtime_multiplier * cfg.salary(new_rank)
            merit_after = merit_left + gain_merit
            money_after = money_left + gain_money  # 工资已在回合开头到账，别重复计
            here_after = self._progress(new_rank, money_after, merit_after, my_origin)
            score = (1.0 - here) + self.w.promotion_bonus + here_after * 0.5
            promoted = True
            # 组合里有一纸调令、又能"先升官"：它排在最前面，升职就是它带来的
            via_family = Card.PROMOTE_FAMILY in cards
            # 走的是不是金钱那条路（行贿）：主席那一级由卡决定，其他台阶政绩够就走政绩
            bribed = (
                not has_merit_card if cfg.needs_both(rank) else money_left < money
            )
        else:
            # 没能"先升官"：下面按普通晋升卡算"干完活再升"时先不算一纸调令
            # （它不能花这一轮贪的钱，只能靠政绩，单独在后面判断；
            #   以前不分这么细，AI 资源不够也照打，94% 白用）
            via_family = False
            if Card.PROMOTE_FAMILY in cards:
                rest = [c for c in cards if c is not Card.PROMOTE_FAMILY]
                has_merit_card = any(c.can_use_merit for c in rest)
                has_money_card = any(c.can_use_money for c in rest)
                any_promo_card = any(c.is_promotion for c in rest)
            promoted = False
            if cfg.needs_both(rank):
                # 双条件台阶：任意晋升卡都行，但钱和政绩要同时够
                promoted = (
                    any_promo_card
                    and tc is not None and merit_after >= tc
                    and mc is not None and money_after >= mc
                )
                bribed = promoted and not has_merit_card
            elif has_merit_card and tc is not None and merit_after >= tc:
                promoted, bribed = True, False
            elif has_money_card and mc is not None and money_after >= mc:
                promoted, bribed = True, True
            else:
                bribed = False

            # 一纸调令排在干活后面：用这一轮赚到的凑够门槛就能升——政绩照算；
            # 这一轮贪的钱也能花，但要等举报结算完，被抓了钱就没了、官也升不成（下面风险里算）
            if (
                not promoted
                and Card.PROMOTE_FAMILY in cards
                and not any(c.is_promotion for c in cards if c is not Card.PROMOTE_FAMILY)
                and not cfg.needs_both(rank)
                and rank + 1 < cfg.president_rank
            ):
                if tc is not None and merit_after >= tc:
                    promoted, bribed, via_family = True, False, True
                elif mc is not None and money_after >= mc:
                    promoted, bribed, via_family = True, True, True
                    family_needs_loot = True

            if promoted:
                score = (1.0 - here) + self.w.promotion_bonus
            else:
                score = self._own_progress(public, private, rank, money_after, merit_after) - here
                # 打了晋升卡却升不上去 = 这张牌白费
                if any(c.is_promotion for c in cards):
                    score -= 0.03

        family = via_family and not family_needs_loot  # 真是它送上去、又不靠赃款，才是"官不撤"
        if Card.PROMOTE_FAMILY in cards:
            # 每局只有一次：手里有普通晋升卡能升的时候别浪费它；
            # 打了却不是它送上去的 = 纯浪费（修之前 AI 会配着贿赂升职一起打，
            # 因为风险按"官不撤"算轻了）
            # 每局一次的卡越往后越值钱（基层升县级最不值得用它），按官职给机会成本
            reserve = self.w.family_card_reserve * (1.0 + 0.5 * max(0, 2 - rank))
            score -= reserve if via_family else 1.0

        # ---- 被举报查实的风险：贪污和行贿都算 ----
        # 以前只有贪污/以权谋私才扣这一项，而且"先升官再干活"那条路整个跳过了。
        # 可举报同样抓行贿（REPORT_CATCHES_BRIBERY）：查实了官作废、钱打水漂。
        # AI 把买官当成零风险，钱越多越急着买——复盘：富二代开局从 10 块加到 15 块，
        # 第 1、2 轮买官被查实的比例从 7% 涨到 17% / 23%，胜率反而掉了。
        dirty = bool(n_corrupt or n_graft)
        risk = 0.0
        if dirty:
            # 被攻击撞上会被迫掏打点费，这也是贪污的成本之一
            if cfg.attack_mode == "denial" and cfg.attack_on_corruption == "merit_to_attacker":
                p_hit = self._p_being_attacked()
                score -= p_hit * float(cfg.attack_hush_money_ratio) * (
                    gain_money / mc if mc else 0.0
                )
        if dirty or (bribed and cfg.report_catches_bribery):
            p_caught = self._report_pressure(public, private)
            if dirty and event is not None and event.flag("storm_report"):
                # 知道这轮是反腐风暴：贪得最多的那 1/3 必被查，贪了基本跑不掉
                p_caught = max(p_caught, self.w.tipoff_storm_caught)
            loss = 0.0
            if dirty:
                gained = gain_money * self._exposed_share(my_origin)
                major = gained >= cfg.major_corruption_threshold
                loss += (rank * 1.0 + self._progress(rank, money_after, 0)) if major else \
                    self._progress(rank, gained, 0)
            if family_needs_loot:
                # 靠这一轮的赃款用一纸调令：被抓了钱抄走、官没升成，**这张卡也白扔了**
                loss += self.w.family_card_reserve * (1.0 + 0.5 * max(0, 2 - rank))
            if bribed and cfg.report_catches_bribery and not family_needs_loot:
                # （靠赃款的一纸调令被抓时只是没升成，钱的损失已经算在上面的贪污里了）
                if family:
                    # 一纸调令用钱升：查实只记警告（钱本来就花了），官不撤
                    loss += self.w.family_warning_cost
                else:
                    # 买官的钱打水漂（按规则里打水漂的比例）
                    loss += self.w.bribe_caught_loss * float(cfg.bribe_forfeit_ratio)
            if promoted and not family:
                loss += self.w.promotion_bonus  # 这一级也作废了（一纸调令不会被撤）
            risk = self.w.caught_dread * p_caught * loss
            score -= risk
        # 学习型 AI 拿这些当特征（见 score_combos / LearnedAgent）
        self._econ_detail = {
            "promote": float(promoted),
            "promote_money": float(promoted and bribed),
            "promote_merit": float(promoted and not bribed),
            "dirty": float(dirty),
            "merit_gain": gain_merit / tc if tc else 0.0,
            "money_gain": gain_money / mc if mc else 0.0,
            "risk": risk,
        }
        return score, promoted and rank + 1 >= cfg.president_rank

    # -- 干扰牌的期望收益（单位：官职；生产牌走 _score_economy） --------

    def _attack_damage(self, opp, model) -> tuple[int, int]:
        """按当前 attack_mode 算出 (目标损失, 我拿到)，单位是政绩点数。"""
        cfg = self.cfg
        t_rank, t_merit = opp["rank"], opp["merit"]
        mode = cfg.attack_mode
        if mode == "denial":
            # 拦下晋升 -> 政绩清零；否则只蹭掉一点。攻击者不从政绩里拿东西。
            tc = cfg.merit_cost(t_rank)
            about_to_cash_in = tc is not None and t_merit >= tc
            if about_to_cash_in and cfg.attack_wipes_merit_on_block:
                # 他得真的摸到并打出晋升卡才会被清零，按摸到的概率打个折
                damage = int(t_merit * self._p_plays_promotion_card())
            else:
                damage = min(cfg.attack_merit_penalty, t_merit)
            return damage, 0

        if mode == "negative_sum":
            full = math.floor(
                Fraction(EXPECTED_CARD_VALUE)
                * cfg.attack_damage_turns
                * cfg.work_multiplier(t_rank)
            )
            damage = min(full, t_merit)
            gain = math.floor(
                Fraction(damage)
                * cfg.attack_gain_ratio
                * cfg.work_multiplier(self._my_rank)
                / cfg.work_multiplier(t_rank)
            )
            return damage, gain
        if mode == "steal_merit":
            damage = math.floor(Fraction(t_merit) * cfg.attack_steal_fraction)
            return damage, damage
        if mode == "steal_work":
            expected = int(self._expected_work(t_rank) * model.work_rate)
            damage = math.floor(Fraction(expected) * cfg.attack_steal_fraction)
            gain = damage
            gap = t_rank - self._my_rank
            if gap > 0 and cfg.attack_steal_rank_bonus:
                gain += math.floor(Fraction(gain) * cfg.attack_steal_rank_bonus * gap)
            if cfg.attack_hat_reward:
                # 他没在干活就扣得成帽子，我自己记一点功
                gain += int(
                    (1 - model.work_rate)
                    * rules.work_merit(cfg.attack_hat_reward, self._my_rank, None, cfg)
                )
            return damage, gain
        # merit_penalty：只扣不拿，而且 WORK 玩家免疫
        if cfg.attack_spares_workers:
            return int(cfg.attack_merit_penalty * (1 - model.work_rate)), 0
        return cfg.attack_merit_penalty, 0

    def _p_plays_promotion_card(self) -> float:
        """对手够门槛时，他这轮真的摸到并打出晋升卡的概率（按牌库权重估）。"""
        cfg = self.cfg
        d = cfg.card_deal_distribution
        total = sum(d.values())
        usable = d.get("PROMOTE_MERIT", 0) + d.get("PROMOTE_ANY", 0)
        if total <= 0 or usable <= 0:
            return 0.0
        return 1.0 - ((total - usable) / total) ** cfg.hand_size

    def _p_being_attacked(self) -> float:
        """我被攻击的频率。互攻会两败俱伤，所以出手前得掂量对方会不会回手。"""
        return min(0.9, (self._times_attacked + 0.5) / (self._rounds_seen + 2.0))

    def _score_attack(self, public, private, opp) -> tuple[float, int | None]:
        """攻击的期望收益。

        拆成三块：抢到手的政绩（归我，不稀释）、对方的损失（公共品，要稀释）、
        以及互攻两败俱伤的风险折扣。
        """
        cfg = self.cfg
        t_rank, t_merit = opp["rank"], opp["merit"]
        t_origin = opp.get("origin")
        tc = rules.merit_cost_at(t_rank, t_origin, cfg)
        if tc is None:
            return 0.0, opp["id"]
        model = self.models.get(opp["id"], OpponentModel())
        my_rank, my_merit, my_money = private["rank"], private["merit"], private["money"]
        my_tc = cfg.merit_cost(my_rank)

        self._my_rank = my_rank
        damage, my_gain = self._attack_damage(opp, model)

        # --- denial 模式：唯一的正收益是撞上对方本轮在贪污，起获赃款 ---
        self_value = 0.0
        if cfg.attack_mode == "denial":
            p_corrupt = min(0.95, model.corrupt_rate * 2.0)
            haul = self._expected_corrupt(t_rank)
            if cfg.attack_on_corruption == "confiscate_to_attacker":
                mc = cfg.money_cost(my_rank)
                if mc:
                    self_value += p_corrupt * (haul / mc)
            elif cfg.attack_on_corruption == "merit_to_attacker":
                my_tc = cfg.merit_cost(my_rank)
                if my_tc:
                    gain = haul * float(cfg.attack_corruption_merit_ratio)
                    self_value += p_corrupt * (gain / my_tc)
            # 逼对方掏打点费也是实打实的打击（但好处是全桌分的，归进 deny）
            hush_hit = p_corrupt * float(cfg.attack_hush_money_ratio) * (
                haul / (cfg.money_cost(t_rank) or 1)
            )
            # 抓贪腐的那点好处也要看对象：揪着一个离胜利很远的人薅羊毛，
            # 不该压过去拦一个快赢的人。按威胁值打个折。
            self_value *= 0.35 + 0.65 * self._threat(
                t_rank, self._progress(t_rank, model.money_est, t_merit, t_origin)
            )

        if my_gain > 0 and my_tc is not None:
            here = self._progress(my_rank, my_money, my_merit)
            after = self._progress(my_rank, my_money, my_merit + my_gain)
            self_value = after - here
            if my_merit + my_gain >= my_tc:
                self_value = (1.0 - here) + self.w.promotion_bonus
            if cfg.attack_mutual_cancels_gain and cfg.attack_mode == "negative_sum":
                # 他要是也在打我，这一下就白费了
                self_value *= 1.0 - self._p_being_attacked()
        # 抢功的好处要看打的是谁：盯着一个基层抢政绩，等于放着官最大的那个不管
        self_value *= self._target_focus(public, opp)

        # --- 对方的损失（好处全桌分，要打稀释折扣）---
        hush_hit = locals().get("hush_hit", 0.0)
        threat = self._threat(
            t_rank, self._progress(t_rank, model.money_est, t_merit, t_origin)
        )
        share = self._share(public, threat)
        # 打掉的政绩最多只值"他一级晋升的进度"。不封顶的话，denial 模式下
        # damage 是对方整个政绩存量，能算出"一次攻击挡掉两级"这种荒谬估值。
        deny = min(damage / tc, 1.0) * threat

        if t_merit >= tc:
            p_block = 1.0
        else:
            need = tc - t_merit
            work_gain = self._expected_work(t_rank)
            p_cover = 1.0 if need <= work_gain * 0.8 else (0.5 if need <= work_gain * 1.2 else 0.0)
            p_block = model.work_rate * p_cover
        # 拦晋升值多少，要看它是"毁掉"还是只"拖一轮"：
        # 清零关掉之后这一刀不掉政绩，对方下轮照样能升，价值小得多。
        block_value = (
            (1.0 + self.w.promotion_bonus)
            if cfg.attack_wipes_merit_on_block
            else self.w.promotion_bonus
        )
        if not cfg.attack_wipes_merit_on_block:
            # 挡下时他还要掉一部分政绩（比例 + 固定值）：按他的门槛折成进度
            block_value += (
                float(cfg.attack_block_merit_loss) * t_merit
                + rules.work_merit(cfg.attack_block_merit_penalty, t_rank, None, cfg)
            ) / tc
        if (cfg.origin(t_origin) or {}).get("id") == "PEASANT":
            # 贫农「政治正确」：一个人挡不住他，得有别人也一起打才按得住
            block_value *= self._peasant_block_chance()
        deny += p_block * block_value * threat

        if cfg.attack_resets_tenure and cfg.tenure_required > 0:
            deny += (opp["tenure"] / cfg.tenure_required) * threat
        deny += hush_hit * threat

        # 整体再封一次顶：一次攻击最多值"让他少升一级"
        deny = min(deny, (1.0 + self.w.promotion_bonus) * threat)

        # 但拦住"这一轮就可能登顶"的人是例外：那不是少升一级，是阻止整局结束。
        # 这份好处也不该按人数稀释——他赢了，桌上每个人都输。
        # 单张攻击只按得住政绩那条路，所以只算 cover 那一份（见 _endgame_cover）。
        if self._about_to_win(opp, model):
            cover = self._endgame_cover(opp, model)[0]
            return (
                self.w.interfere_attack * self_value
                + self._endgame_stop_value(opp, model) * cover,
                opp["id"],
            )
        return self.w.interfere_attack * (self_value + deny * share), opp["id"]

    def _report_hit_prob(self, public, opp) -> tuple[float, float, float]:
        """举报他这一下能查实的概率 -> (他在贪的概率, 他在买官的概率, 查实概率)。纯函数，不碰 rng。"""
        return self._read(public, opp, self.models.get(opp["id"], OpponentModel()))

    def _read(self, public, opp, model: "OpponentModel") -> tuple[float, float, float]:
        """按公开信息 + 这个人的档案读他（对手和"别人眼里的我"共用）。"""
        cfg = self.cfg
        t_rank = opp["rank"]
        origin = opp.get("origin")
        mc_t, tc_t = self._costs(t_rank, origin)
        if mc_t is None:
            return 0.0, 0.0, 0.0
        rnd = int(public.get("round", 1))
        ratio = model.money_est / mc_t if mc_t else 0.0
        merit_ready = tc_t is not None and opp["merit"] >= tc_t
        quiet = model.quiet_rate
        oid = (cfg.origin(origin) or {}).get("id")

        # 因子表的来源和含义见文件开头 READ_DAMPING 那段
        f = (_band(CORRUPT_BY_MONEY, ratio) * _band(CORRUPT_BY_ROUND, rnd)
             * _band(CORRUPT_BY_QUIET, quiet) * CORRUPT_BY_ORIGIN.get(oid, 1.0)
             * CORRUPT_BY_RANK[min(t_rank, len(CORRUPT_BY_RANK) - 1)])
        p_corrupt = min(0.9, CORRUPT_BASE * f ** READ_DAMPING
                        * _band(CORRUPT_BY_RECORD, model.record))
        p_corrupt += self.w.read_shrink * (CORRUPT_BASE - p_corrupt)

        p_bribe = 0.0
        if cfg.report_catches_bribery:
            named = rnd - model.last_seen_corrupting <= 1  # 上一轮被传闻点名 = 手里有钱
            f = (_band(BRIBE_BY_MONEY, ratio) * _band(BRIBE_BY_ROUND, rnd)
                 * _band(BRIBE_BY_QUIET, quiet) * BRIBE_BY_ORIGIN.get(oid, 1.0)
                 * BRIBE_BY_RANK[min(t_rank, len(BRIBE_BY_RANK) - 1)]
                 * (1.22 if named else 0.97) * (0.63 if merit_ready else 1.05))
            p_bribe = min(0.9, BRIBE_BASE * f ** READ_DAMPING
                          * _band(BRIBE_BY_RECORD, model.record))
            if cfg.needs_both(t_rank):
                # 省级这一步"掏钱"就是冲主席。数据：AI 判他要登顶时 28% 真冲了
                # （政绩已经够的 35%），没判要登顶的只有 3%
                # （学习型 AI：判要登顶时 60% 真冲了，没判的 7%）
                p_bribe = 0.6 if self._about_to_win(opp, model) else 0.07
        cal = self._read_calibration()
        p_corrupt, p_bribe = min(0.95, p_corrupt * cal), min(0.95, p_bribe * cal)
        p_hit = min(0.95, 1.0 - (1.0 - p_corrupt) * (1.0 - p_bribe))
        return p_corrupt, p_bribe, p_hit

    def _read_calibration(self) -> float:
        """这一桌的人比我的读法更脏还是更干净（0.4~2.5 的乘数）。

        每轮公开：谁被玩家举报了、谁被查实了。被举报的人里实际查实了几个，
        对比我当初对这些人估的查实概率，估高了往下调、估低了往上调。
        因子表是拿 AI 对局拟合的，真人的打法不一样——这一项让读法在每一局里自己对上。
        """
        return max(0.4, min(2.5, (self._cal_hits + 3.0) / (self._cal_expect + 3.0)))

    def _score_report(self, public, private, opp) -> tuple[float, int | None]:
        """举报是一次对隐藏信息的下注：他这轮到底有没有经济问题。

        现在"有问题"包括两件事：贪了钱，或者拿钱买官。后者尤其关键——
        一个攒够钱、马上要买官上位的人，举报就是确定能拦住他的手段。
        """
        cfg = self.cfg
        pid = opp["id"]
        model = self.models.get(pid, OpponentModel())
        t_rank = opp["rank"]
        t_origin = opp.get("origin")
        my_rank, my_money = private["rank"], private["money"]
        mc_t, tc_t = self._costs(t_rank, t_origin)
        merit_ready = tc_t is not None and opp["merit"] >= tc_t
        p_corrupt, p_bribe, p_hit = self._report_hit_prob(public, opp)
        if p_hit <= 0:
            return 0.0, pid

        # --- 查实能抄到多少（只没收本轮那一笔 + 查获的行贿，还要按比例分）---
        haul = p_corrupt * self._expected_corrupt(t_rank)
        seized_bribe = p_bribe * (mc_t or 0)
        pool = (haul + seized_bribe) * float(cfg.report_reward_ratio)
        expected_split = 1.0 + 0.25 * max(0, len(public["players"]) - 2)
        my_cut = max(0.0, pool / expected_split - cfg.report_reward_fee)

        mc = cfg.money_cost(my_rank)
        # 攒着的钱要按"还来不来得及花"打折；能当场换成一级官职的那部分不打折。
        cash_value = (my_cut / mc) * self._cash_horizon(public) if mc else 0.0
        if mc and my_money + my_cut >= mc:
            cash_value += self.w.promotion_bonus  # 这笔赃款直接把我送上去
        # 分赃的好处同样看对象：举报一个兜里有钱的基层，不如去按住官最大的那个
        cash_value *= self._target_focus(public, opp)

        # --- 打击面：冻结他这一轮的晋升 + 记一次警告（攒满就降级）---
        t_prog = self._progress(t_rank, model.money_est, opp["merit"], t_origin)
        threat = self._threat(t_rank, t_prog)
        setback = threat * 0.5
        # 查实必然冻结他本轮的晋升。他要是正打算升，这一下就是实打实拦住一级。
        about_to_promote = merit_ready or (mc_t is not None and model.money_est >= mc_t)
        if about_to_promote:
            setback += (1.0 + self.w.promotion_bonus) * threat
        # 离降级越近，这一次警告越值钱。
        # 红二代「硬保」例外：他降不下来，警告攒到天上也没用，这一项归零。
        if (cfg.origin(t_origin) or {}).get("id") != "RED":
            wmax = max(1, cfg.warnings_before_demotion)
            setback += (opp.get("warnings", 0) + 1) / wmax * threat * 0.5

        # 把人按住基本是公共品，赃款才是我的；但他越接近登顶，这份好处越是我自己的
        setback *= self._share(public, threat)

        # 拦住"下一步就夺冠"的人是例外：那不是少升一级，是阻止整局结束，
        # 这份好处也不该按人数稀释——他赢了，桌上每个人都输。
        # 单张举报只抓得住贿赂升职（或者他这轮得现贪），只算 cover 那一份。
        if self._about_to_win(opp, model):
            cover = self._endgame_cover(opp, model)[1]
            return (
                self.w.interfere_report * p_hit * cash_value
                + self._endgame_stop_value(opp, model) * cover,
                pid,
            )
        return self.w.interfere_report * p_hit * (cash_value + setback), pid


class LearnedAgent(SmartAgent):
    """学习型 AI：眼睛和手写 AI 一样（同一套观察、同一套组合打分拆出来的特征），
    怎么权衡这些特征由自我对局学出来（learn.py）。

    出牌 = 对每个组合的特征做线性打分，softmax 随机抽一个——天然是混合策略：
    这游戏是石头剪刀布，每次都出同一手的人会被看穿。
    权重里 "hand"（手写 AI 的总分）那一项初始化成 1/温度、其他为 0，第 0 代就约等于手写 AI。
    `record=True` 时把每次决策的 ∇log π 记进 trace，learn.py 按输赢回传。
    """

    def __init__(self, *args, theta: dict[str, float], record: bool = False,
                 theta_by_origin: dict[str, dict[str, float]] | None = None, **kw) -> None:
        super().__init__(*args, **kw)
        self.theta = theta
        # 每个出身专项训练出来的权重（learn.py --focus-origin）；没有的出身用 theta
        self.theta_by_origin = theta_by_origin or {}
        self.record = record
        self.trace: list[dict[str, float]] = []
        self.chosen_stats: Counter = Counter()  # 选中的组合里各特征的累计（learn.py 看打法怎么变）
        self.want_features = True
        # 对照实验用：我是头号挑战者、有人要登顶、手里有干扰牌时，只许在带干扰牌的组合里挑
        self.force_block = False

    def decide(self, public, private):
        self.observe(public, private)
        hand = [Card(d["card"]) if isinstance(d, dict) else Card(d) for d in private["hand"]]
        n_picks = min(int(public.get("picks_per_round", 1)), len(hand))
        if not hand or n_picks <= 0:
            self.last_scores = []
            return []
        scored = self.score_combos(public, private)
        self.last_scores = scored
        feats = self.last_features
        if not scored:
            return self._finish(public, private, [])
        scored, feats = self._apply_dogpile(public, hand, scored, feats)
        if self.force_block:
            opps = [o for o in public["players"] if o["id"] != self.id]
            killers = [o for o in opps
                       if self._about_to_win(o, self.models.get(o["id"], OpponentModel()))]
            if killers and max(self._contender(o["id"]) for o in killers) > 0.69:
                keep = [i for i, (_, cards) in enumerate(scored)
                        if Card.ATTACK in cards or Card.REPORT in cards]
                if keep:
                    scored = [scored[i] for i in keep]
                    feats = [feats[i] for i in keep]
        oid = (self.cfg.origin(private.get("origin")) or {}).get("id")
        theta = self.theta_by_origin.get(oid, self.theta)
        self._theta_now = theta  # 挑目标（_pick_target）也用这一份
        logits = [sum(theta.get(k, 0.0) * v for k, v in f.items()) for f in feats]
        top = max(logits)
        weights = [math.exp(min(0.0, x - top)) for x in logits]
        total = sum(weights)
        probs = [w / total for w in weights]
        r, acc, idx = self.rng.random(), 0.0, len(probs) - 1
        for i, p in enumerate(probs):
            acc += p
            if r < acc:
                idx = i
                break
        for k in ("dirty", "n_report", "n_attack", "promote", "promote_money", "family"):
            self.chosen_stats[k] += feats[idx].get(k, 0.0)
        if self.record:
            # softmax 线性策略：∇log π(a) = φ(a) − Σ π(b) φ(b)
            # 特征是稀疏的（"打的是谁"只有带干扰牌的组合才有），缺的当 0
            keys = set().union(*feats)
            self.trace.append({
                k: feats[idx].get(k, 0.0) - sum(p * f.get(k, 0.0) for p, f in zip(probs, feats))
                for k in keys
            })
        return self._finish(public, private, list(scored[idx][1]))

    def _pick_target(self, public, private, card, opponents, used):
        """学"打谁"：候选范围和手写一样（有人快赢就在快赢的人里挑，否则避开这轮打过的），
        在候选里按学到的权重打分、抽签。特征里有目标的出身和"出身 × 进度"，
        让它自己学会"官二代门槛低、要早点拦""贫农单刀攻击没用""红二代举报了也降不了级"。"""
        target = self._dogpile_target(public)
        if target is not None and not getattr(self, "_can_win_now", False):
            return target["id"]
        killer = [o for o in opponents
                  if self._about_to_win(o, self.models.get(o["id"], OpponentModel()))]
        pool = killer or [o for o in opponents if o["id"] not in used] or opponents
        if not pool:
            return None
        if len(pool) == 1:
            return pool[0]["id"]
        tag = "A" if card is Card.ATTACK else "R"
        scorer = self._score_attack if card is Card.ATTACK else self._score_report
        feats = [self._target_features(public, tag, o, scorer(public, private, o)[0]) for o in pool]
        theta = getattr(self, "_theta_now", self.theta)
        logits = [sum(theta.get(k, TARGET_INIT.get(k, 0.0)) * v for k, v in f.items()) for f in feats]
        top = max(logits)
        weights = [math.exp(min(0.0, x - top)) for x in logits]
        total = sum(weights)
        probs = [w / total for w in weights]
        r, acc, idx = self.rng.random(), 0.0, len(probs) - 1
        for i, p in enumerate(probs):
            acc += p
            if r < acc:
                idx = i
                break
        if self.record:
            keys = set().union(*feats)
            self.trace.append({
                k: feats[idx].get(k, 0.0) - sum(p * f.get(k, 0.0) for p, f in zip(probs, feats))
                for k in keys
            })
        return pool[idx]["id"]

    def _target_features(self, public, tag: str, o: dict[str, Any], base: float) -> dict[str, float]:
        m = self.models.get(o["id"], OpponentModel())
        mc, tc = self._costs(o["rank"], o.get("origin"))
        prog = min(1.5, o["merit"] / tc) if tc else 0.0
        rnd = int(public.get("round", 1))
        f = {
            f"{tag}:t_base": base,
            f"{tag}:t_progress": prog,
            f"{tag}:t_rank{o['rank']}": 1.0,
            f"{tag}:t_closing": float(self._about_to_win(o, m)),
            f"{tag}:t_money": min(2.0, m.money_est / mc) if mc else 0.0,
            f"{tag}:t_hit_me": float(self._last_attackers.get(o["id"], -9) >= rnd - 1),
        }
        oid = (self.cfg.origin(o.get("origin")) or {}).get("id")
        if oid:
            f[f"{tag}:o_{oid}"] = 1.0
            f[f"{tag}:o_{oid}*progress"] = prog
        if tag == "R":
            f["R:t_phit"] = self._report_hit_prob(public, o)[2]
        return f


# 学"打谁"那部分权重的起点：只看手写规则给这个目标的打分（= 第 0 代照搬手写的挑法）
TARGET_INIT = {"A:t_base": 50.0, "R:t_base": 50.0}


def load_policy(path: str) -> dict[str, float]:
    import json

    with open(path, encoding="utf-8") as fh:
        return dict(json.load(fh)["theta"])


def load_policy_bundle(path: str) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """(默认权重, {出身: 专项权重})。普通权重文件没有 theta_by_origin，第二项为空。"""
    import json

    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return dict(d["theta"]), {k: dict(v) for k, v in (d.get("theta_by_origin") or {}).items()}


_POLICY_CACHE: dict[str, tuple[dict[str, float], dict[str, dict[str, float]]]] = {}


def make_pool(cfg: Config = DEFAULT_CONFIG, rng: random.Random | None = None,
              **flags: Any) -> "AgentPool":
    """按 cfg.ai_policy 决定是手写 AI 还是学习型 AI（权重文件不在就退回手写）。

    服务器和平衡分析（analysis.py / sweep.py 里的 "smart"）都走这里，
    所以平衡报告测的永远是服务器上实际在用的那个 AI。
    """
    from pathlib import Path

    policy, by_origin = None, None
    if cfg.ai_policy:
        path = Path(cfg.ai_policy)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        if path.exists():
            key = str(path)
            if key not in _POLICY_CACHE:
                _POLICY_CACHE[key] = load_policy_bundle(key)
            policy, by_origin = _POLICY_CACHE[key]
    return AgentPool(cfg=cfg, rng=rng or random.Random(), policy=policy,
                     policy_by_origin=by_origin or None, **flags)


# --------------------------------------------------------------------------
# 给分析/模拟用的适配层
# --------------------------------------------------------------------------


@dataclass
class AgentPool:
    """按 player_id 持有 SmartAgent，跨轮保留记忆。"""

    cfg: Config = DEFAULT_CONFIG
    weights: Weights = field(default_factory=Weights)
    rng: random.Random = field(default_factory=random.Random)
    allow_attack: bool = True
    allow_report: bool = True
    allow_corrupt: bool = True
    agents: dict[int, SmartAgent] = field(default_factory=dict)
    policy: dict[str, float] | None = None  # 给了就用学习型 AI（LearnedAgent）
    policy_by_origin: dict[str, dict[str, float]] | None = None  # 每个出身的专项权重
    record: bool = False

    def get(self, player_id: int) -> SmartAgent:
        if player_id not in self.agents:
            kw = dict(
                cfg=self.cfg,
                weights=self.weights,
                rng=self.rng,
                allow_attack=self.allow_attack,
                allow_report=self.allow_report,
                allow_corrupt=self.allow_corrupt,
            )
            self.agents[player_id] = (
                LearnedAgent(player_id, theta=self.policy, record=self.record,
                             theta_by_origin=self.policy_by_origin, **kw)
                if self.policy is not None else SmartAgent(player_id, **kw)
            )
        return self.agents[player_id]


MAX_PANIC_REDRAWS = 3
"""终局抢救时最多连换几手牌。

价格每换一次翻倍，所以真正的刹车是钱包；这个上限只是防止
一个巨富 AI 把整轮时间耗在洗牌上。
"""


def wants_redraw(game, player_id: int, pool: AgentPool) -> bool:
    """要不要花钱重抽这一手牌。

    只有一种情况值得：**有人快要夺冠，而我手上一张干扰牌都没有**。
    这时候一个回合的产出毫无意义（他赢了就结束了），砸钱换牌去拦他才是对的。
    其余时候不换：换牌的钱和攒钱升职抢的是同一个钱包。

    "快要夺冠"取两条，满足一条就算：估计资源已经够了（`_about_to_win`），
    或者人已经站在主席门口、晋升进度过了 75%。后面这条是因为钱是暗的，
    估计值总是偏低——等它真的过线，往往就是他登顶的那一轮了。
    """
    agent = pool.get(player_id)
    public = game.public_state()
    private = game.private_state(player_id)
    agent.observe(public, private)

    if not private.get("redraw_affordable"):
        return False
    # 富二代的免费换牌：不花钱，手牌烂就换，跟真人一样
    if private.get("redraw_cost") == 0 and agent._hand_is_weak(private):
        return True
    if agent._dogpile_target(public) is not None:
        # 围堵默契：手里攻击、举报都没有就花钱重抽，抽到能拦他的牌为止
        return not any(Card(d["card"]).needs_target for d in private["hand"])
    if any(Card(d["card"]).needs_target for d in private["hand"]):
        return False  # 手上已经有举报或攻击了

    for opp in public["players"]:
        if opp["id"] == player_id:
            continue
        model = agent.models.get(opp["id"], OpponentModel())
        if agent._about_to_win(opp, model):
            return True
        # 还差一点、但已经站在主席门口的人，也值得砸钱换一手牌去拦。
        # 等到估计值真的过线往往已经晚了——他那一轮就登顶了。
        if opp["rank"] == agent.cfg.president_rank - 1:
            prog = agent._progress(
                opp["rank"], model.money_est, opp["merit"], opp.get("origin")
            )
            if prog >= 0.75:
                return True
    return False


def turn(game, player_id: int, pool: AgentPool) -> list[dict]:
    """AI 的一整个回合：该换牌就换（可能连换几次），再挑牌。

    服务器和所有平衡分析都走这一个入口。以前只有服务器会换牌，analysis.py 里的
    对局从来不换——于是"换牌"相关的改动（比如富二代每轮一次免费换牌）
    在平衡报告里的价值恒为 0，测了也白测。
    """
    for _ in range(MAX_PANIC_REDRAWS):
        if not wants_redraw(game, player_id, pool):
            break
        game.redraw(player_id)
    return choose(game, player_id, pool)


def choose_origin(game, player_id: int, pool: AgentPool) -> str:
    """AI 挑出身。**v1 就是随机挑**。

    故意不做聪明的：平衡数据全部来自"强制随机分配"的对局，AI 会不会挑好
    身份不影响那些数字。等六张牌的强弱定下来了再回头教它挑，
    否则会陷入"AI 偏爱强身份 -> 强身份看起来更强"的循环论证。
    """
    agent = pool.get(player_id)
    return agent.rng.choice(game.private_state(player_id)["origin_choices"])["id"]


def choose(game, player_id: int, pool: AgentPool) -> list[dict]:
    """用 game 的公开/私密 payload 驱动 AI —— 和浏览器拿到的完全一样。

    返回 game.select_actions() 能直接吃的格式。
    """
    agent = pool.get(player_id)
    picks = agent.decide(game.public_state(), game.private_state(player_id))
    return [{"action": c, "target": t} for c, t in picks]
