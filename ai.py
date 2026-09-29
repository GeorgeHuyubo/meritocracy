"""会推理的 AI 玩家。

和 analysis.py 里那些启发式 bot 最大的区别：

* 它**只吃 `game.public_state()` 和自己的 `game.private_state(pid)`**，
  也就是浏览器里那个玩家能看到的一模一样的 JSON。作弊在结构上就不可能。
* 它维护一份对每个对手**金钱的估计**。金钱是隐藏的，但可以从公开信息推断：
    - 财富广播点名了谁，以及那一档的金额区间（档位见 config.WEALTH_BROADCAST_TIERS）
    - 某人政绩没涨 => 他本轮没打 WORK => 有可能在贪
    - 政绩涨了也**不能**判他清白：一轮打两张牌，可以工作+贪污，
      而以权谋私本身就同时给政绩和钱，看起来和埋头干活一模一样
    - 完全没有财富广播 => 本轮**没有任何人**贪污，所有人的钱都没变
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
from itertools import combinations
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import rules
from config import Config, DEFAULT_CONFIG
from models import Card

EXPECTED_CARD_VALUE = 10  # 兜底用的牌面期望（正常情况下用发牌时摇好的真实点数）


def _early_promotion_rank(cfg, rank, money, merit, cards):
    """如果这组牌能在"动手之前"就把官升了，返回 (新官职, 升职后的钱, 升职后的政绩)。

    要和 rules.apply_promotion_costs 保持一致；AI 不这么算就会误判组合技的价值。
    """
    if not any(c.is_promotion for c in cards) and cfg.promotion_requires_card:
        return None
    mc, tc = cfg.money_cost(rank), cfg.merit_cost(rank)
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
    interfere_report: float = 0.2
    caught_dread: float = 1.0  # 对"贪污被抓"的恐惧程度
    base_report_pressure: float = 0.22  # 单个对手本轮举报我的基准概率
    leader_suspicion: float = 2.0  # 我是明面领先者时，被举报概率的放大倍数
    # 威胁评估：干扰一个离赢还很远的人值几折。
    # 0.45 太平了（基层 0.45 vs 省级 1.0，只差 2.2 倍），AI 会无差别攻击。
    # 配合 threat_exponent=2 之后差距拉到约 12 倍，攻击自然集中到快赢的人身上。
    threat_floor: float = 0.08
    threat_exponent: float = 2.0
    # 拦住一个下一级就是国家主席的人，等于直接阻止别人赢下整局，
    # 这份好处不该按"全桌平分"打折——他赢了我就全输。
    endgame_block_weight: float = 4.0
    # 判断"他下一步就夺冠"时，钱这一项按门槛的几折算。
    # 钱是暗的，money_est 是下界（贪污看不见、以权谋私伪装成干活），
    # 1.0 意味着只信估计值，结果就是终局刹车形同虚设。
    endgame_money_doubt: float = 0.6
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
        # 全场层面的自适应估计：这桌人到底抓得有多凶
        self.est_corrupt_attempts = 0.0  # 估计发生过多少次贪污
        self._times_attacked = 0  # 我自己被攻击过几次
        self._rounds_seen = 0
        self._my_rank = 0
        self.observed_demotions = 0  # 公开可见的"被查实"次数（举报压力的证据）

    # ------------------------------------------------------------------
    # 观察：把公开 payload 变成对隐藏金钱的估计
    # ------------------------------------------------------------------

    def observe(self, public: dict[str, Any]) -> None:
        players = {p["id"]: p for p in public["players"]}
        for pid in players:
            if pid != self.id:
                self.models.setdefault(pid, OpponentModel())

        result = public.get("last_result")
        if result is None or result.get("round") == self._last_round_seen:
            self._prev_players = players
            return
        self._last_round_seen = result["round"]

        facts = {f["player_id"]: f for f in result.get("player_facts", [])}
        self._rounds_seen += 1
        if facts.get(self.id, {}).get("attacked"):
            self._times_attacked += 1
        top_ids = set(result.get("wealth_top_ids") or [])
        tier = result.get("wealth_tier")
        anyone_corrupted = bool(top_ids)
        tier_low, tier_high = (
            rules.wealth_tier_range(tier, self.cfg) if tier is not None else (0, None)
        )

        for pid, model in self.models.items():
            cur, prev = players.get(pid), self._prev_players.get(pid)
            fact = facts.get(pid)
            if cur is None or prev is None or fact is None:
                continue
            model.observed_rounds += 1
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
            gained = 0.0
            if pid in top_ids:
                # 被点名 = 他就是本轮的贪污榜首，金额落在这个档位里
                gained = float(tier_low if tier_high is None else (tier_low + tier_high) / 2)
                if tier_high is None:
                    gained = max(gained, self._expected_corrupt(rank_before))
                model.last_seen_corrupting = result["round"]
                model.corrupt_evidence += 1.0
            elif not anyone_corrupted:
                gained = 0.0  # 没有广播 => 全场无人贪污，这是确定信息
            elif worked:
                # 「政绩涨了」**不等于**「没捞钱」——每轮能打 2 张牌，
                # 完全可以 工作 + 贪污；而以权谋私更是一张牌同时给政绩和钱，
                # 打出来看上去和埋头工作一模一样，钱却照样算赃款。
                # 以前这里直接记 0，于是闷声发财的领先者被系统性低估：
                # 复盘 F7KF 第 7 轮，老张实际有 45 块（门槛 37），AI 只估到 26，
                # 「他下一步就夺冠」那道刹车因此一次都没踩。
                p_dirty = 0.45 if int(self.cfg.picks_per_round) > 1 else 0.0
                gained = p_dirty * self._expected_graft_money(rank_before)
                model.corrupt_evidence += p_dirty
            else:
                # 他没打 WORK，可能在贪、在举报、在攻击。贪的话也一定不超过榜首。
                p_corrupt = 0.45
                cap = tier_high if tier_high is not None else self._expected_corrupt(rank_before)
                gained = p_corrupt * min(self._expected_corrupt(rank_before), float(cap))
                model.corrupt_evidence += p_corrupt
            model.money_est += gained + self.cfg.salary(rank_before)  # 工资是公开可算的
            self.est_corrupt_attempts += 1.0 if pid in top_ids else (gained > 0) * 0.45
            # 查实的证据现在看"记没记警告"，不能再看降级：
            # 攒够两次才降一级，只数降级会把举报压力低估一半。
            if fact.get("warnings_issued", 0) > 0 or fact["demotion"] != "NONE":
                self.observed_demotions += 1

            # --- 被举报没收：只没收本轮那一笔，存款不再被抄 ---
            if fact.get("warnings_issued", 0) > 0 or fact["demotion"] != "NONE":
                model.money_est = max(0.0, model.money_est - gained)

            # --- 晋升对钱的影响 ---
            if fact["promotion"] == "MONEY":
                cost = self.cfg.money_cost(rank_before) or 0
                model.money_est = max(model.money_est, float(cost))
                model.money_est = math.ceil((model.money_est - cost) / self.cfg.overflow_divisor)
                model.money_est = max(0.0, model.money_est)

        self._prev_players = players

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

    def _progress(self, rank: int, money: float, merit: float) -> float:
        """离下一级还剩多少，0~1。

        双条件台阶上要两样都够，所以看的是**短板**那一项，不是强项。
        """
        cfg = self.cfg
        mc, tc = cfg.money_cost(rank), cfg.merit_cost(rank)
        if mc is None or tc is None:
            return 1.0
        if cfg.needs_both(rank):
            return min(min(money / mc, 1.0), min(merit / tc, 1.0))
        return max(min(money / mc, 1.0), min(merit / tc, 1.0))

    def _threat(self, rank: int, progress: float) -> float:
        """这个对手有多值得我花一个回合去按住 —— 看他离夺冠还有多远。

        用"还差几级"而不是"现在几级"，并且是加速的：差 3 级的人基本不值得理，
        差半级的人必须按住。
        """
        top = self.cfg.president_rank
        steps_left = (top - rank) - progress
        if steps_left <= 0:
            return 1.0
        closeness = max(0.0, 1.0 - steps_left / top)
        return self.w.threat_floor + (1.0 - self.w.threat_floor) * (
            closeness ** self.w.threat_exponent
        )

    def _about_to_win(self, opp: dict[str, Any], model: "OpponentModel") -> bool:
        """他是不是下一次晋升就直接当主席了（而且资源已经够了）。

        政绩是公开的，按实数比。**钱是暗的**，`money_est` 只能算个下界——
        贪污看不见，以权谋私看起来又像在老实干活。所以钱这一项留出
        `endgame_money_doubt` 的余量：宁可多拦一次，也不要在他登顶那轮
        才发现自己估少了。少拦一次的代价是整局输掉，多拦一次只亏一个回合。
        """
        cfg = self.cfg
        rank = opp["rank"]
        if rank != cfg.president_rank - 1:
            return False
        tc, mc = cfg.merit_cost(rank), cfg.money_cost(rank)
        merit_ready = tc is not None and opp["merit"] >= tc
        money_ready = mc is not None and model.money_est >= mc * self.w.endgame_money_doubt
        if cfg.needs_both(rank):
            return merit_ready and money_ready
        return merit_ready or money_ready

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
        prior_strength = 3.0
        empirical = (
            prior_strength * self.w.base_report_pressure + self.observed_demotions
        ) / (prior_strength + self.est_corrupt_attempts)
        p = min(0.95, max(0.02, empirical))

        ranks = [(p_["rank"], p_["merit"]) for p_ in public["players"]]
        mine = (private["rank"], private["merit"])
        if ranks and mine >= max(ranks):
            p = min(0.95, p * self.w.leader_suspicion)  # 明面上的领先者最招人举报
        return p

    # ------------------------------------------------------------------
    # 决策
    # ------------------------------------------------------------------

    def decide(
        self, public: dict[str, Any], private: dict[str, Any]
    ) -> list[tuple[str, int | None]]:
        """返回本轮要打的 N 张牌 [(卡名, 目标id), ...]。

        不能逐张挑最高分就完事——「WORK + 政绩升职」是个组合技，
        单独看每张牌会漏掉协同。所以先给干扰牌定好目标，再**成对**评估经济牌。
        """
        self.observe(public)

        dealt = [
            (Card(d["card"]) if isinstance(d, dict) else Card(d),
             int(d.get("value", 0)) if isinstance(d, dict) else 0)
            for d in private["hand"]
        ]
        hand = [c for c, _ in dealt]
        values = [v for _, v in dealt]
        n_picks = min(int(public.get("picks_per_round", 1)), len(hand))
        if not hand or n_picks <= 0:
            return []

        opponents = [p for p in public["players"] if p["id"] != self.id]

        # 1) 干扰牌：各自挑好最优目标，算出独立分
        solo: dict[Card, tuple[float, int | None]] = {}
        for card in set(hand):
            if card is Card.ATTACK and self.allow_attack and opponents:
                solo[card] = self._best_target(
                    [self._score_attack(public, private, o) for o in opponents]
                )
            elif card is Card.REPORT and self.allow_report and opponents:
                solo[card] = self._best_target(
                    [self._score_report(public, private, o) for o in opponents]
                )
            else:
                solo[card] = (0.0, None)

        # 2) 枚举所有出牌组合（按手牌下标，所以同名牌能出两张）
        best_score, best_combo = None, None
        for combo in combinations(range(len(hand)), n_picks):
            cards = [hand[i] for i in combo]
            if not self.allow_corrupt and Card.CORRUPT in cards:
                continue
            score = self._score_economy(
                public, private, cards, [values[i] for i in combo]
            )
            for c in cards:
                if c in (Card.ATTACK, Card.REPORT):
                    score += solo[c][0]
            score += self.rng.uniform(-self.w.noise, self.w.noise)
            if best_score is None or score > best_score:
                best_score, best_combo = score, cards

        chosen = list(best_combo or hand[:n_picks])
        chosen = self._order_picks(private, chosen)

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
        self.decision_log.append(
            f"r{public.get('round')} -> " + ", ".join(
                f"{c}{'' if t is None else f'@{t}'}" for c, t in picks
            )
        )
        return picks

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
    ) -> float:
        """联合评估这组牌里的生产牌 + 晋升卡：先赚后升，一轮最多升一级。"""
        cfg = self.cfg
        rank, money, merit = private["rank"], private["money"], private["merit"]
        mc, tc = cfg.money_cost(rank), cfg.merit_cost(rank)
        here = self._progress(rank, money, merit)

        # 点数在发牌时就摇好了，所以这里用**确定值**算，不再用期望值猜
        vals = values or [0] * len(cards)
        n_corrupt = sum(1 for c in cards if c is Card.CORRUPT)
        n_graft = sum(1 for c in cards if c is Card.GRAFT)
        gain_merit = gain_money = 0.0
        for c, v in zip(cards, vals):
            if c is Card.WORK:
                gain_merit += (
                    rules.work_merit(v, rank, None, cfg) if v > 0 else self._expected_work(rank)
                )
            elif c is Card.CORRUPT:
                gain_money += (
                    rules.corrupt_money(v, rank, None, cfg)
                    if v > 0
                    else self._expected_corrupt(rank)
                )
            elif c is Card.GRAFT:
                if v > 0:
                    gain_money += rules.corrupt_money(v, rank, None, cfg)
                    gain_merit += rules.graft_merit(v, rank, None, cfg)
                else:
                    gain_money += self._expected_graft_money(rank)
                    gain_merit += self._expected_graft_merit(rank)
        merit_after = merit + gain_merit
        money_after = money + gain_money
        # 工资不用在这里加：它在回合开头就发了，private["money"] 里已经含着了。
        # 再加一次就是凭空多算一笔。

        # 结算顺序是动态的：本来就够门槛的话会先升官、再按新倍率干活。
        # 这里照着重算一遍，否则 AI 会低估"WORK + 晋升卡"这套组合技。
        early = _early_promotion_rank(cfg, rank, money, merit, cards)
        if early is not None:
            new_rank, money, merit = early
            gain_merit = gain_money = 0.0
            for c, v in zip(cards, vals):
                if c is Card.WORK:
                    gain_merit += (
                        rules.work_merit(v, new_rank, None, cfg)
                        if v > 0 else EXPECTED_CARD_VALUE * float(cfg.work_multiplier(new_rank))
                    )
                elif c is Card.CORRUPT:
                    gain_money += (
                        rules.corrupt_money(v, new_rank, None, cfg)
                        if v > 0 else self._expected_corrupt(new_rank)
                    )
                elif c is Card.GRAFT:
                    if v > 0:
                        gain_money += rules.corrupt_money(v, new_rank, None, cfg)
                        gain_merit += rules.graft_merit(v, new_rank, None, cfg)
            merit_after = merit + gain_merit
            money_after = money + gain_money  # 工资已在回合开头到账，别重复计
            here_after = self._progress(new_rank, money_after, merit_after)
            return (1.0 - here) + self.w.promotion_bonus + here_after * 0.5

        has_merit_card = any(c.can_use_merit for c in cards)
        has_money_card = any(c.can_use_money for c in cards)
        any_promo_card = any(c.is_promotion for c in cards)
        if not cfg.promotion_requires_card:
            has_merit_card = has_money_card = any_promo_card = True

        promoted = False
        if cfg.needs_both(rank):
            # 双条件台阶：任意晋升卡都行，但钱和政绩要同时够
            promoted = (
                any_promo_card
                and tc is not None and merit_after >= tc
                and mc is not None and money_after >= mc
            )
        elif has_merit_card and tc is not None and merit_after >= tc:
            promoted = True
        elif has_money_card and mc is not None and money_after >= mc:
            promoted = True

        if promoted:
            score = (1.0 - here) + self.w.promotion_bonus
        else:
            score = self._progress(rank, money_after, merit_after) - here
            # 打了晋升卡却升不上去 = 这张牌白费
            if any(c.is_promotion for c in cards):
                score -= 0.03

        if n_corrupt or n_graft:
            # 被攻击撞上会被迫掏打点费，这也是贪污的成本之一
            if cfg.attack_mode == "denial" and cfg.attack_on_corruption == "merit_to_attacker":
                p_hit = self._p_being_attacked()
                score -= p_hit * float(cfg.attack_hush_money_ratio) * (
                    gain_money / mc if mc else 0.0
                )
            p_caught = self._report_pressure(public, private)
            gained = gain_money
            major = gained >= cfg.major_corruption_threshold
            loss = (rank * 1.0 + self._progress(rank, money_after, 0)) if major else \
                self._progress(rank, gained, 0)
            if promoted:
                loss += self.w.promotion_bonus
            score -= self.w.caught_dread * p_caught * loss
        return score

    # -- 各张牌的期望收益（单位：官职） --------------------------------

    def _score_work(self, rank, money, merit, tc, here) -> tuple[float, int | None]:
        if tc is None:
            return 0.0, None
        gain = self._expected_work(rank)
        after = self._progress(rank, money, merit + gain)
        score = after - here
        if merit + gain >= tc:
            score = (1.0 - here) + self.w.promotion_bonus
        return score, None

    def _score_corrupt(self, public, private, rank, money, merit, mc, here) -> tuple[float, int | None]:
        if mc is None:
            return 0.0, None
        gain = self._expected_corrupt(rank)
        after = self._progress(rank, money + gain, merit)
        score = after - here
        promotes = money + gain >= mc
        if promotes:
            score = (1.0 - here) + self.w.promotion_bonus

        # 风险：本轮贪的这一笔在 rank>=1 时几乎必然踩到重大贪腐线
        p_caught = self._report_pressure(public, private)
        major = gain >= self.cfg.major_corruption_threshold
        if major:
            loss = rank * 1.0 + self._progress(rank, money + gain, 0)  # 官职 + 全部家当
        else:
            loss = self._progress(rank, gain, 0)
        if promotes:
            loss += self.w.promotion_bonus  # 到手的晋升也飞了
        return score - self.w.caught_dread * p_caught * loss, None

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
        tc = cfg.merit_cost(t_rank)
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
                t_rank, self._progress(t_rank, model.money_est, t_merit)
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

        # --- 对方的损失（好处全桌分，要打稀释折扣）---
        hush_hit = locals().get("hush_hit", 0.0)
        threat = self._threat(t_rank, self._progress(t_rank, model.money_est, t_merit))
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
        deny += p_block * block_value * threat

        if cfg.attack_resets_tenure and cfg.tenure_required > 0:
            deny += (opp["tenure"] / cfg.tenure_required) * threat
        deny += hush_hit * threat

        # 整体再封一次顶：一次攻击最多值"让他少升一级"
        deny = min(deny, (1.0 + self.w.promotion_bonus) * threat)

        # 但拦住"下一步就夺冠"的人是例外：那不是少升一级，是阻止整局结束。
        # 这份好处也不该按人数稀释——他赢了，桌上每个人都输。
        if self._about_to_win(opp, model):
            return (
                self.w.interfere_attack
                * (self_value + deny * self.w.endgame_block_weight),
                opp["id"],
            )
        return self.w.interfere_attack * (self_value + deny * share), opp["id"]

    def _score_report(self, public, private, opp) -> tuple[float, int | None]:
        """举报是一次对隐藏信息的下注：他这轮到底有没有经济问题。

        现在"有问题"包括两件事：贪了钱，或者拿钱买官。后者尤其关键——
        一个攒够钱、马上要买官上位的人，举报就是确定能拦住他的手段。
        """
        cfg = self.cfg
        pid = opp["id"]
        model = self.models.get(pid, OpponentModel())
        t_rank = opp["rank"]
        my_rank, my_money = private["rank"], private["money"]
        mc_t = cfg.money_cost(t_rank)
        tc_t = cfg.merit_cost(t_rank)

        # --- 他这轮贪污的概率 ---
        p_corrupt = 0.5 * model.corrupt_rate + 0.5 * (model.corrupt_rate * 2.0) * model.quiet_rate
        p_corrupt = min(0.95, p_corrupt)
        if mc_t is not None and model.money_est >= mc_t - self._expected_corrupt(t_rank):
            p_corrupt = min(0.9, p_corrupt * 1.6)  # 他就差这一笔就能用钱升官了
        if public.get("round", 1) - model.last_seen_corrupting <= 1:
            p_corrupt = min(0.95, p_corrupt * 1.5)  # 上一轮刚被财富广播点名
        merit_ready = tc_t is not None and opp["merit"] >= tc_t
        if merit_ready and not cfg.needs_both(t_rank):
            p_corrupt *= 0.6  # 他政绩已经够了，这轮八成在等着靠政绩升

        # --- 他这轮拿钱买官的概率（行贿也算经济问题）---
        p_bribe = 0.0
        needs_both = cfg.needs_both(t_rank)
        # 双条件台阶（省级 -> 主席）钱和政绩都得花，所以只要他准备升，就一定在行贿。
        # 其他台阶只有政绩不够、非掏钱不可时才算。
        money_is_required = needs_both or not merit_ready
        if cfg.report_catches_bribery and mc_t is not None and money_is_required:
            if model.money_est >= mc_t:
                p_bribe = 0.55
            elif model.money_est >= mc_t * 0.8:
                p_bribe = 0.25
            elif needs_both and merit_ready:
                # 政绩已经够了、就差钱——他正在攒，攒到就登顶。
                # 钱是暗的，估不准，但这一刀值得赌。
                p_bribe = 0.35
        p_hit = min(0.95, 1.0 - (1.0 - p_corrupt) * (1.0 - p_bribe))
        if p_hit <= 0:
            return 0.0, pid

        # --- 查实能抄到多少（只没收本轮那一笔 + 查获的行贿，还要按比例分）---
        haul = p_corrupt * self._expected_corrupt(t_rank)
        seized_bribe = p_bribe * (mc_t or 0)
        pool = (haul + seized_bribe) * float(cfg.report_reward_ratio)
        expected_split = 1.0 + 0.25 * max(0, len(public["players"]) - 2)
        my_cut = pool / expected_split

        mc = cfg.money_cost(my_rank)
        # 攒着的钱要按"还来不来得及花"打折；能当场换成一级官职的那部分不打折。
        cash_value = (my_cut / mc) * self._cash_horizon(public) if mc else 0.0
        if mc and my_money + my_cut >= mc:
            cash_value += self.w.promotion_bonus  # 这笔赃款直接把我送上去

        # --- 打击面：冻结他这一轮的晋升 + 记一次警告（攒满就降级）---
        t_prog = self._progress(t_rank, model.money_est, opp["merit"])
        threat = self._threat(t_rank, t_prog)
        setback = threat * 0.5
        # 查实必然冻结他本轮的晋升。他要是正打算升，这一下就是实打实拦住一级。
        about_to_promote = merit_ready or (mc_t is not None and model.money_est >= mc_t)
        if about_to_promote:
            setback += (1.0 + self.w.promotion_bonus) * threat
        # 离降级越近，这一次警告越值钱
        wmax = max(1, cfg.warnings_before_demotion)
        setback += (opp.get("warnings", 0) + 1) / wmax * threat * 0.5

        # 把人按住基本是公共品，赃款才是我的；但他越接近登顶，这份好处越是我自己的
        setback *= self._share(public, threat)

        # 拦住"下一步就夺冠"的人是例外：那不是少升一级，是阻止整局结束，
        # 这份好处也不该按人数稀释——他赢了，桌上每个人都输。
        if self._about_to_win(opp, model):
            return (
                self.w.interfere_report
                * p_hit
                * (cash_value + threat * self.w.endgame_block_weight),
                pid,
            )
        return self.w.interfere_report * p_hit * (cash_value + setback), pid


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

    def get(self, player_id: int) -> SmartAgent:
        if player_id not in self.agents:
            self.agents[player_id] = SmartAgent(
                player_id,
                cfg=self.cfg,
                weights=self.weights,
                rng=self.rng,
                allow_attack=self.allow_attack,
                allow_report=self.allow_report,
                allow_corrupt=self.allow_corrupt,
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
    agent.observe(public)

    if any(Card(d["card"]).needs_target for d in private["hand"]):
        return False  # 手上已经有举报或攻击了
    if not private.get("redraw_affordable"):
        return False

    for opp in public["players"]:
        if opp["id"] == player_id:
            continue
        model = agent.models.get(opp["id"], OpponentModel())
        if agent._about_to_win(opp, model):
            return True
        # 还差一点、但已经站在主席门口的人，也值得砸钱换一手牌去拦。
        # 等到估计值真的过线往往已经晚了——他那一轮就登顶了。
        if opp["rank"] == agent.cfg.president_rank - 1:
            prog = agent._progress(opp["rank"], model.money_est, opp["merit"])
            if prog >= 0.75:
                return True
    return False


def choose(game, player_id: int, pool: AgentPool) -> list[dict]:
    """用 game 的公开/私密 payload 驱动 AI —— 和浏览器拿到的完全一样。

    返回 game.select_actions() 能直接吃的格式。
    """
    agent = pool.get(player_id)
    picks = agent.decide(game.public_state(), game.private_state(player_id))
    return [{"action": c, "target": t} for c, t in picks]
