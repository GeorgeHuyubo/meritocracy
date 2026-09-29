"""Meritocracy 全部平衡参数。

设计约束：游戏逻辑代码里不允许出现魔法数字，所有数值都从这里读。
模拟器调平衡时只需要改这个文件（或在运行时替换 Config 实例）。

所有倍率一律使用 fractions.Fraction，禁止浮点数，避免 0.1+0.2 这类误差。
只有在算出"最终收益"时才做一次向下取整。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

# --------------------------------------------------------------------------
# 官职
# --------------------------------------------------------------------------

RANK_NAMES: list[str] = [
    "基层公务员",  # 0
    "县级干部",  # 1
    "市级干部",  # 2
    "省级干部",  # 3
    "国家主席",  # 4
]

PRESIDENT_RANK: int = 4  # 到达即获胜
BASE_RANK: int = 0

# 收益倍率：基层 x1 / 县级 x1.5 / 市级 x2 / 省级 x2.5
# 国家主席的倍率只是占位（到达即结束游戏，不会再结算）。
RANK_MULTIPLIERS: list[Fraction] = [
    Fraction(1),
    Fraction(3, 2),
    Fraction(2),
    Fraction(5, 2),
    Fraction(3),
]

# WORK 和 CORRUPT 默认共用上面的倍率。拆开可以做"越往上越必须靠钱"这类调整，
# 例如把 WORK 固定成 x1，逼高位玩家不得不下场捞钱。None = 沿用 RANK_MULTIPLIERS。
WORK_RANK_MULTIPLIERS: list[Fraction] | None = None
MONEY_RANK_MULTIPLIERS: list[Fraction] | None = None

# --------------------------------------------------------------------------
# 晋升
# --------------------------------------------------------------------------

# 索引 = 当前官职，值 = 升到下一级所需的资源
# 基层->县级, 县级->市级, 市级->省级, 省级->国家主席
#
# 一轮出两张牌之后每轮产出大约翻倍，但晋升本身也要占掉一个行动位，
# 实际净产出约 1.5~1.6 倍，所以门槛按 x1.6 上调，维持原来的轮数感觉。
# 门槛跟着牌面期望走：WORK 从 10 降到 8，政绩门槛也要 x0.8，
# 否则政绩路线会平白慢 25%；CORRUPT 从 10 提到 12，金钱门槛相应 x1.2。
# 不这么配的话实测主席率会从 80% 崩到 13%（12 轮全打满还没人登顶）。
# 又按 x0.65 整体下调过一次：工龄从 3 轮改成 4 轮之后，纯熬资历只能到省级
# （12 轮 / 4 = 3 级），必须至少靠资源升一次，门槛太高就会出现"12 轮打满没人登顶"。
# 实测 x0.65 时主席率 67%，两套胜利条件都活着。
# 最后一级（省级 -> 主席）是双条件台阶，两样都要够，所以单项门槛下调到约 0.65。
# 不下调的话主席率只有 37%，大半对局都打满 12 轮比家底，太闷。
# 金钱门槛直接锚在"贪一笔能拿多少"上：期望 15 点 × 官职倍率 = 15 / 22 / 30 / 37。
# 规则因此变得好记：**贪一笔，正好够升一级**（再加上当轮工资还有富余）。
#
# 之前的 [16,30,49,51] 是按别的思路定的，门槛按 ×1/1.9/3.1/3.2 增长，
# 而收益只按官职倍率 ×1/1.5/2/2.5 增长——越往上越够不着，市级时一笔只够 61%。
PROMOTION_MONEY_COSTS: list[int] = [15, 22, 30, 37]
PROMOTION_MERIT_COSTS: list[int] = [15, 27, 43, 43]

# 晋升必须打出晋升卡（工龄晋升除外）。
# False = 回到规则书原版：资源够了自动晋升。
PROMOTION_REQUIRES_CARD: bool = True

# 每轮自动到账的**合法工资**，按官职发，不用打牌、不占行动位。
# 这笔钱是干净的：不算"本轮贪污额"，不会被举报查实、不会被攻击起获、
# 也不进财富广播。作用是给金钱路线一条不用冒险的底薪，
# 同时让"升官"本身有直接的经济回报。
# 金额由官职推出来，所以谁拿多少是全场都能算出来的公开信息。
RANK_SALARY: list[int] = [1, 2, 4, 6, 0]

# 哪些台阶要求**同时**满足金钱和政绩（索引 = 当前官职）。
# 省级 -> 国家主席 这一步设成 True：到了这个位置，光有政绩或光有钱都不够。
PROMOTION_REQUIRES_BOTH: list[bool] = [False, False, False, True]

# 工龄晋升能不能一路熬到国家主席。
# 关掉之后，最后一步必须靠"政绩 + 金钱"双条件挣来，熬资历最多熬到省级。
TENURE_CAN_REACH_PRESIDENT: bool = False

# 晋升后超额资源的衰减：new = ceil((old - cost) / divisor)。divisor = 1 表示不衰减。
#
# 规则书第 9、10 节对金钱和政绩用的是同一个 /5，但那是按"门槛 10~48、每笔约 10"
# 设计的——那时你通常刚好够线，余额本来就没几点。现在贪一笔 15~37、还有工资，
# 经常大幅超线，砍掉的绝对量大得多；最后一级又要双条件，两样会被同时砍一刀。
#
# 所以拆开：政绩是"已经兑现掉的功劳"，清空合理；
# 金钱是实打实的家当，花掉门槛之后剩下的应该留住。
OVERFLOW_DIVISOR: int = 5  # 没单独指定时的默认值（规则书原版两样都用它）
MERIT_OVERFLOW_DIVISOR: int = 5
MONEY_OVERFLOW_DIVISOR: int = 1  # 1 = 只扣门槛，余额全留

# 升一级之后，政绩一律做 /MERIT_OVERFLOW_DIVISOR 衰减——**不管你是怎么升上去的**。
# 关掉的话就退回老规矩：只有实际花掉的那份资源才衰减，于是"攒政绩 + 花钱升职"
# 可以跨级把政绩原封不动地存下来，等于白拿一级，贿赂升职反而严格优于政绩升职。
# 金钱的除数默认是 1（不衰减），所以这条开关实际只对政绩起作用。
PROMOTION_ALWAYS_DECAYS_MERIT: bool = True

# 连续待满多少轮自动晋升一级
TENURE_REQUIRED: int = 4

# 重新抽一手牌的**底价**（按当前官职定价，也就是本轮第一次换牌的价钱）。
# 换掉了"每升一级白送一次"：那个机制和钱没关系，等于免费重开，
# 而且攒着不用毫无损失、AI 还从来不用，等于白送真人一层优势。
#
# 定价要让"前期好换、后期难换"在**两个口径上同时成立**：
#   对每轮工资 [1,2,4,6]   -> 1.0 / 1.5 / 1.5 / 2.0 轮
#   对升职门槛 [15,22,30,37] -> 7% / 14% / 20% / 32%
# 基层一轮工资正好换一次（手气差不至于毫无办法），省级一次要攒两轮、
# 还等于三分之一次升职。
REDRAW_COSTS: list[int] = [1, 3, 6, 12, 0]

# 同一轮里连续换牌，每换一次价格乘这个数：1 -> 2 -> 4 -> 8 ...
# 没有这条的话，有钱人可以在一轮里反复重抽直到摸到想要的牌。
REDRAW_COST_GROWTH: int = 2

# 同一轮有多人登上国家主席时怎么办。规则书没有定义，而这种情况相当常见
# （高水平对局里所有人节奏接近，经常一起冲线）。
# True = 按"金钱 > 政绩"分出唯一胜者，实测能解决其中约 94%；
# False = 全部并列冠军。
PRESIDENT_TIEBREAK: bool = True

# --------------------------------------------------------------------------
# 行动卡
# --------------------------------------------------------------------------

HAND_SIZE: int = 6
# 每轮从手牌里选几张打出去。1 = 最初的"5 选 1"。
PICKS_PER_ROUND: int = 2

# 发牌时每种卡的权重（有放回抽样）。
# 一轮出两张牌之后，干扰行动的机会成本从"一整个回合"降到"半个回合"，
# 所以干扰牌必须变稀有，否则会出现"全场互相攻击、没人建设"的退化局面。
# 实测干扰占比从 22% 降到 12%，主席率从 35% 回到 75%。
#
# 当前权重下（总 16）：
#   摸到 WORK 73.7%（2 个行动位，够用）
#   摸到"政绩升职或通用升职"58.6%，摸到任意晋升卡 72.3%
#   摸到 REPORT 或 ATTACK 各 27.5%
# 再加上工龄兜底（连续 3 轮没晋升自动升），不会出现彻底卡死。
CARD_DEAL_DISTRIBUTION: dict[str, int] = {
    "WORK": 4,
    "CORRUPT": 3,
    "GRAFT": 2,
    "REPORT": 2,
    "ATTACK": 2,
    "PROMOTE_MERIT": 2,
    "PROMOTE_MONEY": 2,
    "PROMOTE_ANY": 2,
}

# (数值, 权重)。期望值必须 = 10。
# WORK 期望 6，CORRUPT 期望 15：政绩路线慢而安全（只怕攻击），
# 金钱路线快一倍多但见光死（怕举报，也怕被攻击撞上）。
WORK_CARD_DISTRIBUTION: list[tuple[int, int]] = [
    (4, 1),
    (5, 2),
    (6, 2),
    (7, 2),
    (8, 1),
]
# 贪污的期望点数 = 埋头工作的 **3 倍**（6 -> 18）。
# 高风险高收益：一张贪污牌顶一次半升职，但 31% 的概率被查实——
# 赃款全没收、工龄清零、记一次严重警告，攒两次就降级。
CORRUPT_CARD_DISTRIBUTION: list[tuple[int, int]] = [
    (16, 1),
    (17, 2),
    (18, 2),
    (19, 2),
    (20, 1),
]

# 以权谋私：钱比纯贪污少（期望 8），但同一张牌顺带带政绩（按 GRAFT_MERIT_RATIO 折算）。
# 赚到的钱照样算"本轮贪污额"，所以一样会被举报查实、被攻击当场起获。
# 存在的意义：让金钱路线不必全程裸奔，可以一边攒钱一边留一点政绩当保底。
GRAFT_CARD_DISTRIBUTION: list[tuple[int, int]] = [
    (8, 1),
    (9, 2),
    (10, 2),
    (11, 2),
    (12, 1),
]
# 以权谋私顺带的那点政绩，要**严格少于**埋头工作 —— 捞钱的顺手之作
# 不该比老老实实干活还长脸。
# 牌面 8~12，所以 1/4 让政绩落在 2~3，完全低于 WORK 的 4~8，两段不重叠。
# （1/2 的时候是 4~6：期望虽然比 WORK 低，但一张好的以权谋私能压过一张差的 WORK，
#   而且还白送 10 块钱 —— 看着就不对。）
GRAFT_MERIT_RATIO: Fraction = Fraction(1, 4)

# --------------------------------------------------------------------------
# 政治攻击 / 匿名举报
# --------------------------------------------------------------------------

# ATTACK 的作用模式：
#   "merit_penalty" —— 规则书原版：目标本轮没打 WORK 就扣固定政绩，打了 WORK 则免疫。
#                      问题是它纯利他（我付一整个回合，好处全桌分），实测任何数值下都不划算。
#   "steal_work"   —— 当前默认，抢功（按本轮产出）：没收目标本轮产出的一半，
#                      所有抢功的人平分这一半，目标保留另一半。
#                      这是整个「鹬蚌相争，渔翁得利」结构的支点：
#                          打正在生产的人 -> 抢得到东西 -> 出手的人自己也前进
#                          打正在干扰别人的人 -> 他没产出 -> **白打一张牌**
#                      于是「#1 埋头干、#2 去抢他」= #2 反超，
#                      而「#1 和 #2 互抢」= 两人都没产出、没抢到，
#                      埋头建设的 #3 渔翁得利。
#                      抢上位者天然更值：省级一张 WORK 产 15，基层只产 6。
#   "steal_merit"  —— 抢功（按政绩存量）：抢走目标**当前政绩**的一部分，归攻击者。
#                      和"举报没收赃款归举报人"对称，并且天然形成克制关系：
#                          干实事的人政绩堆得高 -> 攻击命中，收益巨大
#                          捞钱的人政绩几乎为零 -> 攻击自动扑空
#                      政绩是公开数据，所以这是一个"算得准"的读牌；
#                      举报打的是隐藏的金钱，是一个"要猜"的赌。
#   "negative_sum" —— 负和：目标损失一大笔政绩，攻击者只拿回其中一部分，差额凭空蒸发。
#                      两人互攻则两败俱伤，谁都不得利 —— 鹬蚌相争，渔翁得利。
#   "denial"       —— 纯破坏型：攻击者拿不到政绩，只负责毁。
#                      · 阻断政绩晋升（含通用升职走政绩那条）
#                      · 如果真的拦下了一次政绩晋升 -> 目标政绩**清零**
#                      · 目标本轮没打 WORK -> 政绩额外 -ATTACK_MERIT_PENALTY
#                      · 目标本轮贪污了 -> 本轮赃款没收，**归攻击者**
#                      · 不动工龄
#                      于是攻击是一场"赌他这轮要兑现"的下注：赌对了才有核弹效果，
#                      赌错了顶多蹭掉一点政绩；唯一的正收益来自撞上对方在贪污。
ATTACK_MODE: str = "steal_work"

# 目标本轮没干活（一点政绩都没产）时扣他多少政绩。
# 这是**牌面点数**，会再走目标官职的 WORK 倍率 —— 官越大扣越多
# （基层 -4 / 县级 -6 / 市级 -8 / 省级 -10）。
# steal_work 模式：无功可抢时的处罚。攻击者拿不到这一份，
#   所以"互相攻击"仍然是两败俱伤，渔翁得利那条不受影响。
# merit_penalty / denial 模式：见上面的模式说明。
ATTACK_MERIT_PENALTY: int = 4

# 规则书第 12 节写了"WORK 玩家不会受到这个处罚"。仅 merit_penalty 模式生效。
# 这条豁免让纯政绩路线对攻击完全免疫，是个很强的平衡杠杆，所以做成开关。
ATTACK_SPARES_WORKERS: bool = True

# steal_work 模式：截走目标本轮政绩产出的比例。
# steal_merit 模式：抢走目标当前政绩存量的比例。多人攻击同一目标则平分这一份。
#
# 1/2 = 「被抢的人保留一半，剩下一半所有抢功的人分」。
# 这个数字直接决定三角动态成不成立：太低则出手不划算（退回废牌），
# 太高则抢功变成必选牌、没人敢当第一。用 `analysis.py --section triangle` 标定。
ATTACK_STEAL_FRACTION: Fraction = Fraction(1, 2)

# ---- negative_sum 模式 ----
# 量纲是「回合当量」：1 = 一个回合埋头工作的产出（按对应官职的倍率折算）。
# 用回合当量而不是存量比例，是因为攻击的真实成本就是"我这一回合什么都没产出"。
#   目标 -2 个回合，攻击者 +1 个回合 => 每次交手全场净蒸发 1 个回合。
ATTACK_DAMAGE_TURNS: Fraction = Fraction(2)
# 攻击者拿回伤害额的这个比例（按实际打掉的量折算，打空气就拿不到）。
# 1/2 配合 -2 就是"对方 -2、我 +1"；设成 0 就是纯粹的同归于尽。
ATTACK_GAIN_RATIO: Fraction = Fraction(1, 2)
# 两人互相攻击时：双方照样吃满伤害，但谁都拿不到好处。
ATTACK_MUTUAL_CANCELS_GAIN: bool = True
# 抢来的政绩是否按攻击者自己的官职折算。开着能让两边的政绩变动数字对不上、
# 不容易反推出谁动的手，但会让落后的攻击者亏得更多（官小 = 换算打折）。
ATTACK_STEAL_SCALED_BY_RANK: bool = False

# 抢功时，目标每比自己高一级，额外多抢这么多（0 = 关）。
# 默认不开：抢上位者已经天然更值（高官产出的绝对值就大，省级 15 vs 基层 6），
# 再加显式加成会破坏「正好一半」这个干净的说法。标定时不够再开。
ATTACK_STEAL_RANK_BONUS: Fraction = Fraction(0)

# 政治攻击是否公开攻击者身份。
# 开着 = 明攻击：收益高，但被打的人下一轮知道该报复谁。
# 匿名举报保持暗箭，两张牌形成明/暗对照。
ATTACK_ANNOUNCES_ATTACKER: bool = True

# 拦下一次政绩晋升时，是否把目标政绩清零。
# 关掉 = 只是「暂缓升职」：这一轮升不上去，但政绩一点不掉。
# 开着的话这是全游戏最大的一次性破坏（一刀能削 43 点）而且攻击者一分不拿，
# 纯利他 —— 攻击的收益现在全部来自抢功那一半，破坏这块降到最小。
ATTACK_WIPES_MERIT_ON_BLOCK: bool = False
# denial 模式：撞上目标本轮贪污时怎么处理。
#   "confiscate_to_attacker" —— 没收赃款并归攻击者（和匿名举报重复，实测把贪污路线打死了）
#   "merit_to_attacker"      —— 当前默认：钱不动，攻击者自己拿到少量政绩
#                               （"抓贪腐立功"），把"抄家"留给匿名举报做专属
#   "none"                   —— 撞上贪污什么也不发生
ATTACK_ON_CORRUPTION: str = "merit_to_attacker"
# merit_to_attacker 模式下，攻击者拿到的政绩 = 目标本轮贪污额 x 这个比例，
# 再走攻击者自己官职的 WORK 倍率。
ATTACK_CORRUPTION_MERIT_RATIO: Fraction = Fraction(1, 4)

# 被攻击时如果本轮正在贪，要拿出本轮赃款的这个比例去"上下打点、压事"。
# 这笔钱是**花掉**的，不进攻击者口袋（他拿的是政绩记功）。
# 连带效果：净落袋变少，财富广播也就没那么风光了——花钱消灾，社会影响确实小了。
# 但"你到底贪没贪"仍然按毛收入判定，该被举报还是会被举报。
ATTACK_HUSH_MONEY_RATIO: Fraction = Fraction(1, 2)

# 攻击命中时是否把目标的工龄清零。denial 模式默认不动工龄。
# 举报的真正杀伤是"降一级"（永久损失），攻击原本只能削掉一点会被晋升清空的存量，
# 两者弹头量级差太多。工龄晋升占了全部晋升的约 45%，而且目前没有任何机制能碰它，
# 所以"搅黄资历"是给攻击配的、和降级对等但不重复的弹头。
ATTACK_RESETS_TENURE: bool = False

# 本轮"最终"贪污金额 >= 该值 = 重大贪腐（举报时打回基层而不是降一级）。
#
# 实测：贪污路线没人走，瓶颈**不是钱少而是"被抓就打回基层"太致命**——
# 把牌面收益从 12 一路加到 18 完全没用（出牌率反而降），把这条线从 20 提到 34
# 却能把贪污出牌率从 25% 拉到 34%。
#
# 34 这个值形成的规则是："适可而止就没事，贪得太狠才是大案"：
#   贪一笔 = 基层 12 / 县级 18 / 市级 24 / 省级 30  -> 全都只算小额
#   一轮贪两笔 = 基层 24（仍小额）/ 县级 36 / 市级 48 / 省级 60 -> 县级以上就是重大
MAJOR_CORRUPTION_THRESHOLD: int = 34

# 举报查实的后果改成"严重警告"制：
#   * 本轮贪污所得全部没收
#   * 拿钱买的晋升作废，而且**钱照样没了**（送出去的礼收不回来）
#   * 记一次严重警告，工龄清零
#   * 警告累计到 WARNINGS_BEFORE_DEMOTION 次 -> 降一级、工龄清零、警告清空
# 好处是惩罚不再是"一次被抓就打回基层"那种断崖，而是可以被玩家计入预算的分期账单。
WARNINGS_BEFORE_DEMOTION: int = 2

# 「拿钱买官」本身算不算一条可举报的罪名。
#
# True  = 赚钱和花钱两头都要过举报这一关（各约 31%），金钱路线被双重收税。
#         曾经实测"贪污消融值 +7.05（不用反而强很多）"，但那是用**有偏的**
#         消融方法测的（强行禁半桌）—— 贪污是"人多才安全"的牌，禁半桌会系统性
#         低估它。按真实均衡点（只禁 1 席）重测，三张牌都落在 0 附近、
#         区间跨 0，分不出差别。这条结论作废，要改这个开关得先重做实验。
# False = 举报只针对贪污；但一旦查实，这一轮的贿赂升职照样作废、钱也要不回来。
#         那一档当时测出 +0.67，同样出自有偏方法，一并作废。
# 两者的玩法差别就是"金钱路线有一道关还是两道关"。
REPORT_CATCHES_BRIBERY: bool = True

# 一次"重大贪腐"（本轮贪污额 >= MAJOR_CORRUPTION_THRESHOLD）记几次警告。
# 默认 1 = 和普通查实一样，大案不再有额外后果。
# 设成 2 就恢复"贪一大笔被抓当场降级"。
MAJOR_CORRUPTION_WARNINGS: int = 1

# 多个玩家举报同一个人时，是否叠加降级。默认不叠加（"中央反腐大筛查"事件
# 会给所有人加一次举报，叠加会导致惩罚过重）。
MULTIPLE_REPORTS_STACK: bool = False

# 举报查实的玩家本轮不能晋升（贪污那一轮被举报，赃款已经被没收，自然升不上去）。
DEMOTED_CANNOT_PROMOTE_SAME_ROUND: bool = True

# ---- 赃款没收与分赃 ----
# 举报查实后，被举报人的赃款没收并交给举报人。
REPORT_REWARD_ENABLED: bool = True
# 小额贪污：只没收本轮那一笔贪污金额，存款保留。
REPORT_REWARD_MINOR_TAKES_ALL: bool = False
# 重大贪腐：本轮贪污额 + 全部存款一起抄没（等价于把 money 全部转给举报人）。
REPORT_REWARD_MAJOR_TAKES_ALL: bool = True
# 多人举报同一个人时平分赃款，向下取整，除不尽的零头充公。
# "中央反腐大筛查"事件产生的那一份举报没有举报人，不参与分配（赃款直接充公）。
REPORT_REWARD_SPLIT_EVENLY: bool = True

# 举报人实际拿到手的比例：查实后没收的赃款里，只有这一份归举报人，其余充公。
# 举报同时还会**冻结对方本轮的晋升**（贿赂升职也挡得住），光靠降级+没收
# 这张牌就已经很强了，所以拿钱的部分要收一收。
REPORT_REWARD_RATIO: Fraction = Fraction(1, 2)

# --------------------------------------------------------------------------
# 全局事件
# --------------------------------------------------------------------------

# 事件效果支持的字段（缺省即不生效）：
#   work_bonus        : int      WORK 基础政绩 +N（在官职倍率之前）
#   work_multiplier   : Fraction WORK 基础政绩 xN（在官职倍率之前）
#   money_bonus       : int      CORRUPT 基础金钱 +N（在官职倍率之前）
#   money_multiplier  : Fraction CORRUPT 基础金钱 xN（在官职倍率之前）
#   attack_disabled   : bool     本轮所有 ATTACK 无效
#   mass_report       : bool     本轮所有玩家额外视为被匿名举报一次
#   storm_report      : bool     只查办本轮贪污额排前 EVENT_STORM_FRACTION 的人
EVENT_DEFINITIONS: list[dict[str, Any]] = [
    {
        "id": "CALM",
        "name": "风平浪静",
        "description": "官场波澜不惊，本轮没有特殊情况。",
        # 权重从 30 降到 12：原来近三分之一的轮次揭牌是"什么也没发生"，
        # 揭事件这个环节大部分时候没有反馈。留一点是为了别每轮都天下大乱。
        "weight": 12,
        "effects": {},
    },
    {
        "id": "ANTI_CORRUPTION",
        "name": "反腐风暴",
        "description": "中央下来查了，本轮贪得最多的那 1/3 人被就地查办。",
        # 原来是"所有人各记一次举报"，实测存在感最高（3.71）却完全不造成反转
        # （领先易手 -0.5%）—— 因为它均匀打击所有贪污者，不改变相对排序。
        # 改成只打榜单前 1/3，就变成了一次真正的洗牌。
        "weight": 10,
        "effects": {"storm_report": True},
    },
    {
        "id": "BOOM",
        "name": "经济一片大好",
        "description": "经济形势喜人，本轮所有金钱收益 x2。",
        "weight": 15,
        "effects": {"money_multiplier": Fraction(2)},
    },
    {
        "id": "RECESSION",
        "name": "经济下行",
        "description": "财政吃紧，本轮所有金钱收益 x0.5（向下取整）。",
        "weight": 15,
        "effects": {"money_multiplier": Fraction(1, 2)},
    },
    {
        "id": "KEY_PROJECT",
        "name": "重点项目",
        "description": "上级下达重点项目，本轮 WORK 基础政绩 +4。",
        # +1 实测完全是噪音：全桌 6 人总共只多出 8 点政绩，人均 1.3 点，
        # 而晋升门槛是 15~66，存在感评分 0.16、领先易手只比基准高 1.0%。
        # 提到 +4 之后一次 WORK 从 8 变 12（+50%），才算一个值得等的轮次。
        "weight": 15,
        "effects": {"work_bonus": 4},
    },
    {
        # 规则书第 17 节要求过这张牌，定义保留着方便随时恢复（权重 0 = 永远不会被抽到）。
        # 下架原因：实测它是全场唯一显著**降低**反转率的事件（领先易手 -6.2%），
        # 而且只保护走政绩路线的领先者——本轮谁也拦不住他兑现晋升，
        # 走金钱路线的人却照样怕举报。属于无条件的反追赶机制。
        "id": "STABLE",
        "name": "政治环境稳定",
        "description": "风气清明，本轮所有政治攻击无效。",
        "weight": 0,
        "effects": {"attack_disabled": True},
    },
]

# --------------------------------------------------------------------------
# 财富广播
# --------------------------------------------------------------------------

# (最小金额, 最大金额或 None, 单人模板, 多人模板)
WEALTH_BROADCAST_TIERS: list[tuple[int, int | None, str, str]] = [
    (1, 2, "{names} 最近生活条件改善了。", "{names} 最近都生活条件改善了。"),
    (3, 4, "{names} 换了辆新车。", "{names} 最近都换了辆新车。"),
    (5, 7, "{names} 开上豪车了。", "{names} 最近都开上豪车了。"),
    (8, None, "{names} 住上洋房了。", "{names} 最近都住上洋房了。"),
]
WEALTH_BROADCAST_NAME_JOINER: str = " 和 "

# 「反腐风暴」查办多大比例的人：按总人数算，向上取整。
# 6 人局 -> ceil(6/3) = 2 人；并列卡在分界线上的一起算进去。
# 本轮没贪污的人不会被卷进来，实际查办人数不会超过当轮贪污的人数。
EVENT_STORM_FRACTION: Fraction = Fraction(1, 3)

# --------------------------------------------------------------------------
# 对局
# --------------------------------------------------------------------------

MAX_ROUNDS: int = 12

# 胜负判定的完整顺序是：
#   1. 当上国家主席 —— 即时获胜，游戏立刻结束（不走下面的比较）
#   2. 打满 MAX_ROUNDS 还没人登顶 -> 按 FINAL_RANKING_KEYS 排名次
#
# 规则书第 2 节原本是 金钱 > 政绩 > 官职，现在把官职提到政绩前面：
# 官职是永久的、花过代价的，政绩只是还没兑现的存量，让它当最后一道分隔更合理。
FINAL_RANKING_KEYS: tuple[str, ...] = ("money", "rank", "merit")
MIN_PLAYERS: int = 2
MAX_PLAYERS: int = 6

# Web 层：抽到事件后停留多久再结算（秒），给玩家看清事件的时间。
REVEAL_EVENT_SECONDS: float = 2.5


@dataclass(frozen=True)
class Config:
    """把上面所有模块级常量打包成一个可替换的对象。

    game.py / rules.py 只通过 Config 实例读数值，方便模拟器做参数扫描：

        cfg = dataclasses.replace(DEFAULT_CONFIG, max_rounds=20)
    """

    rank_names: list[str] = field(default_factory=lambda: list(RANK_NAMES))
    president_rank: int = PRESIDENT_RANK
    base_rank: int = BASE_RANK
    rank_multipliers: list[Fraction] = field(default_factory=lambda: list(RANK_MULTIPLIERS))
    work_rank_multipliers: list[Fraction] | None = None
    money_rank_multipliers: list[Fraction] | None = None

    promotion_money_costs: list[int] = field(default_factory=lambda: list(PROMOTION_MONEY_COSTS))
    promotion_merit_costs: list[int] = field(default_factory=lambda: list(PROMOTION_MERIT_COSTS))
    rank_salary: list[int] = field(default_factory=lambda: list(RANK_SALARY))
    promotion_requires_both: list[bool] = field(
        default_factory=lambda: list(PROMOTION_REQUIRES_BOTH)
    )
    tenure_can_reach_president: bool = TENURE_CAN_REACH_PRESIDENT
    overflow_divisor: int = OVERFLOW_DIVISOR
    merit_overflow_divisor: int = MERIT_OVERFLOW_DIVISOR
    money_overflow_divisor: int = MONEY_OVERFLOW_DIVISOR
    promotion_always_decays_merit: bool = PROMOTION_ALWAYS_DECAYS_MERIT
    promotion_requires_card: bool = PROMOTION_REQUIRES_CARD
    tenure_required: int = TENURE_REQUIRED
    redraw_costs: list[int] = field(default_factory=lambda: list(REDRAW_COSTS))
    redraw_cost_growth: int = REDRAW_COST_GROWTH
    president_tiebreak: bool = PRESIDENT_TIEBREAK

    hand_size: int = HAND_SIZE
    picks_per_round: int = PICKS_PER_ROUND
    card_deal_distribution: dict[str, int] = field(
        default_factory=lambda: dict(CARD_DEAL_DISTRIBUTION)
    )
    work_card_distribution: list[tuple[int, int]] = field(
        default_factory=lambda: list(WORK_CARD_DISTRIBUTION)
    )
    corrupt_card_distribution: list[tuple[int, int]] = field(
        default_factory=lambda: list(CORRUPT_CARD_DISTRIBUTION)
    )
    graft_card_distribution: list[tuple[int, int]] = field(
        default_factory=lambda: list(GRAFT_CARD_DISTRIBUTION)
    )
    graft_merit_ratio: Fraction = GRAFT_MERIT_RATIO

    attack_mode: str = ATTACK_MODE
    attack_merit_penalty: int = ATTACK_MERIT_PENALTY
    attack_spares_workers: bool = ATTACK_SPARES_WORKERS
    attack_steal_fraction: Fraction = ATTACK_STEAL_FRACTION
    attack_steal_scaled_by_rank: bool = ATTACK_STEAL_SCALED_BY_RANK
    attack_steal_rank_bonus: Fraction = ATTACK_STEAL_RANK_BONUS
    attack_announces_attacker: bool = ATTACK_ANNOUNCES_ATTACKER
    attack_resets_tenure: bool = ATTACK_RESETS_TENURE
    attack_wipes_merit_on_block: bool = ATTACK_WIPES_MERIT_ON_BLOCK
    attack_on_corruption: str = ATTACK_ON_CORRUPTION
    attack_corruption_merit_ratio: Fraction = ATTACK_CORRUPTION_MERIT_RATIO
    attack_hush_money_ratio: Fraction = ATTACK_HUSH_MONEY_RATIO
    attack_damage_turns: Fraction = ATTACK_DAMAGE_TURNS
    attack_gain_ratio: Fraction = ATTACK_GAIN_RATIO
    attack_mutual_cancels_gain: bool = ATTACK_MUTUAL_CANCELS_GAIN
    major_corruption_threshold: int = MAJOR_CORRUPTION_THRESHOLD
    warnings_before_demotion: int = WARNINGS_BEFORE_DEMOTION
    report_catches_bribery: bool = REPORT_CATCHES_BRIBERY
    major_corruption_warnings: int = MAJOR_CORRUPTION_WARNINGS
    multiple_reports_stack: bool = MULTIPLE_REPORTS_STACK
    demoted_cannot_promote_same_round: bool = DEMOTED_CANNOT_PROMOTE_SAME_ROUND
    report_reward_enabled: bool = REPORT_REWARD_ENABLED
    report_reward_minor_takes_all: bool = REPORT_REWARD_MINOR_TAKES_ALL
    report_reward_major_takes_all: bool = REPORT_REWARD_MAJOR_TAKES_ALL
    report_reward_split_evenly: bool = REPORT_REWARD_SPLIT_EVENLY
    report_reward_ratio: Fraction = REPORT_REWARD_RATIO

    event_definitions: list[dict[str, Any]] = field(
        default_factory=lambda: [dict(e) for e in EVENT_DEFINITIONS]
    )
    wealth_broadcast_tiers: list[tuple[int, int | None, str, str]] = field(
        default_factory=lambda: list(WEALTH_BROADCAST_TIERS)
    )
    wealth_broadcast_name_joiner: str = WEALTH_BROADCAST_NAME_JOINER
    event_storm_fraction: Fraction = EVENT_STORM_FRACTION

    max_rounds: int = MAX_ROUNDS
    final_ranking_keys: tuple[str, ...] = FINAL_RANKING_KEYS
    min_players: int = MIN_PLAYERS
    max_players: int = MAX_PLAYERS
    reveal_event_seconds: float = REVEAL_EVENT_SECONDS

    # ---- 便捷读取 ----

    def rank_name(self, rank: int) -> str:
        return self.rank_names[rank]

    def rank_multiplier(self, rank: int) -> Fraction:
        return self.rank_multipliers[rank]

    def work_multiplier(self, rank: int) -> Fraction:
        """WORK 的官职倍率。没单独配就沿用通用倍率。"""
        table = self.work_rank_multipliers or self.rank_multipliers
        return table[rank]

    def money_multiplier(self, rank: int) -> Fraction:
        """CORRUPT 的官职倍率。没单独配就沿用通用倍率。"""
        table = self.money_rank_multipliers or self.rank_multipliers
        return table[rank]

    def salary(self, rank: int) -> int:
        return self.rank_salary[rank]

    def redraw_cost(self, rank: int, used_this_round: int = 0) -> int:
        """这一级、本轮第 used_this_round+1 次换牌要花多少钱。

        底价按官职定，同一轮里每多换一次再翻 redraw_cost_growth 倍。
        返回 0 表示这一级不提供换牌。
        """
        if not (0 <= rank < len(self.redraw_costs)):
            return 0
        base = self.redraw_costs[rank]
        if base <= 0:
            return 0
        return base * self.redraw_cost_growth ** max(0, used_this_round)

    def needs_both(self, rank: int) -> bool:
        """这一级晋升是不是要求金钱和政绩同时达标。"""
        if rank >= self.president_rank:
            return False
        return bool(self.promotion_requires_both[rank])

    def money_cost(self, rank: int) -> int | None:
        """升到下一级需要的金钱；已是最高级返回 None。"""
        if rank >= self.president_rank:
            return None
        return self.promotion_money_costs[rank]

    def merit_cost(self, rank: int) -> int | None:
        if rank >= self.president_rank:
            return None
        return self.promotion_merit_costs[rank]


DEFAULT_CONFIG = Config()
