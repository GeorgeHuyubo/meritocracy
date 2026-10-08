"""rules.py 单元测试。

覆盖需求清单第 26 节的 1–16、19、20 条（17、18、21 在 test_game.py）。
"""

from __future__ import annotations

import dataclasses
import math
import random
import sys
from collections import Counter
from fractions import Fraction
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rules  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402
from helpers import CLASSIC_CARDS, ScriptedRng, corrupt_roll, work_roll  # noqa: E402
from models import (  # noqa: E402
    Action,
    GameEvent,
    Card,
    DemotionKind,
    Origin,
    PlayerState,
    PromotionKind,
)

# 本文件统一用固定牌面（8~12，期望 10）+ 固定重大贪腐线 8 结算，
# 这样脚本化的骰点和"小额/重大"的判定才稳定。真实配置的数值另有用例专门守。
CFG = dataclasses.replace(
    DEFAULT_CONFIG,
    work_card_distribution=list(CLASSIC_CARDS),
    corrupt_card_distribution=list(CLASSIC_CARDS),
    major_corruption_threshold=8,
    # 本文件里的用例都在断言精确金额，工资会把它们全打乱；
    # 工资本身另有 TestSalary 专门测。
    rank_salary=[0, 0, 0, 0, 0],
    # 举报跑腿费同理：分赃金额的用例按"赃款 x 比例"断言，跑腿费另有专门的用例
    report_reward_fee=0,
)
REAL_CFG = DEFAULT_CONFIG
CALM = rules.event_by_id("CALM")
KEY_PROJECT = rules.event_by_id("KEY_PROJECT")
# 规则书第 18 节的算例写的是"重点项目 +1"。真实配置已经调到 +4（+1 实测毫无存在感），
# 所以算例单独钉一份 +1 的事件来测"先加成再乘倍率"这个顺序。
KEY_PROJECT_SPEC = GameEvent(
    id="KEY_PROJECT", name="重点项目", description="", effects={"work_bonus": 1}
)
BOOM = rules.event_by_id("BOOM")
RECESSION = rules.event_by_id("RECESSION")
STABLE = rules.event_by_id("STABLE")
ANTI_CORRUPTION = rules.event_by_id("ANTI_CORRUPTION")

# 只有 4 点的贪污牌，用来构造"小额贪污"（默认牌组最低 8 点已经踩到重大贪腐线）
SMALL_CORRUPT_CFG = dataclasses.replace(CFG, corrupt_card_distribution=[(4, 1)])

# 规则书第 12 节写的是"固定扣政绩 + WORK 玩家免疫"那一版攻击。
# 默认配置现在用的是抢功模式（见 TestAttackStealWork），所以这些用例显式钉住原版。
LEGACY_ATTACK_CFG = dataclasses.replace(CFG, attack_mode="merit_penalty")

# 规则书原版：门槛 10/18/30/48 与 18/32/52/80，资源够了自动晋升，一轮出一张牌。
# 默认配置已经改成"5 选 2 + 晋升必须用卡 + 门槛 x1.6"，但规则书里的算例仍然要能跑通，
# 所以把原版单独钉成一份配置，专门用来测那些算例。
SPEC_CFG = dataclasses.replace(
    CFG,
    promotion_money_costs=[10, 18, 30, 48],
    promotion_merit_costs=[18, 32, 52, 80],
    promotion_requires_card=False,
    # 规则书原版：金钱晋升不动政绩。线上规则已经改成"怎么升都要 /5"，
    # 但这些例子是用来钉住规则书的，所以这里保持原样。
    promotion_always_decays_merit=False,
    picks_per_round=1,
    promotion_requires_both=[False, False, False, False],
    tenure_can_reach_president=True,
    money_overflow_divisor=5,  # 规则书第 9 节：金钱也要 /5（真实配置已改成不衰减）
)
# 零和抢功模式（按政绩存量）。默认已改成 steal_work + 比例 1/2，
# 这一档要把模式和比例都显式钉回来。
STEAL_ATTACK_CFG = dataclasses.replace(
    CFG, attack_mode="steal_merit", attack_resets_tenure=True,
    attack_steal_fraction=Fraction(1),
)
# 纯破坏模式。线上已改成"抢功 + 暂缓升职"，但 denial 仍是受支持的模式，
# 这一档把它连同"拦晋升清零政绩"一起钉住。
DENIAL_ATTACK_CFG = dataclasses.replace(
    CFG, attack_mode="denial", attack_wipes_merit_on_block=True
)
NEGSUM_ATTACK_CFG = dataclasses.replace(
    CFG, attack_mode="negative_sum", attack_resets_tenure=True
)


def player(pid: int, **kw) -> PlayerState:
    base = dict(money=0, merit=0, rank=0, tenure=0)
    base.update(kw)
    return PlayerState(id=pid, name=f"玩家{pid}", **base)


def resolve(players, actions, event=CALM, script=None, cfg=CFG, rnd=1, salaries=None):
    """测试里写 {pid: Action(...)} 更省事，这里自动包成 resolve_round 要的列表。"""
    normalized = {
        pid: (a if isinstance(a, list) else [a]) for pid, a in actions.items()
    }
    return rules.resolve_round(
        players=players,
        actions=normalized,
        event=event,
        rng=ScriptedRng(script or []),
        round_number=rnd,
        cfg=cfg,
        salaries=salaries,
    )


# ==========================================================================
# 1. WORK 收益计算
# ==========================================================================


class TestWorkMerit(unittest.TestCase):
    def test_base_values_at_base_rank(self):
        for base in (8, 9, 10, 11, 12):
            self.assertEqual(rules.work_merit(base, 0, CALM, CFG), base)

    def test_spec_example_city_rank(self):
        # 规则书第 6 节：市级玩家 WORK +9 -> 9 x 2 = 18
        self.assertEqual(rules.work_merit(9, 2, CALM, CFG), 18)

    def test_spec_example_event_then_multiplier(self):
        # 规则书第 18 节：WORK +9，重点项目 +1，市级 x2 -> (9 + 1) x 2 = 20
        self.assertEqual(rules.work_merit(9, 2, KEY_PROJECT_SPEC, CFG), 20)

    def test_current_key_project_bonus(self):
        # 真实配置是 +4：WORK +9，市级 x2 -> (9 + 4) x 2 = 26
        self.assertEqual(rules.work_merit(9, 2, KEY_PROJECT, CFG), 26)

    def test_money_event_does_not_touch_work(self):
        self.assertEqual(rules.work_merit(10, 2, BOOM, CFG), 20)

    def test_through_resolve_round(self):
        p = player(1, rank=2)
        resolve([p], {1: Action(Card.WORK)}, KEY_PROJECT_SPEC, script=[work_roll(9)])
        self.assertEqual(p.merit, 20)


# ==========================================================================
# 2. CORRUPT 收益计算
# ==========================================================================


class TestCorruptMoney(unittest.TestCase):
    def test_spec_example_city_rank(self):
        # 规则书第 7 节：市级玩家 CORRUPT +11 -> 11 x 2 = 22
        self.assertEqual(rules.corrupt_money(11, 2, CALM, CFG), 22)

    def test_spec_example_boom(self):
        # 规则书第 18 节：CORRUPT +10，市级 x2，经济一片大好 x2 -> 40
        self.assertEqual(rules.corrupt_money(10, 2, BOOM, CFG), 40)

    def test_recession_floors_down(self):
        # 9 x 0.5 x 1 = 4.5 -> 4（向下取整，且中间不能有浮点误差）
        self.assertEqual(rules.corrupt_money(9, 0, RECESSION, CFG), 4)
        # 县级 x1.5: 9 x 0.5 x 1.5 = 6.75 -> 6
        self.assertEqual(rules.corrupt_money(9, 1, RECESSION, CFG), 6)

    def test_recorded_as_corrupt_amount(self):
        p = player(1, rank=2)
        out = resolve([p], {1: [Action(Card.CORRUPT), Action(Card.PROMOTE_MONEY)]},
                      BOOM, script=[corrupt_roll(10)])
        self.assertEqual(out.outcomes[1].corrupt_amount, 40)
        self.assertEqual(out.outcomes[1].money_gained, 40)
        # 40 够不够市级->省级的门槛要看配置；金钱余额现在不衰减
        cost = CFG.money_cost(2)
        if 40 >= cost:
            self.assertEqual(p.rank, 3)
            self.assertEqual(p.money, 40 - cost)
        else:
            self.assertEqual(p.rank, 2)   # 不够就升不了
            self.assertEqual(p.money, 40)


# ==========================================================================
# 3. 官职倍率
# ==========================================================================


class TestRankMultipliers(unittest.TestCase):
    def test_all_ranks(self):
        # 基层 x1 / 县级 x1.5 / 市级 x2 / 省级 x2.5
        expected = {0: 10, 1: 15, 2: 20, 3: 25}
        for rank, value in expected.items():
            self.assertEqual(rules.work_merit(10, rank, CALM, CFG), value)
            self.assertEqual(rules.corrupt_money(10, rank, CALM, CFG), value)

    def test_fraction_rank_floors(self):
        # 县级 x1.5：9 x 1.5 = 13.5 -> 13
        self.assertEqual(rules.work_merit(9, 1, CALM, CFG), 13)


# ==========================================================================
# 4./6. 金钱晋升
# ==========================================================================


class TestMoneyPromotion(unittest.TestCase):
    """按规则书原版配置测（门槛与自动晋升，见 SPEC_CFG）。"""

    @staticmethod
    def resolve(players, actions, event=CALM, script=None, rnd=1):
        return resolve(players, actions, event, script, cfg=SPEC_CFG, rnd=rnd)

    def test_spec_example(self):
        # 规则书第 9 节：市级 -> 省级需要 30，玩家有 43 -> 晋升后 money = 3
        p = player(1, rank=2, money=43)
        rules.apply_money_promotion(p, SPEC_CFG)
        self.assertEqual(p.rank, 3)
        self.assertEqual(p.money, 3)

    def test_exact_threshold_leaves_zero(self):
        p = player(1, rank=0, money=10)
        rules.apply_money_promotion(p, SPEC_CFG)
        self.assertEqual((p.rank, p.money), (1, 0))

    def test_merit_untouched(self):
        p = player(1, rank=0, money=14, merit=7)
        rules.apply_money_promotion(p, SPEC_CFG)
        self.assertEqual(p.merit, 7)

    def test_in_round(self):
        p = player(1, rank=0, money=8)
        self.resolve([p], {1: Action(Card.CORRUPT)}, CALM, script=[corrupt_roll(8)])
        # 8 + 8 = 16 >= 10 -> ceil(6/5) = 2
        self.assertEqual((p.rank, p.money), (1, 2))


# ==========================================================================
# 5./6. 政绩晋升
# ==========================================================================


class TestMeritPromotion(unittest.TestCase):
    """按规则书原版配置测（门槛与自动晋升，见 SPEC_CFG）。"""

    @staticmethod
    def resolve(players, actions, event=CALM, script=None, rnd=1):
        return resolve(players, actions, event, script, cfg=SPEC_CFG, rnd=rnd)

    def test_spec_overflow_example(self):
        # 规则书第 10 节的通用例子：需要 10，玩家有 19 -> 晋升后 merit = 2
        cfg = dataclasses.replace(CFG, promotion_merit_costs=[10, 32, 52, 80])
        p = player(1, rank=0, merit=19)
        rules.apply_merit_promotion(p, cfg)
        self.assertEqual((p.rank, p.merit), (1, 2))

    def test_default_thresholds(self):
        p = player(1, rank=0, merit=20)
        rules.apply_merit_promotion(p, SPEC_CFG)  # 20 - 18 = 2 -> ceil(2/5) = 1
        self.assertEqual((p.rank, p.merit), (1, 1))

    def test_money_untouched(self):
        p = player(1, rank=0, merit=18, money=9)
        rules.apply_merit_promotion(p, SPEC_CFG)
        self.assertEqual(p.money, 9)


class TestOverflowRounding(unittest.TestCase):
    """第 6 条：超额资源 /5 向上取整（政绩用；金钱已改成不衰减）。"""

    def test_money_and_merit_now_decay_differently(self):
        self.assertEqual(REAL_CFG.merit_overflow_divisor, 5)
        self.assertEqual(REAL_CFG.money_overflow_divisor, 1)
        # 同样剩 13：政绩砍到 3，金钱原样保留
        self.assertEqual(
            rules.overflow_after_promotion(13, REAL_CFG, REAL_CFG.merit_overflow_divisor), 3
        )
        self.assertEqual(
            rules.overflow_after_promotion(13, REAL_CFG, REAL_CFG.money_overflow_divisor), 13
        )

    def test_ceiling(self):
        self.assertEqual(rules.overflow_after_promotion(13, CFG), 3)  # 13/5 = 2.6 -> 3
        self.assertEqual(rules.overflow_after_promotion(9, CFG), 2)
        self.assertEqual(rules.overflow_after_promotion(10, CFG), 2)
        self.assertEqual(rules.overflow_after_promotion(11, CFG), 3)
        self.assertEqual(rules.overflow_after_promotion(1, CFG), 1)
        self.assertEqual(rules.overflow_after_promotion(0, CFG), 0)


# ==========================================================================
# 7. 同时满足两种晋升条件时优先政绩
# ==========================================================================


class TestPromotionPriority(unittest.TestCase):
    """按规则书原版配置测（门槛与自动晋升，见 SPEC_CFG）。"""

    @staticmethod
    def resolve(players, actions, event=CALM, script=None, rnd=1):
        return resolve(players, actions, event, script, cfg=SPEC_CFG, rnd=rnd)

    def test_merit_preferred_and_money_kept(self):
        p = player(1, rank=0, money=25, merit=20)
        out = self.resolve([p], {})
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.MERIT)
        self.assertEqual(p.rank, 1)
        self.assertEqual(p.merit, 1)  # ceil((20-18)/5)
        self.assertEqual(p.money, 25)  # 金钱保持不变

    def test_only_money_uses_money(self):
        p = player(1, rank=0, money=25, merit=3)
        out = self.resolve([p], {})
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.MONEY)
        self.assertEqual((p.rank, p.money, p.merit), (1, 3, 3))

    def test_at_most_one_promotion_per_round(self):
        # 政绩够升两级也只升一级
        p = player(1, rank=0, merit=200)
        self.resolve([p], {})
        self.assertEqual(p.rank, 1)


# ==========================================================================
# 8./9. 政治攻击
# ==========================================================================


class TestAttack(unittest.TestCase):
    """规则书第 12 节：固定政绩处罚版（attack_mode="merit_penalty"）。

    默认配置现在用的是抢功模式，所以本类统一用 LEGACY_ATTACK_CFG 结算。
    """

    # 规则书原版的攻击 + 规则书原版的晋升（自动、原门槛）
    CFG_ = dataclasses.replace(
        SPEC_CFG, attack_mode="merit_penalty", attack_resets_tenure=False,
        attack_merit_penalty=1,
    )

    @staticmethod
    def resolve(players, actions, event=CALM, script=None, rnd=1):
        return resolve(players, actions, event, script, cfg=TestAttack.CFG_, rnd=rnd)

    def test_blocks_merit_promotion(self):
        target = player(1, rank=0, merit=20, money=0)
        attacker = player(2, rank=0)
        out = self.resolve([target, attacker], {2: Action(Card.ATTACK, 1)})
        self.assertTrue(out.outcomes[1].attacked)
        self.assertTrue(out.outcomes[1].merit_promotion_blocked)
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.NONE)
        self.assertEqual(target.rank, 0)

    def test_money_promotion_still_allowed(self):
        target = player(1, rank=0, merit=20, money=10)
        attacker = player(2, rank=0)
        out = self.resolve([target, attacker], {2: Action(Card.ATTACK, 1)})
        # 政绩晋升被阻，但金钱晋升照常（此时只满足金钱一种"可用"途径）
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.MONEY)
        self.assertEqual(target.rank, 1)

    def test_non_work_target_loses_merit(self):
        target = player(1, merit=5)
        attacker = player(2)
        out = self.resolve([target, attacker], {1: Action(Card.CORRUPT), 2: Action(Card.ATTACK, 1)},
                      script=[corrupt_roll(8)])
        self.assertEqual(target.merit, 4)
        self.assertEqual(out.outcomes[1].attack_merit_loss, 1)

    def test_idle_target_also_loses_merit(self):
        target = player(1, merit=5)
        attacker = player(2)
        self.resolve([target, attacker], {2: Action(Card.ATTACK, 1)})
        self.assertEqual(target.merit, 4)

    def test_work_target_is_immune_to_penalty(self):
        target = player(1, merit=5)
        attacker = player(2)
        out = self.resolve(
            [target, attacker],
            {1: Action(Card.WORK), 2: Action(Card.ATTACK, 1)},
            script=[work_roll(8)],
        )
        self.assertEqual(out.outcomes[1].attack_merit_loss, 0)
        self.assertEqual(target.merit, 13)  # 5 + 8，没有 -1

    def test_merit_never_below_zero(self):
        target = player(1, merit=0)
        attacker = player(2)
        self.resolve([target, attacker], {2: Action(Card.ATTACK, 1)})
        self.assertEqual(target.merit, 0)

    def test_stable_event_disables_attack(self):
        target = player(1, merit=5, rank=0)
        attacker = player(2)
        out = self.resolve([target, attacker], {2: Action(Card.ATTACK, 1)}, STABLE)
        self.assertFalse(out.outcomes[1].attacked)
        self.assertEqual(target.merit, 5)


class TestAttackStealMerit(unittest.TestCase):
    """零和抢功模式（attack_mode="steal_merit"）：目标掉多少，攻击者就拿多少。"""

    @staticmethod
    def resolve(players, actions, event=CALM, script=None, cfg=STEAL_ATTACK_CFG, rnd=1):
        return resolve(players, actions, event, script, cfg=cfg, rnd=rnd)

    def test_steals_the_target_merit_stock(self):
        target = player(1, merit=15)
        attacker = player(2, merit=0)
        out = self.resolve([target, attacker], {2: Action(Card.ATTACK, 1)})
        self.assertEqual(target.merit, 0)  # 默认比例 = 1，连锅端
        self.assertEqual(attacker.merit, 15)
        self.assertEqual(out.outcomes[1].merit_stolen_by_attackers, 15)
        self.assertEqual(out.outcomes[2].merit_from_attacks, 15)

    def test_fraction_is_configurable(self):
        cfg = dataclasses.replace(STEAL_ATTACK_CFG, attack_steal_fraction=Fraction(1, 3))
        target = player(1, merit=15)
        attacker = player(2, merit=0)
        self.resolve([target, attacker], {2: Action(Card.ATTACK, 1)}, cfg=cfg)
        self.assertEqual((target.merit, attacker.merit), (10, 5))

    def test_whiffs_on_someone_with_no_merit(self):
        """在闷声捞钱的人政绩几乎为零，攻击自动扑空——这就是"捞钱克攻击"。"""
        target = player(1, merit=2, money=9)  # 9 < 10，不会因晋升干扰观察
        attacker = player(2, merit=0)
        out = self.resolve([target, attacker], {2: Action(Card.ATTACK, 1)})
        # 只有 2 点政绩可抢，几乎等于白打——攻击者付出的是一整个回合
        self.assertEqual(out.outcomes[1].merit_stolen_by_attackers, 2)
        self.assertEqual(attacker.merit, 2)
        self.assertEqual(target.money, 9)  # 他的钱一分没动，那得靠举报

    def test_steal_happens_after_this_round_gains(self):
        target = player(1, merit=0)
        attacker = player(2, merit=0)
        self.resolve(
            [target, attacker],
            {1: Action(Card.WORK), 2: Action(Card.ATTACK, 1)},
            script=[work_roll(12)],
        )
        # 本轮先赚到 12，再被连锅端走
        self.assertEqual(target.merit, 0)
        self.assertEqual(attacker.merit, 12)

    def test_multiple_attackers_split_the_loot(self):
        target = player(1, merit=15)
        a, b = player(2), player(3)
        self.resolve([target, a, b], {2: Action(Card.ATTACK, 1), 3: Action(Card.ATTACK, 1)})
        self.assertEqual((a.merit, b.merit), (7, 7))  # floor(15/2) 各一份，零头留给目标
        self.assertEqual(target.merit, 1)

    def test_still_blocks_merit_promotion(self):
        cfg = dataclasses.replace(STEAL_ATTACK_CFG, attack_steal_fraction=Fraction(1, 10))
        target = player(1, merit=100)  # 只抢掉一成，仍然够 18 的门槛
        attacker = player(2)
        out = self.resolve(
            [target, attacker],
            {1: Action(Card.PROMOTE_MERIT), 2: Action(Card.ATTACK, 1)},
            cfg=cfg,
        )
        self.assertTrue(out.outcomes[1].merit_promotion_blocked)
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.NONE)
        self.assertEqual(target.rank, 0)

    def test_no_promotion_card_means_nothing_was_blocked(self):
        """没打晋升卡就没有"被挡下"这回事：挨了攻击、资源又够门槛，结算页也不该说"本轮晋升被阻止"。

        踩过（对局 TX3B 第 3 轮）：真人打的是举报 + 贪污，挨了一刀又被反腐风暴查办，
        规则书原版"够了自动升"那套判定照跑，结算页显示"本轮晋升被阻止"。
        """
        target = player(1, merit=100, money=50)
        attacker = player(2)
        reporter = player(3)
        out = resolve(
            [target, attacker, reporter],
            {1: [Action(Card.CORRUPT, value=16)], 2: [Action(Card.ATTACK, 1)],
             3: [Action(Card.REPORT, 1)]},
            cfg=REAL_CFG,
        )
        o = out.outcomes[1]
        self.assertTrue(o.attacked and o.report_effective)
        self.assertFalse(o.merit_promotion_blocked)
        self.assertFalse(o.promotion_blocked_by_attack_report)
        self.assertFalse(o.private_view()["promotion_blocked"])

    def test_stable_event_still_disables_it(self):
        target = player(1, merit=15)  # 15 < 18，不会因为晋升干扰观察
        attacker = player(2)
        self.resolve([target, attacker], {2: Action(Card.ATTACK, 1)}, STABLE)
        self.assertEqual((target.merit, attacker.merit), (15, 0))

    def test_attacker_identity_stays_out_of_public_facts(self):
        target = player(1, merit=30)
        attacker = player(2)
        out = self.resolve([target, attacker], {2: Action(Card.ATTACK, 1)})
        for fact in out.public_view()["player_facts"]:
            self.assertNotIn("merit_from_attacks", fact)


class TestAttackNegativeSum(unittest.TestCase):
    """负和模式（attack_mode="negative_sum"）：目标掉一大块，攻击者只拿回一半。"""

    @staticmethod
    def resolve(players, actions, event=CALM, script=None, cfg=NEGSUM_ATTACK_CFG, rnd=1):
        return resolve(players, actions, event, script, cfg=cfg, rnd=rnd)

    def test_target_loses_two_turns_attacker_gains_one(self):
        target = player(1, merit=40)  # 家底够厚，吃得下满额伤害
        attacker = player(2, merit=0)
        out = self.resolve([target, attacker], {2: Action(Card.ATTACK, 1)})
        # 基层 x1：满额伤害 = 2 个回合当量 = 20；攻击者拿回一半 = 10
        self.assertEqual(out.outcomes[1].attack_merit_loss, 20)
        self.assertEqual(out.outcomes[2].merit_from_attacks, 10)
        self.assertEqual(target.merit, 20)
        self.assertEqual(attacker.merit, 10)

    def test_damage_is_capped_by_what_the_target_actually_has(self):
        target = player(1, merit=6)
        attacker = player(2, merit=0)
        out = self.resolve([target, attacker], {2: Action(Card.ATTACK, 1)})
        self.assertEqual(out.outcomes[1].attack_merit_loss, 6)
        self.assertEqual(attacker.merit, 3)  # 只打掉 6，就只拿回 3

    def test_it_is_negative_sum(self):
        """全场政绩总量一定减少——这就是渔翁得利的来源。"""
        target = player(1, merit=40)
        attacker = player(2, merit=0)
        bystander = player(3, merit=10)
        before = target.merit + attacker.merit + bystander.merit
        self.resolve([target, attacker, bystander], {2: Action(Card.ATTACK, 1)})
        after = target.merit + attacker.merit + bystander.merit
        self.assertLess(after, before)
        self.assertEqual(before - after, 10)  # -20 +10，净蒸发一个回合当量
        self.assertEqual(bystander.merit, 10)  # 没参战的人毫发无损，相对地位上升

    def test_mutual_attack_destroys_both_and_benefits_nobody(self):
        a = player(1, merit=40)
        b = player(2, merit=40)
        bystander = player(3, merit=10)
        out = self.resolve(
            [a, b, bystander],
            {1: Action(Card.ATTACK, 2), 2: Action(Card.ATTACK, 1)},
        )
        self.assertEqual(a.merit, 20)  # 各自 -2 个回合当量
        self.assertEqual(b.merit, 20)
        self.assertEqual(out.outcomes[1].merit_from_attacks, 0)
        self.assertEqual(out.outcomes[2].merit_from_attacks, 0)
        # 两人合计蒸发 40 点政绩，旁观者一动不动就白赚了相对位次
        self.assertEqual(bystander.merit, 10)

    def test_mutual_damage_is_symmetric_regardless_of_order(self):
        """伤害基于攻击前的快照，所以先后结算顺序不影响结果。"""
        a = player(1, merit=30)
        b = player(2, merit=30)
        self.resolve([a, b], {1: Action(Card.ATTACK, 2), 2: Action(Card.ATTACK, 1)})
        self.assertEqual(a.merit, b.merit)

    def test_whiffs_on_a_money_player(self):
        target = player(1, merit=0, money=9)
        attacker = player(2, merit=0)
        out = self.resolve([target, attacker], {2: Action(Card.ATTACK, 1)})
        self.assertEqual(out.outcomes[1].attack_merit_loss, 0)
        self.assertEqual(attacker.merit, 0)
        self.assertEqual(target.money, 9)  # 钱要靠举报，攻击碰不到

    def test_gain_ratio_zero_is_pure_mutual_destruction(self):
        cfg = dataclasses.replace(NEGSUM_ATTACK_CFG, attack_gain_ratio=Fraction(0))
        target = player(1, merit=40)
        attacker = player(2, merit=0)
        self.resolve([target, attacker], {2: Action(Card.ATTACK, 1)}, cfg=cfg)
        self.assertEqual((target.merit, attacker.merit), (20, 0))

    def test_tenure_is_still_wrecked(self):
        target = player(1, merit=40, tenure=2)
        attacker = player(2)
        out = self.resolve([target, attacker], {2: Action(Card.ATTACK, 1)})
        self.assertEqual(out.outcomes[1].tenure_reset_by_attack, 2)
        self.assertEqual(target.tenure, 1)  # 清零之后本轮没晋升，工龄重新 +1


class TestAttackDenial(unittest.TestCase):
    """纯破坏模式（attack_mode="denial"）。线上已换成抢功，这里钉住这个模式本身。"""

    @staticmethod
    def resolve(players, actions, event=CALM, script=None, rnd=1):
        return resolve(players, actions, event, script, cfg=DENIAL_ATTACK_CFG, rnd=rnd)

    def test_wipes_merit_when_it_actually_blocks_a_promotion(self):
        tc = CFG.merit_cost(0)
        target = player(1, merit=tc + 20)
        attacker = player(2)
        out = self.resolve(
            [target, attacker],
            {1: [Action(Card.PROMOTE_MERIT)], 2: [Action(Card.ATTACK, 1)]},
        )
        self.assertEqual(target.rank, 0)  # 升职失败
        self.assertEqual(target.merit, 0)  # 政绩清零
        # 清零发生在生产和攻击处罚**之前**，所以整堆政绩一次性作废
        self.assertEqual(out.outcomes[1].merit_wiped_by_attack, tc + 20)

    def test_also_blocks_the_generic_promotion_card(self):
        tc = CFG.merit_cost(0)
        target = player(1, merit=tc + 5, money=0)
        self.resolve(
            [target, player(2)],
            {1: [Action(Card.PROMOTE_ANY)], 2: [Action(Card.ATTACK, 1)]},
        )
        self.assertEqual((target.rank, target.merit), (0, 0))

    def test_worker_who_is_not_cashing_in_is_immune(self):
        target = player(1, merit=5)
        out = self.resolve(
            [target, player(2)],
            {1: [Action(Card.WORK)], 2: [Action(Card.ATTACK, 1)]},
            script=[work_roll(10)],
        )
        self.assertEqual(out.outcomes[1].attack_merit_loss, 0)
        self.assertEqual(target.merit, 15)  # 5 + 10，一点没掉

    def test_non_worker_loses_a_fixed_chunk(self):
        target = player(1, merit=20)
        out = self.resolve(
            [target, player(2)],
            {1: [Action(Card.REPORT, 2)], 2: [Action(Card.ATTACK, 1)]},
        )
        self.assertEqual(out.outcomes[1].attack_merit_loss, CFG.attack_merit_penalty)
        self.assertEqual(target.merit, 20 - CFG.attack_merit_penalty)

    def test_catching_corruption_earns_the_attacker_merit_not_money(self):
        """默认 merit_to_attacker：抄家是举报的活，攻击只是"揭发立功"。"""
        target = player(1, merit=0, money=5)
        attacker = player(2, money=0, merit=0)
        out = self.resolve(
            [target, attacker],
            {1: [Action(Card.CORRUPT)], 2: [Action(Card.ATTACK, 1)]},
            script=[corrupt_roll(10)],
        )
        self.assertEqual(out.outcomes[1].corrupt_amount, 10)
        self.assertEqual(out.outcomes[1].money_confiscated, 0)  # 没被抄家
        self.assertEqual(attacker.money, 0)  # 攻击者一分钱没拿
        # 10 x 1/4 = 2 点政绩记在攻击者头上
        self.assertEqual(attacker.merit, 2)
        self.assertEqual(out.outcomes[2].merit_from_attacks, 2)

    def test_being_caught_costs_you_hush_money(self):
        """被人攥住把柄，得自己掏钱上下打点，这笔钱是花掉的、不进谁的口袋。"""
        target = player(1, merit=0, money=5)
        attacker = player(2, money=0, merit=0)
        out = self.resolve(
            [target, attacker],
            {1: [Action(Card.CORRUPT)], 2: [Action(Card.ATTACK, 1)]},
            script=[corrupt_roll(10)],
        )
        o = out.outcomes[1]
        self.assertEqual(o.hush_money_paid, 5)         # 10 的一半
        self.assertEqual(o.net_corrupt_gain, 5)        # 净落袋只剩一半
        self.assertEqual(target.money, 10)             # 5 存款 + 10 赃款 − 5 打点
        self.assertEqual(attacker.money, 0)            # 钱没进攻击者口袋

    def test_hush_money_does_not_shrink_the_gossip(self):
        """打点费不从传闻里扣：传闻按贪污毛额排，花钱消灾压不住风声。"""
        quiet = player(1, merit=0, money=0)
        attacker = player(2)
        loud = player(3, merit=0, money=0)
        out = self.resolve(
            [quiet, attacker, loud],
            {1: [Action(Card.CORRUPT)], 2: [Action(Card.ATTACK, 1)],
             3: [Action(Card.CORRUPT)]},
            script=[corrupt_roll(12), corrupt_roll(8)],
        )
        # 1 号毛收入更高(12)，打点花掉一半只剩 6；3 号原样 8 —— 广播照样点 1 号
        self.assertEqual(out.outcomes[1].net_corrupt_gain, 6)
        self.assertEqual(out.outcomes[3].net_corrupt_gain, 8)
        self.assertIn("玩家1", out.wealth_broadcast[0])
        self.assertNotIn("玩家3", out.wealth_broadcast[0])

    def test_still_gets_reported_on_the_gross_amount(self):
        """打点只是压住了风声，你到底贪没贪还是按毛收入算，该抓照抓。"""
        target = player(1, rank=0, money=0)
        attacker = player(2)
        reporter = player(3)
        out = self.resolve(
            [target, attacker, reporter],
            {1: [Action(Card.CORRUPT)], 2: [Action(Card.ATTACK, 1)],
             3: [Action(Card.REPORT, 1)]},
            script=[corrupt_roll(10)],
        )
        self.assertGreater(out.outcomes[1].hush_money_paid, 0)
        self.assertTrue(out.outcomes[1].report_effective)

    def test_clean_target_pays_nothing(self):
        target = player(1, merit=0, money=20)
        resolve([target, player(2)],
                {1: [Action(Card.WORK)], 2: [Action(Card.ATTACK, 1)]},
                script=[work_roll(10)])
        self.assertEqual(target.money, 20)

    def test_confiscating_mode_is_still_available(self):
        # 撞上贪污这套逻辑只在 denial 模式里，所以基底要用 denial
        cfg = dataclasses.replace(
            DENIAL_ATTACK_CFG, attack_on_corruption="confiscate_to_attacker"
        )
        target = player(1, merit=0, money=5)
        attacker = player(2, money=0)
        resolve(
            [target, attacker],
            {1: [Action(Card.CORRUPT)], 2: [Action(Card.ATTACK, 1)]},
            script=[corrupt_roll(10)], cfg=cfg,
        )
        self.assertEqual(target.money, 5)
        self.assertEqual(attacker.money, 10)

    def test_clean_target_gives_the_attacker_nothing(self):
        target = player(1, merit=0, money=9)
        attacker = player(2, merit=0)
        self.resolve([target, attacker],
                     {1: [Action(Card.WORK)], 2: [Action(Card.ATTACK, 1)]},
                     script=[work_roll(10)])
        self.assertEqual(attacker.merit, 0)

    def test_does_not_touch_tenure(self):
        target = player(1, merit=20, tenure=2)
        self.resolve([target, player(2)],
                     {1: [Action(Card.WORK)], 2: [Action(Card.ATTACK, 1)]},
                script=[work_roll(10)])
        self.assertEqual(target.tenure, 3)  # 照常 +1，没被清

    def test_attacker_gains_nothing_from_merit(self):
        tc = CFG.merit_cost(0)
        target = player(1, merit=tc + 30)
        attacker = player(2, merit=0)
        resolve([target, attacker],
                {1: [Action(Card.PROMOTE_MERIT)], 2: [Action(Card.ATTACK, 1)]})
        self.assertEqual(attacker.merit, 0)  # 清掉的政绩凭空蒸发，不进攻击者口袋


# ==========================================================================
# 10.-14. 匿名举报
# ==========================================================================


class TestReport(unittest.TestCase):
    """举报查实的后果：没收 + 严重警告，攒够两次才降级。"""

    def test_no_corruption_means_invalid(self):
        target = player(1, rank=2, money=5, merit=3)
        reporter = player(2)
        out = resolve([target, reporter], {1: Action(Card.WORK), 2: Action(Card.REPORT, 1)},
                      script=[work_roll(8)])
        self.assertTrue(out.outcomes[1].reported)
        self.assertFalse(out.outcomes[1].report_effective)
        self.assertEqual(out.outcomes[1].demotion, DemotionKind.NONE)
        self.assertEqual(target.rank, 2)
        self.assertEqual(target.warnings, 0)

    def test_first_time_caught_is_a_warning_not_a_demotion(self):
        target = player(1, rank=2, money=5, merit=7, tenure=3)
        reporter = player(2)
        out = resolve(
            [target, reporter],
            {1: Action(Card.CORRUPT), 2: Action(Card.REPORT, 1)},
            script=[corrupt_roll(8)],
        )
        o = out.outcomes[1]
        self.assertTrue(o.report_effective)
        self.assertEqual(o.warnings_issued, 1)
        self.assertEqual(target.warnings, 1)
        self.assertEqual(o.demotion, DemotionKind.NONE)
        self.assertEqual(target.rank, 2)  # 还没降
        self.assertEqual(target.tenure, 0)  # 但工龄清零了
        self.assertEqual(target.merit, 7)  # 政绩不动

    def test_second_warning_demotes_and_clears_the_slate(self):
        target = player(1, rank=2, money=5, merit=7, warnings=1)
        reporter = player(2)
        out = resolve(
            [target, reporter],
            {1: Action(Card.CORRUPT), 2: Action(Card.REPORT, 1)},
            script=[corrupt_roll(8)],
        )
        self.assertEqual(out.outcomes[1].demotion, DemotionKind.MINOR)
        self.assertEqual(target.rank, 1)
        self.assertEqual(target.warnings, 0)  # 降完清空，重新记
        self.assertEqual(target.tenure, 0)

    def test_only_this_round_haul_is_confiscated(self):
        """没收的是本轮贪的那一笔，存款不动——不再有"抄家"这回事。"""
        target = player(1, rank=1, money=40, merit=7)
        reporter = player(2)
        out = resolve(
            [target, reporter],
            {1: Action(Card.CORRUPT), 2: Action(Card.REPORT, 1)},
            RECESSION,
            script=[corrupt_roll(8)],
        )
        haul = out.outcomes[1].corrupt_amount
        self.assertEqual(out.outcomes[1].money_confiscated, haul)
        self.assertEqual(target.money, 40)  # 存款一分没少（赃款进来又出去）
        self.assertEqual(target.merit, 7)

    def test_a_big_haul_is_not_extra_punished_by_default(self):
        """大案默认只记一次警告——MAJOR_CORRUPTION_WARNINGS 调成 2 才恢复重罚。"""
        target = player(1, rank=3, money=40, merit=55, tenure=2)
        reporter = player(2)
        out = resolve(
            [target, reporter],
            {1: [Action(Card.CORRUPT, value=17)], 2: [Action(Card.REPORT, 1)]},
            cfg=REAL_CFG,
        )
        self.assertGreaterEqual(
            out.outcomes[1].corrupt_amount, REAL_CFG.major_corruption_threshold
        )
        self.assertEqual(out.outcomes[1].warnings_issued, 1)
        self.assertEqual(target.rank, 3)  # 没被打回基层

    def test_the_switch_restores_harsh_punishment_for_big_hauls(self):
        cfg = dataclasses.replace(REAL_CFG, major_corruption_warnings=2)
        target = player(1, rank=3, money=40, merit=55)
        reporter = player(2)
        out = resolve(
            [target, reporter],
            {1: [Action(Card.CORRUPT, value=17)], 2: [Action(Card.REPORT, 1)]},
            cfg=cfg,
        )
        self.assertEqual(out.outcomes[1].warnings_issued, 2)
        self.assertEqual(out.outcomes[1].demotion, DemotionKind.MINOR)
        self.assertEqual(target.rank, 2)  # 一次记满两警告，当场降一级

    def test_demotion_floors_at_base_rank(self):
        target = player(1, rank=0, money=0, merit=7, warnings=1)
        resolve(
            [target, player(2)],
            {1: Action(Card.CORRUPT), 2: Action(Card.REPORT, 1)},
            RECESSION,
            script=[corrupt_roll(8)],
        )
        self.assertEqual(target.rank, 0)  # 不会变成 -1

    def test_warnings_carry_across_rounds(self):
        """警告是累计的，不是每轮清零——这是整个机制的意义所在。"""
        target = player(1, rank=2, money=0, merit=0)
        reporter = player(2)
        for expected in (1, 0):  # 第 2 次记满降级后清空
            resolve(
                [target, reporter],
                {1: Action(Card.CORRUPT), 2: Action(Card.REPORT, 1)},
                script=[corrupt_roll(8)],
            )
            self.assertEqual(target.warnings, expected)
        self.assertEqual(target.rank, 1)


class TestBriberyIsReportable(unittest.TestCase):
    """拿钱买官也是经济问题：被举报的话钱没了、官也升不成。"""

    def _run(self, picks, reported=True, **kw):
        target = player(1, **{"rank": 0, "money": 20, "merit": 0, **kw})
        reporter = player(2)
        acts = {1: picks}
        if reported:
            acts[2] = [Action(Card.REPORT, 1)]
        out = resolve([target, reporter], acts, cfg=REAL_CFG)
        return target, out.outcomes[1], out.outcomes[2]

    def test_bribing_alone_is_enough_to_be_caught(self):
        """这一轮一张贪污牌都没打，光是拿钱买官也会被查实。"""
        tgt, o, _ = self._run([Action(Card.PROMOTE_MONEY)])
        self.assertEqual(o.corrupt_amount, 0)
        self.assertTrue(o.report_effective)
        self.assertEqual(o.warnings_issued, 1)

    def test_the_bribe_money_is_gone_and_the_promotion_fails(self):
        tgt, o, _ = self._run([Action(Card.PROMOTE_MONEY)])
        self.assertEqual(o.bribe_lost, REAL_CFG.money_cost(0))
        self.assertEqual(o.promotion, PromotionKind.NONE)
        self.assertEqual(tgt.rank, 0)
        self.assertEqual(tgt.money, 20 - REAL_CFG.money_cost(0))

    def test_nobody_reports_means_the_bribe_works(self):
        tgt, o, _ = self._run([Action(Card.PROMOTE_MONEY)], reported=False)
        self.assertEqual(o.promotion, PromotionKind.MONEY)
        self.assertEqual(o.bribe_lost, 0)
        self.assertEqual(tgt.rank, 1)

    def test_a_merit_promotion_is_not_the_report_card_business(self):
        """举报克的是金钱路线。政绩升职没花钱，举报拦不住，也没有"行贿的钱"可赔。

        （要拦政绩升职是政治攻击的活 —— 这就是两张牌的分工。）
        """
        tc = REAL_CFG.merit_cost(0)
        tgt, o, _ = self._run(
            [Action(Card.CORRUPT, value=15), Action(Card.PROMOTE_MERIT)], merit=tc
        )
        self.assertTrue(o.report_effective)  # 贪污照样查实、照样记警告
        self.assertEqual(o.bribe_lost, 0)
        self.assertEqual(o.promotion, PromotionKind.MERIT)  # 但升职拦不住

    def test_caught_corrupting_and_bribing_pays_both(self):
        tgt, o, _ = self._run(
            [Action(Card.CORRUPT, value=15), Action(Card.PROMOTE_MONEY)]
        )
        self.assertGreater(o.money_confiscated, 0)
        self.assertGreater(o.bribe_lost, 0)
        self.assertEqual(o.warnings_issued, 1)  # 两件事仍然只记一次警告

    def test_the_seized_bribe_goes_to_the_reporter(self):
        """抓到一个光买官没贪钱的，举报人不能一分钱拿不到。

        踩过：查获的行贿款没进分赃池，直接凭空蒸发，举报人白干一回合。
        """
        tgt, o, mine = self._run([Action(Card.PROMOTE_MONEY)])
        self.assertGreater(o.bribe_lost, 0)
        self.assertEqual(o.money_confiscated, 0)  # 这轮没贪，没有赃款
        self.assertGreater(mine.money_from_reports, 0)
        self.assertEqual(mine.money_from_reports, o.bribe_lost // 2 - REAL_CFG.report_reward_fee)

    def test_the_final_step_counts_as_bribery_too(self):
        """最后一步算不算行贿，取决于**打的是哪张卡**，不取决于钱花没花。

        省级->主席钱和政绩一起花，但"我走的是正规程序还是关系"是你自己声明的：
        打贿赂升职 = 走关系 -> 举报抓得到；打政绩升职 = 走正规 -> 举报碰不到。
        """
        top = REAL_CFG.president_rank - 1
        tc, mc = REAL_CFG.merit_cost(top), REAL_CFG.money_cost(top)

        # 贿赂升职：被举报查实，钱没了、记警告、官没升
        briber = player(1, rank=top, merit=tc + 5, money=mc + 5)
        out = resolve(
            [briber, player(2)],
            {1: [Action(Card.PROMOTE_MONEY)], 2: [Action(Card.REPORT, 1)]},
            cfg=REAL_CFG,
        )
        o = out.outcomes[1]
        self.assertTrue(o.report_effective)
        self.assertEqual(o.bribe_lost, mc)
        self.assertEqual(o.warnings_issued, 1)
        self.assertEqual(o.promotion, PromotionKind.NONE)
        self.assertEqual(briber.rank, top)

        # 政绩升职：同样花了钱，但走的是正规程序，举报碰不到
        honest = player(1, rank=top, merit=tc + 5, money=mc + 5)
        out2 = resolve(
            [honest, player(2)],
            {1: [Action(Card.PROMOTE_MERIT)], 2: [Action(Card.REPORT, 1)]},
            cfg=REAL_CFG,
        )
        o2 = out2.outcomes[1]
        self.assertFalse(o2.report_effective)
        self.assertEqual(o2.bribe_lost, 0)
        self.assertEqual(o2.promotion, PromotionKind.BOTH)
        self.assertEqual(honest.rank, REAL_CFG.president_rank)  # 登顶

    def test_the_switch_makes_bribery_safe_again(self):
        """关掉 REPORT_CATCHES_BRIBERY：举报只管贪污，光买官不构成罪名。"""
        cfg = dataclasses.replace(REAL_CFG, report_catches_bribery=False)
        target = player(1, rank=0, money=20, merit=0)
        out = resolve(
            [target, player(2)],
            {1: [Action(Card.PROMOTE_MONEY)], 2: [Action(Card.REPORT, 1)]},
            cfg=cfg,
        )
        self.assertFalse(out.outcomes[1].report_effective)
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.MONEY)
        self.assertEqual(target.rank, 1)

    def test_merit_path_is_not_convicted_of_bribery(self):
        """政绩够就走政绩那条路，一分钱不用出——不能因为兜里有钱就按行贿论处。

        踩过：只看"钱够不够"，结果本该免费升职的人被判成行贿，
        钱没了、官也没升，顺带把整个金钱路线的价值也压没了。
        """
        tc = REAL_CFG.merit_cost(0)
        tgt, o, _ = self._run([Action(Card.PROMOTE_ANY)], merit=tc + 5, money=20)
        self.assertFalse(o.report_effective)
        self.assertEqual(o.bribe_lost, 0)
        self.assertEqual(o.promotion, PromotionKind.MERIT)
        self.assertEqual(tgt.rank, 1)

    def test_generic_card_is_convicted_when_money_is_the_only_path(self):
        tgt, o, _ = self._run([Action(Card.PROMOTE_ANY)], merit=0, money=20)
        self.assertTrue(o.report_effective)
        self.assertEqual(o.bribe_lost, REAL_CFG.money_cost(0))

    def test_cannot_afford_the_bribe_means_nothing_to_catch(self):
        """钱不够门槛，就谈不上行贿——不该被这条抓住。"""
        tgt, o, _ = self._run([Action(Card.PROMOTE_MONEY)], money=1)
        self.assertFalse(o.report_effective)
        self.assertEqual(o.bribe_lost, 0)


    def test_cannot_report_self(self):
        p = player(1, rank=2)
        out = resolve([p, player(2)], {1: Action(Card.REPORT, 1)}, script=[])
        self.assertFalse(out.outcomes[1].reported)

    def test_mass_report_event_hits_everyone_who_corrupted(self):
        a = player(1, rank=0)
        b = player(2, rank=0)
        out = resolve(
            [a, b],
            {1: Action(Card.CORRUPT), 2: Action(Card.WORK)},
            ANTI_CORRUPTION,
            script=[corrupt_roll(10), work_roll(10)],
        )
        self.assertTrue(out.outcomes[1].report_effective)
        self.assertFalse(out.outcomes[2].report_effective)  # 没贪污 -> 无效


# ==========================================================================
# 15. ATTACK + REPORT 同时命中
# ==========================================================================


class TestAttackPlusReport(unittest.TestCase):
    def _setup(self):
        # 市级玩家，本轮贪污后同时满足金钱(30)和政绩(52)门槛
        target = player(1, rank=2, money=20, merit=60)
        attacker = player(2)
        reporter = player(3)
        actions = {
            1: Action(Card.CORRUPT),
            2: Action(Card.ATTACK, 1),
            3: Action(Card.REPORT, 1),
        }
        cfg = dataclasses.replace(
            SPEC_CFG, attack_mode="merit_penalty", attack_resets_tenure=False,
            attack_merit_penalty=1,
        )
        out = resolve([target, attacker, reporter], actions, script=[corrupt_roll(10)], cfg=cfg)
        return target, reporter, out

    def test_promotion_fails(self):
        _, _, out = self._setup()
        o = out.outcomes[1]
        self.assertTrue(o.attacked)
        self.assertTrue(o.report_effective)
        self.assertTrue(o.promotion_blocked_by_attack_report)
        self.assertEqual(o.promotion, PromotionKind.NONE)

    def test_warned_by_the_report(self):
        target, _, out = self._setup()
        # 第一次被查实只记警告，不降级（攒够两次才降）
        self.assertEqual(out.outcomes[1].warnings_issued, 1)
        self.assertEqual(target.warnings, 1)
        self.assertEqual(target.rank, 2)

    def test_merit_kept(self):
        target, _, _ = self._setup()
        self.assertEqual(target.merit, 59)  # 60 - 1（非 WORK 被攻击的政绩处罚）

    def test_money_goes_to_the_reporter_not_to_a_failed_promotion(self):
        target, reporter, out = self._setup()
        haul = out.outcomes[1].corrupt_amount
        self.assertEqual(out.outcomes[1].money_confiscated, haul)  # 只吐本轮这一笔
        self.assertEqual(target.money, 20)  # 存款保住（赃款进来又出去）
        self.assertEqual(out.outcomes[3].money_from_reports, haul // 2)

    def test_needs_both_thresholds_for_the_flag(self):
        # 只满足金钱门槛时不算规则书第 16 节的情况，但举报降级照旧
        target = player(1, rank=2, money=20, merit=0)
        out = resolve(
            [target, player(2), player(3)],
            {1: Action(Card.CORRUPT), 2: Action(Card.ATTACK, 1), 3: Action(Card.REPORT, 1)},
            script=[corrupt_roll(10)],
            cfg=dataclasses.replace(
                SPEC_CFG, attack_mode="merit_penalty", attack_resets_tenure=False,
                attack_merit_penalty=1,
            ),
        )
        self.assertFalse(out.outcomes[1].promotion_blocked_by_attack_report)
        self.assertTrue(out.outcomes[1].report_effective)
        self.assertEqual(out.outcomes[1].warnings_issued, 1)


# ==========================================================================
# 没收赃款 + 分赃（举报人拿走赃款）
# ==========================================================================


class TestReportReward(unittest.TestCase):
    def test_minor_only_takes_this_round_haul(self):
        target = player(1, rank=0, money=30, merit=0)
        reporter = player(2)
        out = resolve(
            [target, reporter],
            {1: Action(Card.CORRUPT), 2: Action(Card.REPORT, 1)},
            RECESSION,
            script=[corrupt_roll(8)],
        )
        # 8 x 0.5 = 4 < 8 -> 小额：只没收这 4 块，30 的存款保住
        self.assertEqual(out.outcomes[1].money_confiscated, 4)
        self.assertEqual(target.money, 30)
        self.assertEqual(out.outcomes[2].money_from_reports, 2)  # 一半归举报人

    def test_savings_survive_even_a_big_haul(self):
        """不再有"抄家"：只没收本轮贪的那一笔，存款一分不动。"""
        target = player(1, rank=0, money=25, merit=0)
        reporter = player(2)
        out = resolve(
            [target, reporter],
            {1: Action(Card.CORRUPT), 2: Action(Card.REPORT, 1)},
            script=[corrupt_roll(10)],
        )
        self.assertEqual(out.outcomes[1].corrupt_amount, 10)
        self.assertEqual(out.outcomes[1].money_confiscated, 10)  # 只吐本轮这 10
        self.assertEqual(target.money, 25)  # 存款保住
        self.assertEqual(out.outcomes[2].money_from_reports, 5)  # 举报人拿一半

    def test_split_evenly_and_floor(self):
        target = player(1, rank=0, money=1, merit=0)
        a, b = player(2), player(3)
        out = resolve(
            [target, a, b],
            {
                1: Action(Card.CORRUPT),
                2: Action(Card.REPORT, 1),
                3: Action(Card.REPORT, 1),
            },
            script=[corrupt_roll(10)],
        )
        # 只没收本轮贪的 10（存款 1 不动）；举报人那一份是 floor(10/2)=5，
        # 两人再平分 -> 每人 2，零头和另一半都充公
        self.assertEqual(out.outcomes[1].money_confiscated, 10)
        self.assertEqual(target.money, 1)  # 存款保住
        self.assertEqual(out.outcomes[2].money_from_reports, 2)
        self.assertEqual(out.outcomes[3].money_from_reports, 2)
        self.assertEqual(a.money, 2)
        self.assertEqual(b.money, 2)

    def test_event_report_has_no_beneficiary(self):
        # "中央反腐大筛查"那一份举报没有举报人，赃款直接充公
        target = player(1, rank=0, money=5, merit=0)
        other = player(2)
        out = resolve(
            [target, other],
            {1: Action(Card.CORRUPT), 2: Action(Card.WORK)},
            ANTI_CORRUPTION,
            script=[corrupt_roll(10), work_roll(10)],
        )
        self.assertTrue(out.outcomes[1].report_effective)
        self.assertEqual(out.outcomes[1].money_confiscated, 10)  # 本轮赃款
        self.assertEqual(target.money, 5)  # 存款不动
        self.assertEqual(out.outcomes[2].money_from_reports, 0)  # 事件举报没有受益人
        self.assertEqual(other.money, 0)

    def test_reward_is_not_corruption(self):
        """分到的赃款不算贪污：不会让举报人被举报查实（但坊间传闻照样算进去）。

        场上安排：1 号贪了 10 被 2 号举报查实（贪污那项记 0），2 号分到 5；
        4 号贪了 10 没人管——4 号到手比 2 号多，广播点 4 号。
        """
        caught = player(1, rank=0, money=20, merit=0)
        reporter = player(2)
        third = player(3)
        safe = player(4, rank=0)
        out = resolve(
            [caught, reporter, third, safe],
            {
                1: Action(Card.CORRUPT),
                2: Action(Card.REPORT, 1),
                3: Action(Card.REPORT, 2),  # 同时举报举报人
                4: Action(Card.CORRUPT),
            },
            script=[corrupt_roll(10), corrupt_roll(10)],
        )
        self.assertEqual(out.outcomes[2].money_from_reports, 5)  # 没收本轮 10 的一半
        self.assertEqual(out.outcomes[2].corrupt_amount, 0)
        self.assertTrue(out.outcomes[2].reported)
        self.assertFalse(out.outcomes[2].report_effective)  # 他本轮没贪污
        self.assertTrue(out.outcomes[1].report_effective)
        self.assertEqual(out.wealth_top_ids, [4])

    def test_partial_seize_keeps_the_reporters_cut(self):
        """report_seize_ratio < 1：被抓的人留下一部分赃款，但举报人分到的一分不少。"""
        for ratio, kept in ((Fraction(2, 3), 4), (Fraction(1, 2), 6), (Fraction(1, 3), 6)):
            cfg = dataclasses.replace(CFG, report_seize_ratio=ratio)
            caught, reporter = player(1, rank=0), player(2)
            out = resolve([caught, reporter, player(3)],
                          {1: Action(Card.CORRUPT), 2: Action(Card.REPORT, 1)},
                          script=[corrupt_roll(12)], cfg=cfg)
            self.assertEqual(out.outcomes[2].money_from_reports, 6, ratio)  # 12 的一半
            self.assertEqual(caught.money - out.outcomes[1].salary, kept, ratio)

    def test_no_reward_report_still_confiscates_everything(self):
        """举报分成 0：赃款照样全部没收（充公），举报人一分不拿；会计的做账这时才真正保住一半。"""
        cfg = dataclasses.replace(CFG, report_reward_ratio=Fraction(0))
        caught, reporter = player(1, rank=0), player(2)
        out = resolve([caught, reporter, player(3)],
                      {1: Action(Card.CORRUPT), 2: Action(Card.REPORT, 1)},
                      script=[corrupt_roll(12)], cfg=cfg)
        self.assertTrue(out.outcomes[1].report_effective)
        self.assertEqual(out.outcomes[1].money_confiscated, 12)
        self.assertEqual(out.outcomes[2].money_from_reports, 0)
        self.assertNotIn("举报人该分的那份", cfg.origin("ACCOUNTANT")["description"])

    def test_mutual_reports_settle_simultaneously(self):
        a = player(1, rank=0, money=5, merit=0)
        b = player(2, rank=0, money=7, merit=0)
        out = resolve(
            [a, b],
            {1: Action(Card.CORRUPT), 2: Action(Card.CORRUPT)},
            ANTI_CORRUPTION,  # 两人都被举报
            script=[corrupt_roll(10), corrupt_roll(10)],
        )
        # 事件举报没有受益人，没收的赃款直接充公；存款不动
        self.assertEqual(out.outcomes[1].money_confiscated, 10)
        self.assertEqual(out.outcomes[2].money_confiscated, 10)
        self.assertEqual((a.money, b.money), (5, 7))  # 只剩原来的存款

    def test_mutual_player_reports_do_not_depend_on_order(self):
        a = player(1, rank=0, money=5, merit=0)
        b = player(2, rank=0, money=7, merit=0)
        out = resolve(
            [a, b],
            {1: Action(Card.REPORT, 2), 2: Action(Card.REPORT, 1)},
            script=[],
        )
        # 谁都没贪污 -> 两边举报都无效
        self.assertFalse(out.outcomes[1].report_effective)
        self.assertFalse(out.outcomes[2].report_effective)
        self.assertEqual((a.money, b.money), (5, 7))

    def test_reported_player_cannot_promote_this_round(self):
        # 政绩早就够了，但贪污那一轮被举报 -> 本轮升不了
        target = player(1, rank=0, money=0, merit=40)
        reporter = player(2)
        out = resolve(
            [target, reporter],
            {1: Action(Card.CORRUPT), 2: Action(Card.REPORT, 1)},
            script=[corrupt_roll(10)],
        )
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.NONE)
        self.assertEqual(target.rank, 0)
        self.assertEqual(target.merit, 40)  # 政绩保留，下一轮还能用

    def test_disabling_the_reward_keeps_the_money(self):
        cfg = dataclasses.replace(CFG, report_reward_enabled=False)
        target = player(1, rank=0, money=25, merit=0)
        reporter = player(2)
        out = resolve(
            [target, reporter],
            {1: Action(Card.CORRUPT), 2: Action(Card.REPORT, 1)},
            script=[corrupt_roll(10)],
            cfg=cfg,
        )
        self.assertEqual(out.outcomes[1].money_confiscated, 0)
        self.assertEqual(out.outcomes[2].money_from_reports, 0)
        self.assertEqual(target.money, 25 + 10)  # 没人来抄，赃款留在他手里
        self.assertEqual(reporter.money, 0)


# ==========================================================================
# 16. 工龄
# ==========================================================================


class TestTenure(unittest.TestCase):
    """按规则书原版配置测（门槛与自动晋升，见 SPEC_CFG）。"""

    @staticmethod
    def resolve(players, actions, event=CALM, script=None, rnd=1):
        return resolve(players, actions, event, script, cfg=SPEC_CFG, rnd=rnd)

    def test_promotes_after_enough_idle_rounds(self):
        need = SPEC_CFG.tenure_required
        p = player(1)
        for expected in range(1, need):
            self.resolve([p], {})
            self.assertEqual(p.tenure, expected)
            self.assertEqual(p.rank, 0)
        out = self.resolve([p], {})
        self.assertEqual(p.rank, 1)
        self.assertEqual(p.tenure, 0)
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.TENURE)

    def test_resource_promotion_resets_tenure(self):
        p = player(1, tenure=2, merit=18)
        out = self.resolve([p], {})
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.MERIT)
        self.assertEqual(p.tenure, 0)

    def test_any_warning_resets_tenure_even_without_a_demotion(self):
        target = player(1, rank=2, tenure=2)
        out = self.resolve(
            [target, player(2)],
            {1: Action(Card.CORRUPT), 2: Action(Card.REPORT, 1)},
            script=[corrupt_roll(8)],
        )
        self.assertTrue(out.outcomes[1].report_effective)
        self.assertEqual(out.outcomes[1].demotion, DemotionKind.NONE)  # 第一次只是警告
        self.assertEqual(target.tenure, 0)  # 工龄照样清零

    def test_tenure_can_reach_president_when_allowed(self):
        """规则书原版允许熬资历一路熬到主席；真实配置已关掉（见 TestPresidentNeedsBoth）。"""
        cfg = dataclasses.replace(SPEC_CFG, tenure_can_reach_president=True)
        p = player(1, rank=3, tenure=cfg.tenure_required - 1)
        out = resolve([p], {}, cfg=cfg)
        self.assertEqual(p.rank, cfg.president_rank)
        self.assertEqual(out.presidents, [1])


# ==========================================================================
# 19./20. 财富广播
# ==========================================================================


class TestWealthBroadcast(unittest.TestCase):
    def test_no_corruption_no_broadcast(self):
        self.assertEqual(rules.wealth_broadcast([("A", 0), ("B", 0)], CFG), [])

    def test_only_the_top_is_named(self):
        msgs = rules.wealth_broadcast([("玩家1", 20), ("玩家2", 6), ("玩家3", 3)], CFG)
        self.assertEqual(len(msgs), 1)
        self.assertIn("玩家1", msgs[0])
        self.assertNotIn("玩家2", msgs[0])
        self.assertNotIn("玩家3", msgs[0])

    def test_never_reveals_the_amount(self):
        msgs = rules.wealth_broadcast([("玩家1", 20)], CFG)
        self.assertNotIn("20", msgs[0])

    def test_the_wording_carries_no_information_about_the_amount(self):
        """这是改掉分档制的**全部理由**：文案不能再暗示金额。

        老版本按 1-15 / 16-27 / 28-44 / 45+ 分四档，听到"住上洋房了"
        就知道对方至少 45——AI 直接拿档位反推区间，真人却得背档位表。
        判据写成两条对偶：小额挑得出大话，大额也挑得出小话。
        """
        import random as _r

        def lines(amount):
            return {
                rules.wealth_broadcast_detail([("A", amount)], CFG, _r.Random(i))[0][0]
                for i in range(200)
            }

        small, large = lines(1), lines(9999)
        self.assertEqual(small, large, "不同金额能挑到的句子不一样 = 文案在泄露金额")
        self.assertGreater(len(small), 1, "随机挑了半天还是同一句")

    def test_every_pool_has_several_lines(self):
        """同一句每轮重复玩家就不看了。"""
        self.assertGreater(len(CFG.wealth_broadcast_lines), 1)
        self.assertGreater(len(CFG.wealth_broadcast_lines_multi), 1)

    def test_no_line_mentions_a_number(self):
        for tpl in CFG.wealth_broadcast_lines + CFG.wealth_broadcast_lines_multi:
            self.assertFalse(
                any(ch.isdigit() for ch in tpl.replace("{names}", "")), tpl
            )

    def test_ties_are_all_broadcast(self):
        msgs = rules.wealth_broadcast([("玩家1", 6), ("玩家2", 2), ("玩家3", 6)], CFG)
        self.assertEqual(len(msgs), 1)
        self.assertIn("玩家1", msgs[0])
        self.assertIn("玩家3", msgs[0])
        self.assertNotIn("玩家2", msgs[0])
        self.assertIn("都", msgs[0])

    def test_through_resolve_round(self):
        a = player(1, rank=2)
        b = player(2, rank=0)
        out = resolve(
            [a, b],
            {1: Action(Card.CORRUPT), 2: Action(Card.CORRUPT)},
            script=[corrupt_roll(10), corrupt_roll(10)],
        )
        self.assertEqual(len(out.wealth_broadcast), 1)
        self.assertIn("玩家1", out.wealth_broadcast[0])  # 20 > 10
        self.assertNotIn("玩家2", out.wealth_broadcast[0])

    def test_caught_corruption_is_not_broadcast_but_the_reporters_share_is(self):
        """被举报查实的人贪污那项记 0；但举报人分到的赃款算进传闻——照样有传闻，点举报人。"""
        caught = player(1, rank=0)
        reporter = player(2, rank=0)
        out = resolve(
            [caught, reporter],
            {1: Action(Card.CORRUPT), 2: Action(Card.REPORT, 1)},
            script=[corrupt_roll(10)],
        )
        self.assertGreater(out.outcomes[1].corrupt_amount, 0)   # 确实贪了
        self.assertGreater(out.outcomes[1].money_confiscated, 0)  # 但被没收了
        self.assertTrue(out.outcomes[1].report_effective)
        self.assertGreater(out.outcomes[2].money_from_reports, 0)
        self.assertEqual(out.wealth_top_ids, [2])
        self.assertIn("玩家2", out.wealth_broadcast[0])

    def test_storm_victim_is_not_broadcast_but_the_survivor_is(self):
        """榜首被反腐风暴查办了，贪污那项记 0，广播改播躲过一劫的那个。"""
        big = player(1, rank=3)    # 省级，贪一笔就够大案线
        small = player(2, rank=0)  # 基层，风暴只查前 1/3，轮不到他
        out = resolve(
            [big, small],
            {1: Action(Card.CORRUPT), 2: Action(Card.CORRUPT)},
            ANTI_CORRUPTION,
            script=[corrupt_roll(10), corrupt_roll(10)],
        )
        self.assertEqual(out.outcomes[1].net_corrupt_gain, 0)
        self.assertGreater(out.outcomes[2].net_corrupt_gain, 0)
        self.assertTrue(out.outcomes[1].report_effective)
        self.assertFalse(out.outcomes[2].report_effective)
        self.assertEqual(out.wealth_top_ids, [2])

    def test_being_caught_still_depends_on_gross_not_net(self):
        """被不被抓看的是"你贪没贪"（毛收入），不是"你留住多少"。"""
        target = player(1, rank=3)
        a, b = player(2), player(3)
        out = resolve(
            [target, a, b],
            {1: [Action(Card.CORRUPT)], 2: [Action(Card.REPORT, 1)],
             3: [Action(Card.REPORT, 1)]},
            script=[corrupt_roll(10)],
        )
        # 第一份举报就把钱抄光了，第二份仍然算查实（他确实贪了）
        self.assertTrue(out.outcomes[1].report_effective)
        self.assertEqual(out.outcomes[1].net_corrupt_gain, 0)


# ==========================================================================
# 终局排序
# ==========================================================================


class TestSalary(unittest.TestCase):
    """按官职自动到账的合法工资。**在回合开始时就发**，不占行动位。"""

    def test_paid_at_the_start_of_the_round(self):
        people = [player(1, rank=2), player(2, rank=0)]
        paid = rules.pay_salaries(people, REAL_CFG)
        self.assertEqual(paid[1], REAL_CFG.salary(2))
        self.assertEqual(people[0].money, REAL_CFG.salary(2))
        self.assertEqual(people[1].money, REAL_CFG.salary(0))

    def test_resolution_does_not_pay_it_again(self):
        """工资在回合开头发过了，结算里绝不能再发一次。"""
        p = player(1, rank=2)
        before = p.money
        resolve([p, player(2)], {}, cfg=REAL_CFG)  # 一张牌都没打
        self.assertEqual(p.money, before)

    def test_this_round_salary_can_pay_for_a_redraw(self):
        """发在回合开头就是为了这个：刚到手的工资当轮就能拿去换牌。"""
        p = player(1, rank=0, money=0)
        rules.pay_salaries([p], REAL_CFG)
        self.assertGreaterEqual(p.money, REAL_CFG.redraw_cost(0))

    def test_scales_with_rank(self):
        pays = [REAL_CFG.salary(r) for r in range(4)]
        self.assertEqual(pays, sorted(pays))
        self.assertLess(pays[0], pays[-1])

    def test_salary_is_clean_money(self):
        """工资不算贪污：举报查不到。

        它**算**进坊间传闻的收入口径——有人贪了的那一轮，官大的人光靠工资也能上榜。
        但全场都只有工资、没人落袋脏钱的话，坊间就不传了。
        """
        earner = player(1, rank=3)
        reporter = player(2)
        rules.pay_salaries([earner, reporter], REAL_CFG)  # 回合开头先发工资
        out = resolve(
            [earner, reporter],
            {1: [Action(Card.WORK)], 2: [Action(Card.REPORT, 1)]},
            script=[work_roll(10)],
            cfg=REAL_CFG,
            salaries={1: REAL_CFG.salary(3), 2: REAL_CFG.salary(0)},
        )
        self.assertEqual(out.outcomes[1].corrupt_amount, 0)
        self.assertFalse(out.outcomes[1].report_effective)  # 查无实据
        self.assertEqual(earner.money, REAL_CFG.salary(3))
        # 全场没人贪：光拿工资不算新闻，不广播
        self.assertEqual(out.wealth_broadcast, [])
        self.assertEqual(out.wealth_top_ids, [])


class TestPresidentNeedsBoth(unittest.TestCase):
    """省级 -> 国家主席：金钱和政绩必须同时达标。"""

    TOP = DEFAULT_CONFIG.president_rank - 1

    def _p(self, **kw):
        return player(1, rank=self.TOP, **kw)

    def test_merit_alone_is_not_enough(self):
        p = self._p(merit=999, money=0)
        out = resolve([p, player(2)], {1: [Action(Card.PROMOTE_MERIT)]}, cfg=REAL_CFG)
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.NONE)
        self.assertEqual(p.rank, self.TOP)

    def test_money_alone_is_not_enough(self):
        p = self._p(merit=0, money=999)
        out = resolve([p, player(2)], {1: [Action(Card.PROMOTE_MONEY)]}, cfg=REAL_CFG)
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.NONE)
        self.assertEqual(p.rank, self.TOP)

    def test_both_together_works_and_consumes_both(self):
        tc = REAL_CFG.merit_cost(self.TOP)
        mc = REAL_CFG.money_cost(self.TOP)
        p = self._p(merit=tc + 10, money=mc + 10)
        out = resolve([p, player(2)], {1: [Action(Card.PROMOTE_ANY)]}, cfg=REAL_CFG)
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.BOTH)
        self.assertEqual(p.rank, REAL_CFG.president_rank)
        # 政绩余额照样 /5，金钱余额全留
        self.assertEqual(
            p.merit, rules.overflow_after_promotion(10, REAL_CFG, REAL_CFG.merit_overflow_divisor)
        )
        self.assertEqual(p.money, 10)

    def test_any_promotion_card_works_on_this_step(self):
        tc, mc = REAL_CFG.merit_cost(self.TOP), REAL_CFG.money_cost(self.TOP)
        for card in (Card.PROMOTE_MERIT, Card.PROMOTE_MONEY, Card.PROMOTE_ANY):
            p = self._p(merit=tc, money=mc)
            resolve([p, player(2)], {1: [Action(card)]}, cfg=REAL_CFG)
            self.assertEqual(p.rank, REAL_CFG.president_rank, f"{card} 应该也能用")

    def test_attack_still_blocks_it(self):
        tc, mc = REAL_CFG.merit_cost(self.TOP), REAL_CFG.money_cost(self.TOP)
        # 政绩升职没有退路：被攻击就是暂缓
        p = self._p(merit=tc, money=mc)
        out = resolve(
            [p, player(2)],
            {1: [Action(Card.PROMOTE_MERIT)], 2: [Action(Card.ATTACK, 1)]},
            cfg=REAL_CFG,
        )
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.NONE)
        self.assertEqual(p.rank, self.TOP)
        # 通用升职有退路：政绩被挡，改走打点，照样登顶
        p2 = self._p(merit=tc, money=mc)
        out2 = resolve(
            [p2, player(2)],
            {1: [Action(Card.PROMOTE_ANY)], 2: [Action(Card.ATTACK, 1)]},
            cfg=REAL_CFG,
        )
        self.assertEqual(out2.outcomes[1].promotion, PromotionKind.BOTH)
        self.assertEqual(p2.rank, REAL_CFG.president_rank)

    def test_tenure_cannot_carry_you_to_the_top(self):
        """熬资历最多熬到省级，最后一步必须靠双条件挣。"""
        p = self._p(tenure=REAL_CFG.tenure_required - 1, merit=0, money=0)
        out = resolve([p, player(2)], {}, cfg=REAL_CFG)
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.NONE)
        self.assertEqual(p.rank, self.TOP)
        self.assertEqual(p.tenure, REAL_CFG.tenure_required)  # 继续累积，但升不上去

    def test_lower_steps_still_take_either_one(self):
        p = player(1, rank=0, merit=REAL_CFG.merit_cost(0), money=0)
        resolve([p, player(2)], {1: [Action(Card.PROMOTE_MERIT)]}, cfg=REAL_CFG)
        self.assertEqual(p.rank, 1)


class TestBroadcastWording(unittest.TestCase):
    """公报的措辞要对得上每一种实际情况 —— 这些都是真实对局里读出来的毛病。"""

    def test_already_at_the_bottom_says_so(self):
        """降无可降的时候别播"由基层公务员降为基层公务员"。"""
        tgt = player(1, rank=0, merit=5, money=30, warnings=1)
        out = resolve(
            [tgt, player(2, rank=0)],
            {1: [Action(Card.CORRUPT, value=18)], 2: [Action(Card.REPORT, 1)]},
            cfg=REAL_CFG,
        )
        said = " ".join(out.public_messages)
        self.assertIn("再降无可降", said)
        self.assertNotIn("降为基层公务员", said)

    def test_section_16_line_needs_an_actual_promotion_attempt(self):
        """没打晋升卡就没有"晋升泡汤"这回事。"""
        tc, mc = REAL_CFG.merit_cost(0), REAL_CFG.money_cost(0)
        tgt = player(1, rank=0, merit=tc + 5, money=mc + 5)
        out = resolve(
            [tgt, player(2, rank=0), player(3, rank=0)],
            {1: [Action(Card.CORRUPT, value=18)],
             2: [Action(Card.ATTACK, 1)], 3: [Action(Card.REPORT, 1)]},
            cfg=REAL_CFG,
        )
        self.assertFalse(out.outcomes[1].promotion_card_played)
        self.assertNotIn("晋升泡汤", " ".join(out.public_messages))

    def test_many_names_read_as_a_chinese_list(self):
        """五个人别连成"甲 和 乙 和 丙 和 丁 和 戊"。"""
        target = player(1, rank=0, merit=0)
        others = [player(i, rank=0) for i in range(2, 7)]
        acts = {1: [Action(Card.WORK, value=8), Action(Card.WORK, value=8)]}
        for o in others:
            acts[o.id] = [Action(Card.ATTACK, 1)]
        out = resolve([target, *others], acts, cfg=REAL_CFG)
        line = next(m for m in out.public_messages if "功劳" in m)
        self.assertIn("、", line)
        self.assertEqual(line.count(" 和 "), 1)   # 只有最后一个用「和」


class TestAttackBroadcastIsNotSelfContradictory(unittest.TestCase):
    """公报不能一边说"未受政绩处罚"，一边说"经济问题被人揭发"。

    真实对局里出现过这三行连在一起：
        老李 遭到政治攻击，但本轮埋头工作，未受政绩处罚。
        老李 为了压事，破了一笔财。
        老李 的经济问题被人揭发。
    "没受处罚"那句是在搜账之前就发出去的——这一刀其实打得很结实。
    """

    SPARED = "未受政绩处罚"

    def _say(self, picks, merit=0, money=0):
        # 这几句措辞是 denial 模式特有的（线上已换成抢功），钉住这个模式本身
        tgt = player(1, rank=1, merit=merit, money=money)
        atk = player(2, rank=1)
        out = resolve(
            [tgt, atk], {1: picks, 2: [Action(Card.ATTACK, 1)]},
            cfg=DENIAL_ATTACK_CFG,
        )
        return " ".join(out.public_messages), out.outcomes[1]

    def test_no_spared_line_when_corruption_is_exposed(self):
        said, o = self._say(
            [Action(Card.GRAFT, value=12), Action(Card.WORK, value=6)], money=40
        )
        self.assertGreater(o.corrupt_amount, 0)
        self.assertIn("揭发", said)
        self.assertNotIn(self.SPARED, said)

    def test_spared_line_still_shows_when_the_books_are_clean(self):
        said, o = self._say([Action(Card.WORK, value=6)], merit=5)
        self.assertEqual(o.corrupt_amount, 0)
        self.assertIn(self.SPARED, said)

    def test_no_spared_line_when_the_promotion_was_actually_blocked(self):
        tc = REAL_CFG.merit_cost(1)
        said, o = self._say(
            [Action(Card.PROMOTE_MERIT), Action(Card.WORK, value=6)], merit=tc
        )
        self.assertTrue(o.merit_promotion_blocked)
        self.assertNotIn(self.SPARED, said)

    def test_no_blocked_line_when_no_promotion_was_attempted(self):
        """没打晋升卡就没有"被拦下"——别让人以为自己挨了一刀。"""
        tc = REAL_CFG.merit_cost(1)
        said, _ = self._say([Action(Card.WORK, value=6)], merit=tc)
        self.assertNotIn("政绩晋升受阻", said)

    def test_the_wipe_line_is_not_followed_by_a_weaker_duplicate(self):
        tc = REAL_CFG.merit_cost(1)
        said, o = self._say(
            [Action(Card.PROMOTE_MERIT), Action(Card.WORK, value=6)], merit=tc
        )
        self.assertGreater(o.merit_wiped_by_attack, 0)
        self.assertIn("付诸东流", said)
        self.assertNotIn("政绩晋升受阻", said)  # 同一件事不说两遍

    def test_an_attack_is_never_silent(self):
        """每种组合都至少要有一句公报，不能让人挨了打却看不到任何说明。"""
        tc = REAL_CFG.merit_cost(1)
        cases = [
            ("干活+账目干净", [Action(Card.WORK, value=6)], 5, 0),
            ("没干活+有政绩", [Action(Card.PROMOTE_MERIT)], 10, 0),
            ("没干活+零政绩", [], 0, 0),
            ("贪污被搜出", [Action(Card.CORRUPT, value=15)], 0, 40),
            ("晋升被拦", [Action(Card.PROMOTE_MERIT), Action(Card.WORK, value=6)], tc, 0),
        ]
        for label, picks, merit, money in cases:
            said, _ = self._say(picks, merit=merit, money=money)
            self.assertTrue(said.strip(), f"{label}: 一句公报都没有")


class TestDirtyMoneyCannotBeSpentSameRound(unittest.TestCase):
    """这一轮刚贪来的钱，当轮花不出去——得先熬过举报才算落袋。

    少了这条，"贪一笔立刻洗成官职"就能躲开没收：举报人查实了却分不到钱，
    而且被举报的人升一级又被降一级、官职净变化为零，等于举报白打。
    """

    @staticmethod
    def _round(picks, reported=True, money=0):
        tgt = player(1, rank=2, money=money, merit=0)
        me = player(2, rank=2)
        acts = {1: picks}
        if reported:
            acts[2] = [Action(Card.REPORT, 1)]
        out = resolve([tgt, me], acts, cfg=REAL_CFG)
        return tgt, out.outcomes[1], out.outcomes[2]

    def test_laundering_into_a_promotion_no_longer_shields_the_loot(self):
        tgt, o, mine = self._round(
            [Action(Card.CORRUPT, value=15), Action(Card.PROMOTE_MONEY)]
        )
        self.assertEqual(o.promotion, PromotionKind.NONE)  # 晋升卡没兑现
        self.assertEqual(o.money_confiscated, o.corrupt_amount)  # 赃款一分不少地离开他
        # 举报人拿一半，再扣跑腿费
        self.assertEqual(mine.money_from_reports, o.corrupt_amount // 2 - REAL_CFG.report_reward_fee)
        self.assertEqual(tgt.rank, 2)  # 没升上去（也还没到降级线）
        self.assertEqual(tgt.warnings, 1)

    def test_honest_play_is_unaffected(self):
        """没人举报的话，同样的出牌顺序照常兑现，不该有隐形惩罚。"""
        tgt, o, _ = self._round(
            [Action(Card.CORRUPT, value=15), Action(Card.PROMOTE_MONEY)], reported=False
        )
        self.assertEqual(o.promotion, PromotionKind.MONEY)
        self.assertEqual(tgt.rank, 3)

    def test_being_reported_freezes_the_promotion_whatever_the_order(self):
        """停职待查：被举报又确实贪了，晋升卡排在贪污前面也一样冻结。

        以前把晋升卡排前面、花上一轮的旧钱就能躲过去——现在躲不掉了。
        """
        tgt, o, _ = self._round(
            [Action(Card.PROMOTE_MONEY), Action(Card.CORRUPT, value=8)], money=30
        )
        self.assertTrue(o.promotion_frozen_by_report)
        self.assertEqual(o.promotion, PromotionKind.NONE)

    def test_promoting_before_corrupting_works_when_nobody_reports(self):
        """没人举报就照常兑现——冻结只针对"真会被查实"的人。"""
        tgt, o, _ = self._round(
            [Action(Card.PROMOTE_MONEY), Action(Card.CORRUPT, value=8)],
            money=30, reported=False,
        )
        self.assertFalse(o.promotion_frozen_by_report)
        self.assertEqual(o.promotion, PromotionKind.MONEY)

    def test_a_merit_promotion_does_not_wait_for_anything(self):
        """"刚贪的钱当轮花不出去"只管花钱那条路 —— 政绩升职一分钱不用出。"""
        tgt = player(1, rank=1, money=0, merit=REAL_CFG.merit_cost(1))
        me = player(2, rank=1)
        out = resolve(
            [tgt, me],
            {1: [Action(Card.GRAFT, value=12), Action(Card.PROMOTE_MERIT)],
             2: [Action(Card.REPORT, 1)]},
            cfg=REAL_CFG,
        )
        self.assertTrue(out.outcomes[1].report_effective)  # 贪污照样查实
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.MERIT)  # 但升职照升

    def test_the_broadcast_does_not_claim_loot_that_was_not_there(self):
        """抄无可抄的时候别再播"赃款被没收"。"""
        tgt = player(1, rank=2, money=0, merit=0)
        me = player(2, rank=2)
        out = resolve(
            [tgt, me],
            {1: [Action(Card.PROMOTE_MONEY), Action(Card.CORRUPT, value=15)],
             2: [Action(Card.REPORT, 1)]},
            cfg=REAL_CFG,
        )
        o = out.outcomes[1]
        if o.money_confiscated == 0 and o.report_effective:
            said = " ".join(out.public_messages)
            self.assertIn("无余财可抄", said)
            self.assertNotIn("赃款被没收", said)


class TestEveryPromotionDecaysMerit(unittest.TestCase):
    """升一级之后政绩一律 /5——不管是政绩升、贿赂升，还是熬工龄熬上去的。

    不这么做的话，"攒政绩 + 花钱升职"能把政绩原封不动地带过一级：
    等于白拿一级，贿赂升职会严格优于政绩升职。
    """

    def test_money_promotion_still_decays_merit(self):
        p = player(1, rank=1, money=REAL_CFG.money_cost(1), merit=27)
        rules.apply_money_promotion(p, REAL_CFG)
        self.assertEqual(p.rank, 2)
        self.assertEqual(p.merit, math.ceil(27 / REAL_CFG.merit_overflow_divisor))

    def test_money_promotion_does_not_charge_merit(self):
        """衰减不等于扣门槛：政绩没被当成本花掉，只是缩水。"""
        tc = REAL_CFG.merit_cost(1)
        p = player(1, rank=1, money=REAL_CFG.money_cost(1), merit=tc + 20)
        rules.apply_money_promotion(p, REAL_CFG)
        # 如果错按"先扣门槛再 /5"算，结果会是 ceil(20/5)=4
        self.assertEqual(p.merit, math.ceil((tc + 20) / REAL_CFG.merit_overflow_divisor))

    def test_merit_promotion_charges_then_decays(self):
        tc = REAL_CFG.merit_cost(0)
        p = player(1, rank=0, money=99, merit=tc + 20)
        rules.apply_merit_promotion(p, REAL_CFG)
        self.assertEqual(p.merit, math.ceil(20 / REAL_CFG.merit_overflow_divisor))
        self.assertEqual(p.money, 99)  # 没花钱就不动钱

    def test_tenure_promotion_decays_merit_too(self):
        """熬工龄也是升职，不能靠"不打晋升卡"白拿一级还保住政绩。"""
        p = player(1, rank=0, money=0, merit=40, tenure=REAL_CFG.tenure_required - 1)
        out = resolve([p, player(2)], {1: []}, cfg=REAL_CFG)
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.TENURE)
        self.assertEqual(p.rank, 1)
        self.assertEqual(p.merit, math.ceil(40 / REAL_CFG.merit_overflow_divisor))

    def test_bribing_up_is_no_longer_strictly_better(self):
        """同样的政绩存量，贿赂升职不该比政绩升职多留下政绩。"""
        tc = REAL_CFG.merit_cost(1)
        by_merit = player(1, rank=1, money=REAL_CFG.money_cost(1), merit=tc)
        by_money = player(2, rank=1, money=REAL_CFG.money_cost(1), merit=tc)
        rules.apply_merit_promotion(by_merit, REAL_CFG)
        rules.apply_money_promotion(by_money, REAL_CFG)
        self.assertEqual(by_merit.rank, by_money.rank)
        self.assertLessEqual(by_money.merit, tc // 2)  # 不再原封不动地留着

    def test_the_switch_restores_the_old_behaviour(self):
        cfg = dataclasses.replace(REAL_CFG, promotion_always_decays_merit=False)
        p = player(1, rank=1, money=cfg.money_cost(1), merit=27)
        rules.apply_money_promotion(p, cfg)
        self.assertEqual(p.merit, 27)


class TestPlayerChosenOrder(unittest.TestCase):
    """结算严格照玩家选牌的顺序走。

    先升职后干活 -> 产出吃到新官职的倍率，也不会被晋升的 /5 砍掉；
    先干活后升职 -> 产出按旧倍率算，多出来的那点还要被 /5 砍。
    该怎么排是玩家自己的决策，服务端不替他重排。
    """

    def test_promotion_first_then_work(self):
        tc = REAL_CFG.merit_cost(0)
        p = player(1, rank=0, merit=tc, money=0)
        out = resolve(
            [p, player(2)],
            {1: [Action(Card.PROMOTE_MERIT), Action(Card.WORK, value=6)]},
            cfg=REAL_CFG,
        )
        o = out.outcomes[1]
        self.assertTrue(o.promoted_before_production)
        self.assertEqual(p.rank, 1)
        # 干活发生在升职之后，吃到县级 x1.5：6 x 1.5 = 9，而且没被 /5 砍
        self.assertEqual(o.merit_gained, 9)
        self.assertEqual(p.merit, 9)

    def test_work_first_then_promotion_is_taxed(self):
        """同样两张牌，反过来排就要吃亏——这是玩家自己的选择。"""
        tc = REAL_CFG.merit_cost(0)
        p = player(1, rank=0, merit=tc, money=0)
        out = resolve(
            [p, player(2)],
            {1: [Action(Card.WORK, value=6), Action(Card.PROMOTE_MERIT)]},
            cfg=REAL_CFG,
        )
        o = out.outcomes[1]
        self.assertFalse(o.promoted_before_production)
        self.assertEqual(p.rank, 1)
        self.assertEqual(o.merit_gained, 6)          # 按基层 x1 算
        self.assertEqual(p.merit, math.ceil(6 / 5))  # 剩下的还被 /5 砍了

    def test_not_yet_eligible_has_to_work_first(self):
        tc = REAL_CFG.merit_cost(0)
        p = player(1, rank=0, merit=tc - 5, money=0)
        out = resolve(
            [p, player(2)],
            {1: [Action(Card.WORK, value=8), Action(Card.PROMOTE_MERIT)]},
            cfg=REAL_CFG,
        )
        o = out.outcomes[1]
        self.assertEqual(p.rank, 1)       # 攒够了，升上去了
        self.assertEqual(o.merit_gained, 8)

    def test_not_yet_eligible_promoting_first_just_wastes_the_card(self):
        """还没够门槛就把晋升卡排前面，那张牌就白打了——不会帮你补算。"""
        tc = REAL_CFG.merit_cost(0)
        p = player(1, rank=0, merit=tc - 5, money=0)
        out = resolve(
            [p, player(2)],
            {1: [Action(Card.PROMOTE_MERIT), Action(Card.WORK, value=8)]},
            cfg=REAL_CFG,
        )
        self.assertEqual(p.rank, 0)
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.NONE)

    def test_promotion_first_also_boosts_corruption(self):
        mc = REAL_CFG.money_cost(0)
        p = player(1, rank=0, money=mc, merit=0)
        out = resolve(
            [p, player(2)],
            {1: [Action(Card.PROMOTE_MONEY), Action(Card.CORRUPT, value=15)]},
            cfg=REAL_CFG,
        )
        self.assertTrue(out.outcomes[1].promoted_before_production)
        self.assertEqual(p.rank, 1)
        # 县级 x1.5：15 x 1.5 = 22，而不是基层的 15
        self.assertEqual(out.outcomes[1].corrupt_amount, 22)

    def test_buying_rank_first_does_not_dodge_the_storm(self):
        """把晋升卡排到贪污**前面**曾经能完全躲开反腐风暴。

        规则速查里白纸黑字写着"把晋升卡排到贪污前面也躲不掉"，但引擎做不到：
        打牌那一刻还没贪、也没人举报过他，官就当场升了；等风暴按全场贪污额
        点名时，晋升早已兑现。这等于「贿赂升职 <- 举报」这条克制关系废了一半。
        """
        storm = rules.event_by_id("ANTI_CORRUPTION", REAL_CFG)
        mc = REAL_CFG.money_cost(0)
        for label, order in (
            ("买官在前", [Action(Card.PROMOTE_MONEY), Action(Card.CORRUPT, value=15)]),
            ("买官在后", [Action(Card.CORRUPT, value=15), Action(Card.PROMOTE_MONEY)]),
        ):
            p = player(1, rank=0, money=mc)
            out = resolve([p, player(2)], {1: order}, event=storm, cfg=REAL_CFG)
            o = out.outcomes[1]
            self.assertTrue(o.report_effective, label)
            self.assertEqual(o.promotion, PromotionKind.NONE, f"{label}：官不该升成")
            self.assertEqual(p.rank, 0, f"{label}：官职该退回去")
            self.assertEqual(p.money, 0, f"{label}：钱也要不回来")

    def test_the_reporter_is_paid_the_same_whichever_order_he_used(self):
        """撤回之后举报人分到的钱不能因为对方的出牌顺序而变少。"""
        mc = REAL_CFG.money_cost(0)

        def payout(order):
            p = player(1, rank=0, money=mc)
            r = player(2)
            resolve([p, r], {1: order, 2: [Action(Card.REPORT, 1)]}, cfg=REAL_CFG)
            return r.money

        self.assertEqual(
            payout([Action(Card.PROMOTE_MONEY), Action(Card.CORRUPT, value=15)]),
            payout([Action(Card.CORRUPT, value=15), Action(Card.PROMOTE_MONEY)]),
        )

    def test_a_bought_presidency_can_still_be_taken_back(self):
        """最高风险的一种：当轮买到主席，又当轮被查。"""
        storm = rules.event_by_id("ANTI_CORRUPTION", REAL_CFG)
        top = REAL_CFG.president_rank - 1
        p = player(1, rank=top,
                   money=REAL_CFG.money_cost(top),
                   merit=REAL_CFG.merit_cost(top))
        out = resolve(
            [p, player(2)],
            {1: [Action(Card.PROMOTE_MONEY), Action(Card.CORRUPT, value=15)]},
            event=storm, cfg=REAL_CFG,
        )
        self.assertTrue(out.outcomes[1].report_effective)
        self.assertEqual(p.rank, top, "主席之位该被收回")
        self.assertEqual(out.presidents, [], "不能当上主席还赢下整局")

    def test_the_promotion_first_combo_still_works_when_nobody_catches_you(self):
        """修漏洞不能把「先买官、再按新倍率去贪」这个组合技一起毁掉。"""
        mc = REAL_CFG.money_cost(0)
        p = player(1, rank=0, money=mc)
        out = resolve(
            [p, player(2)],
            {1: [Action(Card.PROMOTE_MONEY), Action(Card.CORRUPT, value=15)]},
            cfg=REAL_CFG,
        )
        self.assertEqual(p.rank, 1)
        self.assertEqual(out.outcomes[1].corrupt_amount, 22)  # 县级 x1.5
        self.assertTrue(out.outcomes[1].promoted_before_production)

    def test_a_merit_promotion_is_never_revoked_by_a_report(self):
        """克制矩阵：政绩升职只怕政治攻击，举报碰不到它。"""
        storm = rules.event_by_id("ANTI_CORRUPTION", REAL_CFG)
        p = player(1, rank=0, merit=REAL_CFG.merit_cost(0))
        out = resolve(
            [p, player(2)],
            {1: [Action(Card.PROMOTE_MERIT), Action(Card.CORRUPT, value=15)]},
            event=storm, cfg=REAL_CFG,
        )
        self.assertTrue(out.outcomes[1].report_effective, "照样被查实")
        self.assertEqual(p.rank, 1, "但政绩升上去的官动不了")

    def test_corrupting_first_stays_at_the_old_rate(self):
        mc = REAL_CFG.money_cost(0)
        p = player(1, rank=0, money=mc, merit=0)
        out = resolve(
            [p, player(2)],
            {1: [Action(Card.CORRUPT, value=15), Action(Card.PROMOTE_MONEY)]},
            cfg=REAL_CFG,
        )
        self.assertFalse(out.outcomes[1].promoted_before_production)
        self.assertEqual(p.rank, 1)
        self.assertEqual(out.outcomes[1].corrupt_amount, 15)

    def test_still_only_one_promotion_per_round(self):
        p = player(1, rank=0, merit=REAL_CFG.merit_cost(0) + REAL_CFG.merit_cost(1) + 50)
        out = resolve(
            [p, player(2)],
            {1: [Action(Card.PROMOTE_MERIT), Action(Card.PROMOTE_MERIT)]},
            cfg=REAL_CFG,
        )
        self.assertTrue(out.outcomes[1].promoted_before_production)
        self.assertEqual(p.rank, 1)

    def test_a_promotion_after_report_waits_for_the_loot(self):
        """把晋升卡排在举报后面，就是想花那笔赃款——那就等赃款到账再结算。"""
        mc = REAL_CFG.money_cost(0)
        # 举报人只拿没收额的一半，所以要抓一条大鱼才够升一级
        target = player(1, rank=2, money=40, merit=0)
        me = player(2, rank=0, money=0, merit=0)
        out = resolve(
            [target, me],
            {1: [Action(Card.CORRUPT, value=17)],
             2: [Action(Card.REPORT, 1), Action(Card.PROMOTE_MONEY)]},
            cfg=REAL_CFG,
        )
        self.assertTrue(out.outcomes[1].report_effective)
        self.assertGreaterEqual(out.outcomes[2].money_from_reports, mc)
        self.assertEqual(me.rank, 1)  # 用刚抄来的钱当场升官

    def test_a_promotion_before_report_cannot_spend_the_loot(self):
        """反过来排就花不到——举报的赃款是结算末尾才到账的。"""
        target = player(1, rank=0, money=0, merit=0)
        me = player(2, rank=0, money=0, merit=0)
        out = resolve(
            [target, me],
            {1: [Action(Card.CORRUPT, value=17)],
             2: [Action(Card.PROMOTE_MONEY), Action(Card.REPORT, 1)]},
            cfg=REAL_CFG,
        )
        self.assertGreater(out.outcomes[2].money_from_reports, 0)
        self.assertEqual(me.rank, 0)


class TestAttackedPromotionFallback(unittest.TestCase):
    """两样都够 + 被政治攻击时的结算顺序。

    1. 先结算晋升，优先政绩
    2. 被攻击挡下 -> **暂缓升职，政绩一点不掉**（不再清零）
    3. 手里是"通用升职"的话，改走金钱这条路
    4. 最后才结算 WORK —— 按新官职倍率算，但会被抢功抽走一半
    """

    def _setup(self, card):
        tc, mc = REAL_CFG.merit_cost(0), REAL_CFG.money_cost(0)
        me = player(1, rank=0, merit=tc + 8, money=mc + 5)
        out = resolve(
            [me, player(2)],
            {1: [Action(card), Action(Card.WORK, value=8)], 2: [Action(Card.ATTACK, 1)]},
            cfg=REAL_CFG,
        )
        return me, out.outcomes[1], tc, mc

    def test_generic_card_falls_back_to_money(self):
        me, o, tc, mc = self._setup(Card.PROMOTE_ANY)
        self.assertEqual(o.promotion, PromotionKind.MONEY)  # 改走金钱升上去了
        self.assertEqual(me.rank, 1)
        self.assertTrue(o.promoted_before_production)

    def test_the_block_no_longer_wipes_merit(self):
        """暂缓升职就只是暂缓：政绩一点不掉。"""
        me, o, tc, mc = self._setup(Card.PROMOTE_ANY)
        self.assertEqual(o.merit_wiped_by_attack, 0)

    def test_the_work_lands_at_the_new_rank_minus_the_steal(self):
        me, o, tc, mc = self._setup(Card.PROMOTE_ANY)
        # 8 x 1.5（县级）= 12；抢功抽走一半 -> 只剩 6 落到自己头上
        self.assertEqual(o.merit_gained, 12)
        self.assertEqual(o.merit_stolen_by_attackers, 12 // 2)

    def test_merit_only_card_has_no_fallback(self):
        me, o, tc, mc = self._setup(Card.PROMOTE_MERIT)
        self.assertEqual(o.merit_wiped_by_attack, 0)  # 政绩保留
        self.assertEqual(o.promotion, PromotionKind.NONE)
        self.assertEqual(me.rank, 0)
        self.assertEqual(o.merit_gained, 8)  # 没升成，WORK 只能按基层 x1 算

    def test_money_is_actually_spent_on_the_fallback(self):
        me, o, tc, mc = self._setup(Card.PROMOTE_ANY)
        # 起始 mc+5，扣掉门槛 mc，余额不衰减（工资是回合开头发的，不在结算里）
        self.assertEqual(me.money, 5)

    def test_top_step_merit_card_has_no_fallback(self):
        """最后一步打政绩升职被攻击拦下 = 暂缓，而且**零惩罚**。"""
        top = REAL_CFG.president_rank - 1
        tc, mc = REAL_CFG.merit_cost(top), REAL_CFG.money_cost(top)
        me = player(1, rank=top, merit=tc + 10, money=mc + 20)
        out = resolve(
            [me, player(2)],
            {1: [Action(Card.PROMOTE_MERIT)], 2: [Action(Card.ATTACK, 1)]},
            cfg=REAL_CFG,
        )
        o = out.outcomes[1]
        self.assertEqual(o.promotion, PromotionKind.NONE)
        self.assertEqual(me.rank, top)
        self.assertEqual(me.money, mc + 20)   # 钱一分没花
        self.assertEqual(me.merit, tc + 10)   # 政绩一分没掉
        self.assertEqual(o.bribe_lost, 0)
        self.assertEqual(o.warnings_issued, 0)


class TestPathEconomics(unittest.TestCase):
    """两条路线的性价比：贪一笔就够升一级，干活得干三四次。别把这个关系调飞了。"""

    def test_one_corruption_funds_one_promotion(self):
        expected = 15  # CORRUPT 牌面期望
        for r in range(REAL_CFG.president_rank):
            haul = rules.corrupt_money(expected, r, CALM, REAL_CFG)
            with_pay = haul + REAL_CFG.salary(r)
            self.assertGreaterEqual(
                with_pay, REAL_CFG.money_cost(r),
                f"{REAL_CFG.rank_name(r)}：工资+贪一笔应该够升一级",
            )

    def test_merit_path_takes_several_rounds(self):
        """政绩慢是设计出来的——快的那条要担风险。"""
        expected = 6  # WORK 牌面期望
        for r in range(REAL_CFG.president_rank):
            gain = rules.work_merit(expected, r, CALM, REAL_CFG)
            needed = -(-REAL_CFG.merit_cost(r) // gain)
            self.assertGreaterEqual(needed, 3, f"{REAL_CFG.rank_name(r)}：政绩不该一两下就攒够")
            self.assertLessEqual(needed, 5)

    def test_money_thresholds_track_the_rank_multiplier(self):
        """门槛必须跟着官职倍率走，不然越往上越够不着（踩过这个坑）。"""
        costs = REAL_CFG.promotion_money_costs
        for r in range(1, len(costs)):
            got = costs[r] / costs[0]
            want = float(REAL_CFG.money_multiplier(r) / REAL_CFG.money_multiplier(0))
            self.assertAlmostEqual(got, want, delta=0.12,
                                   msg=f"{REAL_CFG.rank_name(r)} 的门槛和收益脱节了")


class TestStealWork(unittest.TestCase):
    """抢功：没收目标本轮产出的一半，所有抢功的人平分，目标保留另一半。

    这是「鹬蚌相争，渔翁得利」的支点。整个结构靠两条：
      * 打正在生产的人 -> 抢得到 -> 出手的人自己也前进（#2 能反超 #1）
      * 打正在干扰别人的人 -> 他没产出 -> 白打（#1#2 互抢 -> #3 得利）
    """

    def _round(self, actions, players=None):
        ps = players or [player(i, rank=0) for i in (1, 2, 3)]
        out = resolve(ps, actions, cfg=REAL_CFG)
        return ps, out

    def test_target_keeps_half_and_the_attacker_takes_half(self):
        ps, out = self._round({
            1: [Action(Card.WORK, value=6), Action(Card.WORK, value=6)],
            2: [Action(Card.ATTACK, 1)],
        })
        o1, o2 = out.outcomes[1], out.outcomes[2]
        self.assertEqual(o1.merit_gained, 12)
        self.assertEqual(o1.merit_stolen_by_attackers, 6)  # 正好一半
        self.assertEqual(o2.merit_from_attacks, 6)
        self.assertEqual(ps[0].merit, 6)  # 保留另一半
        self.assertEqual(ps[1].merit, 6)

    def test_it_is_a_transfer_not_destruction(self):
        """被抢走多少，抢功的人就拿到多少 —— 账要平。"""
        ps, out = self._round({
            1: [Action(Card.WORK, value=8), Action(Card.WORK, value=8)],
            2: [Action(Card.ATTACK, 1)],
        })
        self.assertEqual(
            out.outcomes[1].merit_stolen_by_attackers,
            out.outcomes[2].merit_from_attacks,
        )

    def test_multiple_attackers_split_that_same_half(self):
        """两个人一起抢，目标还是只掉一半，两人平分 —— 不是各抢一半。"""
        ps, out = self._round({
            1: [Action(Card.WORK, value=6), Action(Card.WORK, value=6)],
            2: [Action(Card.ATTACK, 1)],
            3: [Action(Card.ATTACK, 1)],
        })
        self.assertEqual(out.outcomes[1].merit_stolen_by_attackers, 6)
        self.assertEqual(out.outcomes[2].merit_from_attacks, 3)
        self.assertEqual(out.outcomes[3].merit_from_attacks, 3)

    def test_attacking_a_non_producer_steals_nothing(self):
        """抢不到东西 —— 这是「互相搞 -> 第三名得利」的机制来源。

        （无功可抢时还是会按官职扣他一笔，但那是处罚、不进攻击者口袋，
        所以互攻仍然是两败俱伤，见 TestIdleTargetPenalty。）
        """
        ps, out = self._round({
            1: [Action(Card.ATTACK, 2)],
            2: [Action(Card.ATTACK, 1)],
        })
        self.assertEqual(out.outcomes[1].merit_from_attacks, 0)
        self.assertEqual(out.outcomes[2].merit_from_attacks, 0)
        self.assertEqual(out.outcomes[1].merit_stolen_by_attackers, 0)

    def test_corruption_is_not_stealable(self):
        """贪污产出的是钱不是政绩，抢功抢不到 —— 那是举报的活。"""
        ps, out = self._round({
            1: [Action(Card.CORRUPT, value=18)],
            2: [Action(Card.ATTACK, 1)],
        })
        self.assertEqual(out.outcomes[2].merit_from_attacks, 0)
        self.assertEqual(out.outcomes[1].money_gained, 18)

    def test_the_triangle_holds_end_to_end(self):
        """两个场景一起验：#2 反超 #1；#1#2 互抢则 #3 领先两人。"""
        ps, _ = self._round({
            1: [Action(Card.WORK, value=6), Action(Card.WORK, value=6)],
            2: [Action(Card.ATTACK, 1), Action(Card.WORK, value=6)],
            3: [Action(Card.WORK, value=6), Action(Card.WORK, value=6)],
        })
        self.assertGreater(ps[1].merit, ps[0].merit, "#2 搞了 #1，应该反超")

        ps, _ = self._round({
            1: [Action(Card.ATTACK, 2), Action(Card.WORK, value=6)],
            2: [Action(Card.ATTACK, 1), Action(Card.WORK, value=6)],
            3: [Action(Card.WORK, value=6), Action(Card.WORK, value=6)],
        })
        self.assertGreater(ps[2].merit, ps[0].merit, "#1#2 互抢，渔翁应该领先")
        self.assertGreater(ps[2].merit, ps[1].merit)

    def test_the_attacker_is_named_in_public(self):
        """明枪：被抢的人要知道该报复谁。（匿名举报是暗箭，不在此列。）"""
        ps, out = self._round({
            1: [Action(Card.WORK, value=6), Action(Card.WORK, value=6)],
            2: [Action(Card.ATTACK, 1)],
        })
        said = " ".join(out.public_messages)
        self.assertIn(ps[1].name, said)
        self.assertEqual(out.outcomes[1].public_facts()["attacked_by"], [2])


class TestIdleTargetPenalty(unittest.TestCase):
    """目标本轮没干活就没功劳可抢，改成按**官职**扣他一笔政绩。"""

    def _hit(self, target_rank=0, target_merit=30, target_picks=None):
        tgt = player(1, rank=target_rank, merit=target_merit)
        atk = player(2, rank=0)
        out = resolve(
            [tgt, atk],
            {1: target_picks or [], 2: [Action(Card.ATTACK, 1)]},
            cfg=REAL_CFG,
        )
        return tgt, out.outcomes[1], out.outcomes[2]

    def test_penalty_scales_with_the_target_rank(self):
        """官越大扣越多：基层 4 / 县级 6 / 市级 8 / 省级 10。"""
        losses = []
        for r in range(REAL_CFG.president_rank):
            _, o, _ = self._hit(target_rank=r, target_merit=99)
            losses.append(o.attack_merit_loss)
        self.assertEqual(losses, sorted(losses))
        self.assertLess(losses[0], losses[-1])
        for r, loss in enumerate(losses):
            self.assertEqual(
                loss, rules.work_merit(REAL_CFG.attack_merit_penalty, r, CALM, REAL_CFG)
            )

    def test_the_fine_is_not_transferred(self):
        """罚款是处罚不是转移 —— 不然互攻就不再两败俱伤了。

        攻击者只拿到 ATTACK_HAT_REWARD 那一小笔"记功"（按自己的官职），和罚了多少无关。
        """
        _, o_t, o_a = self._hit()
        self.assertGreater(o_t.attack_merit_loss, 0)
        self.assertEqual(o_t.merit_stolen_by_attackers, 0)
        reward = rules.work_merit(REAL_CFG.attack_hat_reward, 0, None, REAL_CFG)
        self.assertEqual(o_a.merit_from_attacks, reward)
        self.assertLess(o_a.merit_from_attacks, o_t.attack_merit_loss)

    def test_it_cannot_push_merit_below_zero(self):
        tgt, o, _ = self._hit(target_merit=1)
        self.assertEqual(o.attack_merit_loss, 1)
        self.assertEqual(tgt.merit, 0)

    def test_going_for_a_merit_promotion_counts_as_honest_work(self):
        """干正事 = 埋头工作 **或者** 凭政绩升职，不吃这笔罚款。

        拦下政绩晋升说好了是"暂缓、政绩不掉"，不能用罚款把清零偷偷加回来。
        """
        tc = REAL_CFG.merit_cost(0)
        tgt, o, _ = self._hit(
            target_merit=tc + 5, target_picks=[Action(Card.PROMOTE_MERIT)]
        )
        self.assertTrue(o.merit_promotion_blocked)  # 升职确实被拦了
        self.assertEqual(o.attack_merit_loss, 0)    # 但政绩一点不掉
        self.assertEqual(tgt.merit, tc + 5)

    def test_buying_the_office_is_not_honest_work(self):
        """拿钱买官不算干正事，照罚 —— 那是花钱不是干活。"""
        mc = REAL_CFG.money_cost(0)
        tgt, o, _ = self._hit(target_picks=[Action(Card.PROMOTE_MONEY)])
        tgt2 = player(1, rank=0, merit=30, money=mc + 10)
        out = resolve(
            [tgt2, player(2, rank=0)],
            {1: [Action(Card.PROMOTE_MONEY)], 2: [Action(Card.ATTACK, 1)]},
            cfg=REAL_CFG,
        )
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.MONEY)  # 官照升
        self.assertGreater(out.outcomes[1].attack_merit_loss, 0)          # 但照罚

    def test_the_exemption_survives_a_deferred_promotion(self):
        """同时被举报时晋升卡会延后到第 5a 步才结算。

        踩过：豁免判据用的是 merit_promotion_blocked，而那个标志要等延后结算
        才置位 —— 攻击那一步读到的还是 False，于是照罚，政绩白掉一笔。
        """
        tc, mc = REAL_CFG.merit_cost(0), REAL_CFG.money_cost(0)
        me = player(1, rank=0, merit=tc + 5, money=mc + 5)
        out = resolve(
            [me, player(2, rank=0), player(3, rank=0)],
            {1: [Action(Card.PROMOTE_ANY)],
             2: [Action(Card.ATTACK, 1)],
             3: [Action(Card.REPORT, 1)]},
            cfg=REAL_CFG,
        )
        self.assertEqual(out.outcomes[1].attack_merit_loss, 0)
        self.assertEqual(me.merit, tc + 5)  # 政绩保留

    def test_the_message_does_not_claim_zero_merit_when_he_has_plenty(self):
        """豁免戴帽子 ≠ 他政绩是 0，而且穿小鞋那一下明明命中了。

        真实对局里出现过：老张打了政绩升职、政绩 16、升职确实被拦下了，
        公报却说"他本轮既无功劳也无政绩，白打一场"——三处都不对。
        """
        tc = REAL_CFG.merit_cost(0)
        tgt, o, _ = self._hit(
            target_merit=tc + 1,
            target_picks=[Action(Card.PROMOTE_MERIT), Action(Card.CORRUPT, value=18)],
        )
        self.assertEqual(o.attack_merit_loss, 0)      # 豁免了帽子
        self.assertEqual(tgt.merit, tc + 1)           # 政绩一点没掉
        self.assertTrue(o.merit_promotion_blocked)    # 但穿小鞋命中了

    def test_the_whiff_line_is_suppressed_when_the_block_landed(self):
        """穿小鞋命中时不能再播"白打一场"——"暂缓升职"那句已经说明白了。"""
        tc = REAL_CFG.merit_cost(0)
        me = player(1, rank=0, merit=tc + 1, money=30)
        out = resolve(
            [me, player(2, rank=0)],
            {1: [Action(Card.PROMOTE_MERIT), Action(Card.CORRUPT, value=18)],
             2: [Action(Card.ATTACK, 1)]},
            cfg=REAL_CFG,
        )
        said = " ".join(out.public_messages)
        self.assertIn("升职暂缓", said)
        self.assertNotIn("扣了个空", said)
        self.assertNotIn("戴帽子", said)

    def test_a_clean_zero_merit_target_is_a_total_whiff(self):
        tgt, o, _ = self._hit(target_merit=0)
        self.assertEqual(o.attack_merit_loss, 0)
        self.assertEqual(tgt.merit, 0)

    def test_only_a_genuinely_zero_target_says_zero_merit(self):
        """"他政绩本来就是 0"这句只能在政绩真的是 0 时出现。"""
        zero = player(1, rank=0, merit=0, money=30)
        out = resolve(
            [zero, player(2, rank=0)],
            {1: [Action(Card.CORRUPT, value=18)], 2: [Action(Card.ATTACK, 1)]},
            cfg=REAL_CFG,
        )
        self.assertIn("扣了个空", " ".join(out.public_messages))

    def test_a_producer_is_robbed_instead_of_fined(self):
        """干了活就走抢功那条路，不该再叠一份"没干活"的处罚。"""
        tgt, o, o_a = self._hit(
            target_merit=30, target_picks=[Action(Card.WORK, value=6)]
        )
        self.assertEqual(o.merit_stolen_by_attackers, 3)  # 产出 6 的一半
        self.assertEqual(o.attack_merit_loss, 3)          # 就这 3 点，没有额外罚款
        self.assertEqual(o_a.merit_from_attacks, 3)

    def test_corrupting_counts_as_not_working(self):
        """贪污产的是钱不是政绩 —— 照样算"本轮没干活"，要吃这笔罚。"""
        tgt, o, o_a = self._hit(
            target_rank=3, target_merit=40,
            target_picks=[Action(Card.CORRUPT, value=18)],
        )
        self.assertGreater(o.attack_merit_loss, 0)
        # 攻击者只拿那一小笔记功，罚款本身不归他
        self.assertEqual(
            o_a.merit_from_attacks, rules.work_merit(REAL_CFG.attack_hat_reward, 0, None, REAL_CFG)
        )

    def test_mutual_attacks_still_let_the_builder_win(self):
        """加了罚款之后，互攻只会更惨 —— 渔翁得利这条更稳。"""
        a, b, c = player(1, merit=30), player(2, merit=30), player(3, merit=30)
        resolve(
            [a, b, c],
            {1: [Action(Card.ATTACK, 2)], 2: [Action(Card.ATTACK, 1)],
             3: [Action(Card.WORK, value=6), Action(Card.WORK, value=6)]},
            cfg=REAL_CFG,
        )
        self.assertLess(a.merit, c.merit)
        self.assertLess(b.merit, c.merit)


class TestPromotionCounterMatrix(unittest.TestCase):
    """攻击克政绩路线，举报克金钱路线；通用升职有退路，但退路也会被堵。"""

    def _go(self, card, attacked=False, reported=False, merit=0, money=0):
        me = player(1, rank=0, merit=merit, money=money)
        atk, rep = player(2, rank=0), player(3, rank=0)
        acts = {1: [Action(card)]}
        if attacked:
            acts[2] = [Action(Card.ATTACK, 1)]
        if reported:
            acts[3] = [Action(Card.REPORT, 1)]
        out = resolve([me, atk, rep], acts, cfg=REAL_CFG)
        return me, out.outcomes[1], " ".join(out.public_messages)

    TC = property(lambda self: REAL_CFG.merit_cost(0))
    MC = property(lambda self: REAL_CFG.money_cost(0))

    def test_merit_promotion_is_delayed_by_attack_with_no_merit_loss(self):
        me, o, said = self._go(Card.PROMOTE_MERIT, attacked=True, merit=self.TC + 5)
        self.assertEqual(o.promotion, PromotionKind.NONE)
        self.assertEqual(me.merit, self.TC + 5)  # 一点不掉，只是暂缓
        self.assertEqual(o.merit_wiped_by_attack, 0)
        self.assertIn("升职暂缓", said)

    def test_merit_promotion_ignores_reports(self):
        me, o, _ = self._go(Card.PROMOTE_MERIT, reported=True, merit=self.TC + 5)
        self.assertEqual(o.promotion, PromotionKind.MERIT)

    def test_bribery_ignores_attacks(self):
        me, o, _ = self._go(Card.PROMOTE_MONEY, attacked=True, money=self.MC + 5)
        self.assertEqual(o.promotion, PromotionKind.MONEY)

    def test_bribery_is_killed_by_reports_and_the_money_is_gone(self):
        me, o, _ = self._go(Card.PROMOTE_MONEY, reported=True, money=self.MC + 5)
        self.assertEqual(o.promotion, PromotionKind.NONE)
        self.assertEqual(o.bribe_lost, self.MC)

    def test_generic_prefers_merit(self):
        me, o, _ = self._go(Card.PROMOTE_ANY, merit=self.TC + 5, money=self.MC + 5)
        self.assertEqual(o.promotion, PromotionKind.MERIT)
        self.assertEqual(me.money, self.MC + 5)  # 一分钱没花

    def test_generic_falls_back_to_bribery_when_attacked(self):
        me, o, _ = self._go(
            Card.PROMOTE_ANY, attacked=True, merit=self.TC + 5, money=self.MC + 5
        )
        self.assertEqual(o.promotion, PromotionKind.MONEY)

    def test_generic_still_uses_merit_when_only_reported(self):
        me, o, _ = self._go(
            Card.PROMOTE_ANY, reported=True, merit=self.TC + 5, money=self.MC + 5
        )
        self.assertEqual(o.promotion, PromotionKind.MERIT)

    def test_both_counters_together_kill_it(self):
        """攻击堵政绩、举报堵金钱 -> 升职失败，钱损失，政绩保留。"""
        me, o, _ = self._go(
            Card.PROMOTE_ANY, attacked=True, reported=True,
            merit=self.TC + 5, money=self.MC + 5,
        )
        self.assertEqual(o.promotion, PromotionKind.NONE)
        self.assertEqual(o.bribe_lost, self.MC)
        self.assertEqual(me.merit, self.TC + 5)  # 政绩保留
        self.assertEqual(me.money, self.MC + 5 - self.MC)


class TestDeckIsFinite(unittest.TestCase):
    """牌库里写着几张就是几张——不是抽样权重。

    以前是按权重**有放回**抽样，26.7% 的手牌会出现 3 张以上同名牌，
    配置里写 "PROMOTE_MERIT: 2" 却能摸到 3 张政绩升职。
    """

    def test_a_hand_never_exceeds_the_configured_count(self):
        rng = random.Random(20260925)
        for _ in range(4000):
            hand = Counter(d.card.value for d in rules.deal_hand(rng, REAL_CFG))
            for name, count in REAL_CFG.card_deal_distribution.items():
                self.assertLessEqual(
                    hand.get(name, 0), count,
                    f"{name} 配置 {count} 张，却发出了 {hand.get(name)} 张",
                )

    def test_the_hand_is_the_right_size_and_all_cards_are_legal(self):
        rng = random.Random(1)
        legal = set(REAL_CFG.card_deal_distribution)
        for _ in range(200):
            hand = rules.deal_hand(rng, REAL_CFG)
            self.assertEqual(len(hand), REAL_CFG.hand_size)
            self.assertTrue(all(d.card.value in legal for d in hand))

    def test_every_card_type_still_shows_up(self):
        """不放回也不能把某张牌抽没了。"""
        rng = random.Random(7)
        seen = Counter()
        for _ in range(3000):
            seen.update(d.card.value for d in rules.deal_hand(rng, REAL_CFG))
        for name in REAL_CFG.card_deal_distribution:
            self.assertGreater(seen[name], 0, f"{name} 一次都没发出来过")

    def test_deck_must_be_big_enough(self):
        cfg = dataclasses.replace(REAL_CFG, card_deal_distribution={"WORK": 2})
        with self.assertRaises(ValueError):
            rules.deal_hand(random.Random(1), cfg)


class TestCardDistributions(unittest.TestCase):
    """守住真实配置的牌面期望值（平衡调整时别调飞了）。"""

    @staticmethod
    def mean(dist):
        return sum(v * w for v, w in dist) / sum(w for _, w in dist)

    def test_work_and_corrupt_expectations(self):
        """政绩慢而安全、金钱快而危险，这个倍差是设计出来的。"""
        work = self.mean(REAL_CFG.work_card_distribution)
        corrupt = self.mean(REAL_CFG.corrupt_card_distribution)
        graft = self.mean(REAL_CFG.graft_card_distribution)
        self.assertEqual(work, 6.0)
        self.assertEqual(corrupt, 18.0)
        self.assertEqual(graft, 10.0)
        # 贪污的期望点数就是定在干活的 3 倍上——高风险高收益的"高收益"这一半
        self.assertEqual(corrupt, work * 3)
        self.assertTrue(work < graft < corrupt)    # 以权谋私夹在中间

    def test_graft_merit_never_beats_honest_work(self):
        """以权谋私顺带的政绩要**严格少于**埋头工作，每一级都不许重叠。

        踩过：1/2 的时候两段是 4~6 和 4~8 —— 一张好的以权谋私能压过一张差的 WORK，
        而且还白送 10 块钱，看着就不对。
        """
        for r in range(REAL_CFG.president_rank):
            work = [
                rules.work_merit(v, r, CALM, REAL_CFG)
                for v, n in REAL_CFG.work_card_distribution for _ in range(n)
            ]
            graft = [
                rules.graft_merit(v, r, CALM, REAL_CFG)
                for v, n in REAL_CFG.graft_card_distribution for _ in range(n)
            ]
            self.assertLess(
                max(graft), min(work),
                f"{REAL_CFG.rank_name(r)}：以权谋私最高 {max(graft)} 政绩，"
                f"却够得着埋头工作最低的 {min(work)}",
            )

    def test_corruption_scales_with_rank_and_tracks_the_money_threshold(self):
        """贪一笔 ~= 一次升职的钱，这条比例在每一级都要成立。"""
        expected = self.mean(REAL_CFG.corrupt_card_distribution)
        for r in range(REAL_CFG.president_rank):
            haul = rules.corrupt_money(int(expected), r, CALM, REAL_CFG)
            mc = REAL_CFG.money_cost(r)
            self.assertGreater(haul / mc, 1.0, f"{REAL_CFG.rank_name(r)} 贪一笔该够升一级")
            self.assertLess(haul / mc, 1.5, f"{REAL_CFG.rank_name(r)} 贪一笔不该够升一级半")

    def test_haul_size_does_not_change_the_punishment_by_default(self):
        """默认配置下大案线是关着的：贪多贪少都只记一次警告。

        以前是"省级贪一笔必成大案、直接打回基层"，风险曲线在后期垂直起飞，
        导致后期根本没人敢贪。现在惩罚对金额一视同仁。
        """
        small = player(1, rank=0, money=0, merit=0)
        big = player(2, rank=3, money=0, merit=0)
        out = resolve(
            [small, big, player(3)],
            {1: [Action(Card.CORRUPT, value=16)],
             2: [Action(Card.CORRUPT, value=20)],
             3: [Action(Card.REPORT, 1)]},
            cfg=REAL_CFG,
        )
        big_haul = rules.corrupt_money(20, 3, CALM, REAL_CFG)
        self.assertGreater(big_haul, REAL_CFG.major_corruption_threshold)
        # 小鱼被举报了，记一次警告；金额大小不影响记几次
        self.assertEqual(out.outcomes[1].warnings_issued, 1)
        self.assertEqual(
            rules.warnings_for(DemotionKind.MAJOR, REAL_CFG),
            rules.warnings_for(DemotionKind.MINOR, REAL_CFG),
        )

    def test_the_switch_brings_the_big_case_cliff_back(self):
        cfg = dataclasses.replace(REAL_CFG, major_corruption_warnings=2)
        self.assertEqual(rules.warnings_for(DemotionKind.MINOR, cfg), 1)
        self.assertEqual(rules.warnings_for(DemotionKind.MAJOR, cfg), 2)


class TestAntiCorruptionStorm(unittest.TestCase):
    """反腐风暴：只查办本轮贪污额排前 1/3 的人。"""

    def test_picks_the_top_third_by_haul(self):
        # 6 人，配额 = ceil(6/3) = 2
        hauls = {1: 30, 2: 25, 3: 20, 4: 10, 5: 0, 6: 0}
        self.assertEqual(rules.storm_targets(hauls, CFG), [1, 2])

    def test_ties_on_the_cutoff_all_get_caught(self):
        hauls = {1: 30, 2: 25, 3: 25, 4: 25, 5: 0, 6: 0}
        self.assertEqual(rules.storm_targets(hauls, CFG), [1, 2, 3, 4])

    def test_nobody_corrupted_nobody_caught(self):
        self.assertEqual(rules.storm_targets({1: 0, 2: 0, 3: 0}, CFG), [])

    def test_fewer_corruptors_than_the_quota(self):
        """只有一个人贪，就只查他一个，不会凑数去抓清白的人。"""
        self.assertEqual(rules.storm_targets({1: 5, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}, CFG), [1])

    def test_through_resolve_round(self):
        # 三个人都贪，配额 ceil(3/3) = 1，只有贪最多的那个被查办
        big = player(1, rank=1, money=0)    # 县级 x1.5
        small = player(2, rank=0, money=0)  # 基层 x1
        clean = player(3, rank=0, money=0)
        out = resolve(
            [big, small, clean],
            {1: Action(Card.CORRUPT), 2: Action(Card.CORRUPT), 3: Action(Card.WORK)},
            ANTI_CORRUPTION,
            script=[corrupt_roll(10), corrupt_roll(10), work_roll(10)],
        )
        self.assertEqual(out.outcomes[1].corrupt_amount, 15)
        self.assertEqual(out.outcomes[2].corrupt_amount, 10)
        self.assertTrue(out.outcomes[1].report_effective)   # 贪最多，被查
        self.assertFalse(out.outcomes[2].reported)          # 贪得少，躲过一劫
        self.assertFalse(out.outcomes[3].reported)          # 没贪，不沾边

    def test_storm_loot_goes_nowhere(self):
        """风暴没有举报人，赃款直接充公，不会白送给谁。"""
        a = player(1, rank=0, money=0)
        b = player(2, rank=0, money=0)
        out = resolve(
            [a, b], {1: Action(Card.CORRUPT)}, ANTI_CORRUPTION, script=[corrupt_roll(10)]
        )
        self.assertTrue(out.outcomes[1].report_effective)
        self.assertGreater(out.outcomes[1].money_confiscated, 0)
        self.assertEqual(out.outcomes[2].money_from_reports, 0)
        self.assertEqual(b.money, 0)


class TestEventTable(unittest.TestCase):
    def test_stable_is_shelved_but_still_defined(self):
        """政治环境稳定已下架（权重 0），定义留着方便恢复。"""
        stable = next(d for d in REAL_CFG.event_definitions if d["id"] == "STABLE")
        self.assertEqual(stable["weight"], 0)
        drawn = {rules.pick_event(ScriptedRng([i]), REAL_CFG).id for i in range(200)}
        self.assertNotIn("STABLE", drawn)

    def test_every_live_event_has_an_effect_except_the_calm_one(self):
        live = [d for d in REAL_CFG.event_definitions if d["weight"] > 0]
        empty = [d["name"] for d in live if not d["effects"]]
        self.assertEqual(empty, ["风平浪静"], "空效果的事件只应该有风平浪静一张")


class TestFinalRanking(unittest.TestCase):
    """打满轮数、无人登顶时的排序：金钱 > 官职 > 政绩（FINAL_RANKING_KEYS）。

    "当上主席即时获胜"是更高一级的判定，不走这里——见 TestPresidentEndsGameImmediately。
    """

    def test_money_first(self):
        rich = player(1, money=10, merit=0, rank=0)
        senior = player(2, money=9, merit=99, rank=3)
        self.assertEqual([p.id for p in rules.final_winners([rich, senior], CFG)], [1])

    def test_rank_breaks_the_money_tie(self):
        a = player(1, money=10, merit=99, rank=1)
        b = player(2, money=10, merit=0, rank=2)
        self.assertEqual([p.id for p in rules.final_winners([a, b], CFG)], [2])

    def test_merit_breaks_the_last_tie(self):
        a = player(1, money=10, merit=5, rank=2)
        b = player(2, money=10, merit=6, rank=2)
        self.assertEqual([p.id for p in rules.final_winners([a, b], CFG)], [2])

    def test_draw(self):
        a = player(1, money=10, merit=5, rank=1)
        b = player(2, money=10, merit=5, rank=1)
        self.assertEqual({p.id for p in rules.final_winners([a, b], CFG)}, {1, 2})

    def test_order_is_configurable(self):
        """规则书第 2 节原本是 金钱 > 政绩 > 官职，换个配置就能切回去。"""
        cfg = dataclasses.replace(CFG, final_ranking_keys=("money", "merit", "rank"))
        a = player(1, money=10, merit=99, rank=1)
        b = player(2, money=10, merit=0, rank=2)
        # 官职优先时 b 赢，规则书原序（政绩优先）时 a 赢
        self.assertEqual([p.id for p in rules.final_winners([a, b], CFG)], [2])
        self.assertEqual([p.id for p in rules.final_winners([a, b], cfg)], [1])


# ==========================================================================
# 出身卡
# ==========================================================================


class TestOrigins(unittest.TestCase):
    """每个技能一条用例，断言**只有该技能那一项变了**。

    出身是这个游戏里第一次出现"同一条规则对不同玩家不一样"，所以每张牌
    都要和"没有出身"的同一局逐项对照，免得某个技能顺手改了别的东西。
    """

    def test_all_six_are_defined_and_described(self):
        ids = REAL_CFG.origin_ids()
        self.assertEqual(len(ids), 6)
        self.assertEqual(len(set(ids)), 6, "出身 id 撞车了")
        for oid in ids:
            self.assertIn(oid, {o.value for o in Origin}, f"{oid} 在枚举里没有")
            d = REAL_CFG.origin(oid)
            self.assertTrue(d["name"] and d["skill"] and d["description"])

    def test_the_master_switch_turns_everything_off(self):
        """平衡对照组要能一键关掉：关了之后所有技能都不该生效。"""
        off = dataclasses.replace(REAL_CFG, origins_enabled=False)
        rich = player(1, origin=Origin.RICH)
        self.assertEqual(rules.apply_origin_start_bonuses([rich], off), {})
        self.assertEqual(rich.money, 0)
        official = player(2, origin=Origin.OFFICIAL)
        self.assertEqual(
            rules.merit_cost_for(official, off), off.merit_cost(0)
        )

    # ---- 富二代 · 老钱 ----

    def test_rich_starts_with_extra_money(self):
        rich, poor = player(1, origin=Origin.RICH), player(2)
        got = rules.apply_origin_start_bonuses([rich, poor], REAL_CFG)
        self.assertEqual(got, {1: REAL_CFG.origin_old_money_start})
        self.assertEqual(rich.money, REAL_CFG.origin_old_money_start)
        self.assertEqual(poor.money, 0)
        self.assertEqual(rich.merit, 0)  # 只给钱，不给政绩

    def test_the_old_money_is_a_one_off(self):
        """是"开局白拿"，不是每轮领。"""
        rich = player(1, origin=Origin.RICH)
        rules.apply_origin_start_bonuses([rich], REAL_CFG)
        rules.pay_salaries([rich], REAL_CFG)
        self.assertEqual(
            rich.money, REAL_CFG.origin_old_money_start + REAL_CFG.salary(0)
        )

    def test_old_money_is_one_full_tier(self):
        """15 = 基层的金钱门槛：开局白拿正好够买一级。"""
        self.assertEqual(REAL_CFG.origin_old_money_start, REAL_CFG.money_cost(0))

    def test_rich_first_redraw_each_round_is_free(self):
        """富二代：0 -> 1 -> 2 -> 4。免费那次之后从底价开始，不是 0 -> 2 -> 4。"""
        for rank in range(REAL_CFG.president_rank):
            rich = player(1, rank=rank, origin=Origin.RICH)
            plain = player(2, rank=rank)
            base = REAL_CFG.redraw_cost(rank)
            self.assertEqual(
                [rules.redraw_cost_for(rich, k, REAL_CFG) for k in range(4)],
                [0, base, base * 2, base * 4],
            )
            self.assertEqual(
                [rules.redraw_cost_for(plain, k, REAL_CFG) for k in range(3)],
                [base, base * 2, base * 4],
            )

    def test_free_does_not_mean_available_at_the_top(self):
        """主席那一级不提供换牌：富二代也是 None（不能换），不是 0（免费换）。"""
        rich = player(1, rank=REAL_CFG.president_rank, origin=Origin.RICH)
        self.assertIsNone(rules.redraw_cost_for(rich, 0, REAL_CFG))

    # ---- 官二代 · 提携 ----

    def test_patronage_discounts_merit_but_not_money(self):
        for rank in range(REAL_CFG.president_rank):
            plain, vip = player(1, rank=rank), player(2, rank=rank, origin=Origin.OFFICIAL)
            base = rules.merit_cost_for(plain, REAL_CFG)
            cut = rules.merit_cost_for(vip, REAL_CFG)
            self.assertLess(cut, base, f"{rank} 级没打折")
            self.assertEqual(
                cut,
                math.ceil(Fraction(base) * REAL_CFG.origin_patronage_merit_ratio),
            )
            # 金钱门槛一分不动
            self.assertEqual(
                rules.money_cost_for(vip, REAL_CFG),
                rules.money_cost_for(plain, REAL_CFG),
            )

    def test_patronage_is_never_rounded_down_to_free(self):
        """打折不能把门槛抹成 0，也不该出现小数。"""
        for rank in range(REAL_CFG.president_rank):
            vip = player(1, rank=rank, origin=Origin.OFFICIAL)
            cost = rules.merit_cost_for(vip, REAL_CFG)
            self.assertIsInstance(cost, int)
            self.assertGreater(cost, 0)

    def test_patronage_actually_lets_him_promote_earlier(self):
        """光改数字不算数，得真的在结算里生效。"""
        tc = REAL_CFG.merit_cost(0)
        cut = math.ceil(Fraction(tc) * REAL_CFG.origin_patronage_merit_ratio)
        merit = cut  # 刚好够打折后的线，够不着原线
        self.assertLess(merit, tc)
        vip = player(1, rank=0, merit=merit, origin=Origin.OFFICIAL)
        plain = player(2, rank=0, merit=merit)
        out = resolve(
            [vip, plain],
            {1: [Action(Card.PROMOTE_MERIT)], 2: [Action(Card.PROMOTE_MERIT)]},
            cfg=REAL_CFG,
        )
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.MERIT)
        self.assertEqual(out.outcomes[2].promotion, PromotionKind.NONE)

    # ---- 红二代 · 一纸调令 ----

    def test_family_card_ignores_attacks(self):
        """一纸调令攻击挡不住：政绩够门槛就升，哪怕挨了政治攻击。"""
        tc = REAL_CFG.merit_cost(1)
        red = player(1, rank=1, merit=tc, origin=Origin.RED)
        attacker = player(2, rank=1)
        out = resolve(
            [red, attacker],
            {1: [Action(Card.PROMOTE_FAMILY)], 2: [Action(Card.ATTACK, 1)]},
            cfg=REAL_CFG,
        )
        self.assertTrue(out.outcomes[1].attacked)
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.MERIT)
        self.assertTrue(out.outcomes[1].family_promotion)
        self.assertEqual(red.rank, 2)

    def test_family_card_prefers_merit_then_money(self):
        rank = 1
        both = player(1, rank=rank, merit=REAL_CFG.merit_cost(rank),
                      money=REAL_CFG.money_cost(rank), origin=Origin.RED)
        cash = player(2, rank=rank, merit=0, money=REAL_CFG.money_cost(rank), origin=Origin.RED)
        out = resolve(
            [both, cash],
            {1: [Action(Card.PROMOTE_FAMILY)], 2: [Action(Card.PROMOTE_FAMILY)]},
            cfg=REAL_CFG,
        )
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.MERIT)
        self.assertEqual(out.outcomes[1].family_bribe, 0)
        self.assertEqual(out.outcomes[2].promotion, PromotionKind.MONEY)
        self.assertEqual(out.outcomes[2].family_bribe, REAL_CFG.money_cost(rank))

    def test_family_card_bought_with_money_still_counts_as_bribery(self):
        """用钱升的那笔算行贿：举报查实记警告、举报人分到钱，但官不撤、钱不重复扣。"""
        rank = 1
        cost = REAL_CFG.money_cost(rank)
        red = player(1, rank=rank, merit=0, money=cost + 3, origin=Origin.RED)
        reporter = player(2, rank=1)
        out = resolve(
            [red, reporter],
            {1: [Action(Card.PROMOTE_FAMILY)], 2: [Action(Card.REPORT, 1)]},
            cfg=REAL_CFG,
        )
        o = out.outcomes[1]
        self.assertTrue(o.report_effective)
        self.assertEqual(o.warnings_issued, 1)
        self.assertEqual(red.rank, rank + 1)  # 官不撤
        self.assertEqual(red.money, 3)  # 只花了一次门槛
        self.assertGreater(out.outcomes[2].money_from_reports, 0)
        self.assertTrue(any("调令" in m for m in out.public_messages))

    def test_family_card_after_work_uses_this_rounds_merit(self):
        """排在干活后面：这一轮的政绩算数。"""
        tc = REAL_CFG.merit_cost(0)
        red = player(1, rank=0, merit=tc - 5, origin=Origin.RED)
        out = resolve(
            [red, player(2)],
            {1: [Action(Card.WORK, value=8), Action(Card.PROMOTE_FAMILY)]},
            cfg=REAL_CFG,
        )
        self.assertTrue(out.outcomes[1].family_promotion)
        self.assertEqual(red.rank, 1)

    def test_family_card_can_spend_this_rounds_loot_unless_caught(self):
        """排在贪污后面用钱升：这一轮刚贪的钱能花，但要等举报结算完——
        没被抓就升；被抓了赃款整笔没收，钱不够就升不了（不能先贪再洗进官位）。"""
        mc = REAL_CFG.money_cost(0)

        def run(reported):
            red = player(1, rank=0, merit=0, money=0, origin=Origin.RED)
            reporter = player(2, rank=0)
            acts = {1: [Action(Card.CORRUPT, value=18), Action(Card.PROMOTE_FAMILY)]}
            if reported:
                acts[2] = [Action(Card.REPORT, 1)]
            out = resolve([red, reporter], acts, cfg=REAL_CFG)
            return red, out.outcomes[1]

        red, o = run(reported=False)
        self.assertTrue(o.family_promotion, "没被抓：用这一轮的赃款升上去")
        self.assertEqual(red.rank, 1)
        red, o = run(reported=True)
        self.assertTrue(o.report_effective)
        self.assertEqual(o.money_confiscated, o.corrupt_amount, "赃款整笔没收，没被洗走")
        self.assertFalse(o.family_promotion)
        self.assertEqual(red.rank, 0)
        self.assertGreater(mc, 0)

    def test_last_step_merit_card_cannot_launder_this_rounds_loot(self):
        """主席那一级政绩升职也要花钱：这一轮刚贪的钱得等举报结算完才能花。

        对局 D7TE 第 10 轮：老李开局 14 块，以权谋私拿了 27 当场凑够 37 升主席，
        反腐风暴来抄时兜里只剩 4——赃款大半被"洗"进了官位。
        """
        top = REAL_CFG.president_rank - 1
        tc, mc = REAL_CFG.merit_cost(top), REAL_CFG.money_cost(top)

        def run(clean_money, reported):
            p1 = player(1, rank=top, merit=tc, money=clean_money)
            reporter = player(2, rank=0)
            acts = {1: [Action(Card.CORRUPT, value=18), Action(Card.PROMOTE_MERIT)]}
            if reported:
                acts[2] = [Action(Card.REPORT, 1)]
            out = resolve([p1, reporter], acts, cfg=REAL_CFG)
            return p1, out.outcomes[1]

        # 被抓：赃款整笔没收（不会只剩个零头可抄），干净的钱不够 -> 升不上去
        p1, o = run(clean_money=10, reported=True)
        self.assertEqual(o.money_confiscated, o.corrupt_amount)
        self.assertEqual(o.promotion, PromotionKind.NONE)
        self.assertEqual(p1.rank, top)
        # 没被抓：等结算完赃款还在，照样升
        p1, o = run(clean_money=10, reported=False)
        self.assertEqual(o.promotion, PromotionKind.BOTH)
        # 被抓但干净的钱本来就够：举报冻不住政绩路线，照样升
        p1, o = run(clean_money=mc, reported=True)
        self.assertTrue(o.report_effective)
        self.assertEqual(o.promotion, PromotionKind.BOTH)

    def test_family_card_cannot_make_president(self):
        top = REAL_CFG.president_rank - 1
        red = player(1, rank=top, merit=99, money=99, origin=Origin.RED)
        out = resolve([red, player(2)], {1: [Action(Card.PROMOTE_FAMILY)]}, cfg=REAL_CFG)
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.NONE)
        self.assertEqual(red.rank, top)

    # ---- 贫农 · 政治正确 ----

    def test_block_can_cost_part_of_the_merit(self):
        """ATTACK_BLOCK_MERIT_LOSS：被穿小鞋挡下时掉这一比例的政绩；默认 0 一点不掉。"""
        tc = REAL_CFG.merit_cost(1)
        for ratio, expect_loss in ((Fraction(0), 0), (Fraction(1, 5), (tc + 10) // 5)):
            cfg = dataclasses.replace(REAL_CFG, attack_block_merit_loss=ratio)
            target = player(1, rank=1, merit=tc + 10)
            attacker = player(2, rank=1)
            out = resolve(
                [target, attacker],
                {1: [Action(Card.PROMOTE_MERIT)], 2: [Action(Card.ATTACK, 1)]},
                cfg=cfg,
            )
            self.assertTrue(out.outcomes[1].merit_promotion_blocked)
            self.assertEqual(target.merit, tc + 10 - expect_loss)
            self.assertEqual(out.outcomes[1].merit_wiped_by_attack, expect_loss)

    def test_block_penalty_scales_with_rank(self):
        """ATTACK_BLOCK_MERIT_PENALTY：穿小鞋再扣几点，按目标官职倍率折算（和戴帽子一样）。"""
        cfg = dataclasses.replace(REAL_CFG, attack_block_merit_penalty=2)
        for rank in range(cfg.president_rank - 1):
            tc = cfg.merit_cost(rank)
            target, attacker = player(1, rank=rank, merit=tc), player(2, rank=rank)
            out = resolve(
                [target, attacker],
                {1: [Action(Card.PROMOTE_MERIT)], 2: [Action(Card.ATTACK, 1)]},
                cfg=cfg,
            )
            self.assertTrue(out.outcomes[1].merit_promotion_blocked)
            self.assertEqual(target.merit, tc - rules.work_merit(2, rank, None, cfg))

    def test_report_fee_comes_off_each_reporters_share(self):
        """REPORT_REWARD_FEE：举报收益 = 赃款/2 - 1（每人扣，扣掉的充公，不会扣成负数）。"""
        cfg = dataclasses.replace(REAL_CFG, report_reward_fee=1)
        for n_reporters in (1, 2):
            thief = player(1, rank=0)
            reporters = [player(i, rank=0) for i in range(2, 2 + n_reporters)]
            actions = {1: [Action(Card.CORRUPT, value=18)]}
            for r in reporters:
                actions[r.id] = [Action(Card.REPORT, 1)]
            out = resolve([thief, *reporters], actions, cfg=cfg)
            gross = out.outcomes[1].corrupt_amount
            share = (gross // 2) // n_reporters - 1
            for r in reporters:
                self.assertEqual(out.outcomes[r.id].money_from_reports, max(0, share))

    def test_hat_reward_goes_to_each_attacker(self):
        """ATTACK_HAT_REWARD：戴帽子扣成了，每个攻击者按自己官职记一点功；默认 0 不给。"""
        for reward in (0, 2):
            cfg = dataclasses.replace(REAL_CFG, attack_hat_reward=reward)
            idler = player(1, rank=1, merit=20)
            a1, a2 = player(2, rank=0, merit=0), player(3, rank=2, merit=0)
            out = resolve(
                [idler, a1, a2],
                {1: [Action(Card.REPORT, 2)], 2: [Action(Card.ATTACK, 1)], 3: [Action(Card.ATTACK, 1)]},
                cfg=cfg,
            )
            self.assertGreater(out.outcomes[1].attack_merit_loss, 0, "帽子扣成了")
            self.assertEqual(a1.merit, rules.work_merit(reward, 0, None, cfg))
            self.assertEqual(a2.merit, rules.work_merit(reward, 2, None, cfg))

    def test_peasant_ignores_the_smear(self):
        tc = REAL_CFG.merit_cost(0)
        peasant = player(1, rank=0, merit=tc, origin=Origin.PEASANT)
        plain = player(2, rank=0, merit=tc)
        attacker = player(3, rank=0)
        out = resolve(
            [peasant, plain, attacker],
            {
                1: [Action(Card.PROMOTE_MERIT)],
                2: [Action(Card.PROMOTE_MERIT)],
                3: [Action(Card.ATTACK, 1), Action(Card.ATTACK, 2)],
            },
            cfg=REAL_CFG,
        )
        self.assertTrue(out.outcomes[1].attacked)  # 照样挨打
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.MERIT)  # 但拦不住
        self.assertFalse(out.outcomes[1].merit_promotion_blocked)
        self.assertEqual(out.outcomes[2].promotion, PromotionKind.NONE)  # 对照组被拦

    def test_two_attackers_get_through_the_peasant(self):
        """（开关 ORIGIN_PEASANT_MAX_ATTACKERS=1 时）贫农只挡得住一个人：两个人一起攻击，穿小鞋照样生效。"""
        cfg = dataclasses.replace(REAL_CFG, origin_peasant_max_attackers=1)
        tc = cfg.merit_cost(0)
        peasant = player(1, rank=0, merit=tc, origin=Origin.PEASANT)
        a, b = player(2, rank=0), player(3, rank=0)
        out = resolve(
            [peasant, a, b],
            {1: [Action(Card.PROMOTE_MERIT)],
             2: [Action(Card.ATTACK, 1)], 3: [Action(Card.ATTACK, 1)]},
            cfg=cfg,
        )
        self.assertEqual(out.outcomes[1].attacker_count, 2)
        self.assertTrue(out.outcomes[1].merit_promotion_blocked)
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.NONE)

    def test_peasant_old_version_ignores_any_number_of_attackers(self):
        """开关 ORIGIN_PEASANT_MAX_ATTACKERS=None（上一版）：三个人一起攻击也拦不住他政绩升职。"""
        cfg = dataclasses.replace(REAL_CFG, origin_peasant_max_attackers=None)
        tc = cfg.merit_cost(0)
        peasant = player(1, rank=0, merit=tc, origin=Origin.PEASANT)
        a, b, c = player(2, rank=0), player(3, rank=0), player(4, rank=0)
        out = resolve(
            [peasant, a, b, c],
            {1: [Action(Card.PROMOTE_MERIT)], 2: [Action(Card.ATTACK, 1)],
             3: [Action(Card.ATTACK, 1)], 4: [Action(Card.ATTACK, 1)]},
            cfg=cfg,
        )
        self.assertEqual(out.outcomes[1].attacker_count, 3)
        self.assertFalse(out.outcomes[1].merit_promotion_blocked)
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.MERIT)
        self.assertIn("不管几个人", cfg.origin("PEASANT")["description"])

    def test_one_person_attacking_twice_still_counts_as_one(self):
        """同一个人打两张攻击不算"两个人"，贫农照样挡得住。"""
        tc = REAL_CFG.merit_cost(0)
        peasant = player(1, rank=0, merit=tc, origin=Origin.PEASANT)
        a = player(2, rank=0)
        out = resolve(
            [peasant, a],
            {1: [Action(Card.PROMOTE_MERIT)],
             2: [Action(Card.ATTACK, 1), Action(Card.ATTACK, 1)]},
            cfg=REAL_CFG,
        )
        self.assertEqual(out.outcomes[1].attacker_count, 1)
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.MERIT)

    def test_attacker_count_is_recorded_even_when_attacks_are_anonymous(self):
        """（只挡一个人的版本）不公开署名时 attacked_by 是空的，贫农的判定不能靠它。"""
        cfg = dataclasses.replace(REAL_CFG, attack_announces_attacker=False,
                                  origin_peasant_max_attackers=1)
        tc = cfg.merit_cost(0)
        peasant = player(1, rank=0, merit=tc, origin=Origin.PEASANT)
        a, b = player(2, rank=0), player(3, rank=0)
        out = resolve(
            [peasant, a, b],
            {1: [Action(Card.PROMOTE_MERIT)],
             2: [Action(Card.ATTACK, 1)], 3: [Action(Card.ATTACK, 1)]},
            cfg=cfg,
        )
        self.assertEqual(out.outcomes[1].attacked_by, [])
        self.assertEqual(out.outcomes[1].promotion, PromotionKind.NONE)

    def test_peasant_still_takes_the_other_two_attack_effects(self):
        """只免疫"穿小鞋"。抢功和戴帽子照打，不然这张就太全能了。"""
        peasant = player(1, rank=0, origin=Origin.PEASANT)
        attacker = player(2, rank=0)
        out = resolve(
            [peasant, attacker],
            {1: [Action(Card.WORK)], 2: [Action(Card.ATTACK, 1)]},
            script=[work_roll(10)],
            cfg=REAL_CFG,
        )
        self.assertGreater(out.outcomes[1].merit_stolen_by_attackers, 0, "抢功该照样生效")

    # ---- 卷王 · 加班 ----

    def test_grinder_single_work_is_plain(self):
        """卷王的牌和普通人一样：单打一张埋头工作，政绩完全相同。"""
        grinder, plain = player(1, rank=2, origin=Origin.GRINDER), player(2, rank=2)
        out = resolve(
            [grinder, plain],
            {1: [Action(Card.WORK, value=7)], 2: [Action(Card.WORK, value=7)]},
            cfg=REAL_CFG,
        )
        self.assertEqual(out.outcomes[1].merit_gained, out.outcomes[2].merit_gained)
        self.assertEqual(out.outcomes[1].overtime_merit, 0)

    def test_grinder_two_works_times_four(self):
        """连干两张：这两张的政绩最后 ×4。流水账单列多出来的那 3 份。"""
        for rank in range(REAL_CFG.president_rank):
            g, plain = player(1, rank=rank, origin=Origin.GRINDER), player(2, rank=rank)
            out = resolve(
                [g, plain],
                {1: [Action(Card.WORK, value=6), Action(Card.WORK, value=4)],
                 2: [Action(Card.WORK, value=6), Action(Card.WORK, value=4)]},
                cfg=REAL_CFG,
            )
            two = rules.work_merit(6, rank, None, REAL_CFG) + rules.work_merit(4, rank, None, REAL_CFG)
            self.assertEqual(out.outcomes[1].merit_gained, 4 * two)
            self.assertEqual(out.outcomes[1].overtime_merit, 3 * two)
            self.assertEqual(out.outcomes[2].merit_gained, two)
            self.assertEqual(out.outcomes[2].overtime_merit, 0)
            rows = {r["label"]: r for r in out.outcomes[1].ledger_lines("x")}
            self.assertEqual(rows["加班：两张工作翻倍多出来的"]["merit"], 3 * two)
            self.assertEqual(rows["干活所得"]["merit"], two)

    def test_overtime_pays_four_salaries(self):
        """加班费：上了别人 4 倍的班就拿 4 倍工资，按官职；合法收入，举报查不到。"""
        k = REAL_CFG.origin_grinder_overtime_multiplier
        for rank in range(REAL_CFG.president_rank):
            g = player(1, rank=rank, money=0, origin=Origin.GRINDER)
            reporter = player(2, rank=rank)
            out = resolve(
                [g, reporter],
                {1: [Action(Card.WORK, value=6), Action(Card.WORK, value=6)],
                 2: [Action(Card.REPORT, 1)]},
                cfg=REAL_CFG,
            )
            o = out.outcomes[1]
            self.assertEqual(o.overtime_pay, k * REAL_CFG.salary(rank))
            self.assertEqual(g.money, k * REAL_CFG.salary(rank))
            self.assertFalse(o.report_effective, "加班费是合法收入，举报查不到")
            rows = {r["label"]: r for r in o.ledger_lines("x")}
            self.assertEqual(rows["加班费"]["money"], k * REAL_CFG.salary(rank))

    def test_no_overtime_pay_for_one_work_or_non_grinders(self):
        g, plain = player(1, origin=Origin.GRINDER), player(2)
        out = resolve(
            [g, plain],
            {1: [Action(Card.WORK, value=6)],
             2: [Action(Card.WORK, value=6), Action(Card.WORK, value=6)]},
            cfg=REAL_CFG,
        )
        self.assertEqual(out.outcomes[1].overtime_pay, 0)
        self.assertEqual(out.outcomes[2].overtime_pay, 0)

    def test_overtime_multiplies_the_event_boosted_work(self):
        """重点项目这类事件加成先算进每张牌，再整体 ×4。"""
        boost = next(e for e in REAL_CFG.event_definitions if e["effects"].get("work_bonus"))
        event = rules.event_by_id(boost["id"], REAL_CFG)
        g = player(1, rank=0, origin=Origin.GRINDER)
        out = resolve(
            [g, player(2)],
            {1: [Action(Card.WORK, value=5), Action(Card.WORK, value=8)]},
            event, cfg=REAL_CFG,
        )
        two = rules.work_merit(5, 0, event, REAL_CFG) + rules.work_merit(8, 0, event, REAL_CFG)
        self.assertEqual(out.outcomes[1].overtime_merit, 3 * two)

    def test_attackers_only_steal_from_before_the_multiplier(self):
        """×4 在攻击之后：抢功按乘之前的政绩抢，剩下的再 ×4。"""
        g = player(1, rank=1, origin=Origin.GRINDER)
        attacker = player(2, rank=1)
        out = resolve(
            [g, attacker],
            {1: [Action(Card.WORK, value=6), Action(Card.WORK, value=6)],
             2: [Action(Card.ATTACK, 1)]},
            cfg=REAL_CFG,
        )
        two = 2 * rules.work_merit(6, 1, None, REAL_CFG)
        stolen = out.outcomes[1].merit_stolen_by_attackers
        self.assertGreater(stolen, 0)
        self.assertLessEqual(stolen, two)
        self.assertEqual(out.outcomes[1].overtime_merit, 3 * (two - stolen))

    def test_two_works_reach_the_threshold_for_a_grinder(self):
        """政绩门槛 = 3 张本级埋头工作；卷王连干两张 ×4 每级都远超门槛。"""
        dist = [v for v, w in REAL_CFG.work_card_distribution for _ in range(w)]
        e_one = sum(dist) / len(dist)
        k = REAL_CFG.origin_grinder_overtime_multiplier
        for rank in range(REAL_CFG.president_rank):
            mult = float(REAL_CFG.work_multiplier(rank))
            self.assertEqual(REAL_CFG.merit_cost(rank), round(3 * e_one * mult))
            self.assertGreaterEqual(2 * e_one * mult * k, REAL_CFG.merit_cost(rank))

    def test_grinder_hand_is_topped_up_not_front_loaded(self):
        """卷王保底：先照常随机发满一手，不够两张工作才把缺的那几张换掉。

        同一个 seed 下，卷王的手牌 = 普通人的那一手，只是缺的工作被补上——
        本来就有两张以上工作的手牌一张都不动（不是"先塞两张再发四张"）。
        """
        import random as _r
        grinder, plain = player(1, origin=Origin.GRINDER), player(2)
        topped = 0
        for seed in range(300):
            natural = rules.deal_hand_for(plain, _r.Random(seed), REAL_CFG)
            mine = rules.deal_hand_for(grinder, _r.Random(seed), REAL_CFG)
            n_nat = sum(d.card is Card.WORK for d in natural)
            n_mine = sum(d.card is Card.WORK for d in mine)
            self.assertEqual(n_mine, max(n_nat, REAL_CFG.origin_grinder_min_work))
            changed = [i for i in range(len(mine)) if mine[i] != natural[i]]
            self.assertTrue(all(mine[i].card is Card.WORK for i in changed))
            self.assertTrue(all(natural[i].card is not Card.WORK for i in changed))
            if n_nat >= REAL_CFG.origin_grinder_min_work:
                self.assertEqual(mine, natural)
            topped += bool(changed)
        self.assertGreater(topped, 100, "大约六成多的手牌需要补")

    def test_grinder_does_not_touch_corruption(self):
        grinder = player(1, origin=Origin.GRINDER)
        plain = player(2)
        out = resolve(
            [grinder, plain],
            {1: [Action(Card.CORRUPT, value=10)], 2: [Action(Card.CORRUPT, value=10)]},
            cfg=REAL_CFG,
        )
        self.assertEqual(
            out.outcomes[1].money_gained, out.outcomes[2].money_gained
        )

    # ---- 小镇做题家·会计 · 做账 ----

    def _caught(self, origin, amount=40):
        victim = PlayerState(id=1, name="贪", origin=origin)
        reporter = PlayerState(id=2, name="举报人")
        out = resolve(
            [victim, reporter],
            {1: [Action(Card.CORRUPT, value=amount)], 2: [Action(Card.REPORT, 1)]},
            cfg=REAL_CFG,
        )
        return out.outcomes[1], victim, reporter

    def test_laundering_shields_half_the_haul(self):
        plain_o, plain_v, _ = self._caught(None)
        acc_o, acc_v, _ = self._caught(Origin.ACCOUNTANT)
        self.assertGreater(acc_o.laundered, 0)
        self.assertLess(acc_o.money_confiscated, plain_o.money_confiscated)
        self.assertEqual(acc_v.money, acc_o.laundered)  # 洗白的那份留住了
        self.assertEqual(plain_v.money, 0)

    def test_laundering_does_not_shrink_the_reporters_cut(self):
        """**这是这张牌的核心约束。**

        洗白只能吃掉"充公"那一份。要是按没收额去算分成，抓会计的回本就比
        抓别人少一半，结果是没人愿意抓他 —— 那是个很讨厌的副作用。
        """
        _, _, plain_r = self._caught(None)
        _, _, acc_r = self._caught(Origin.ACCOUNTANT)
        self.assertEqual(acc_r.money, plain_r.money)
        self.assertGreater(acc_r.money, 0)

    def test_laundering_does_not_change_whether_he_is_caught(self):
        """做账挡的是钱，不是罪。查实与否、记几次警告都不变。"""
        plain_o, _, _ = self._caught(None)
        acc_o, _, _ = self._caught(Origin.ACCOUNTANT)
        self.assertTrue(acc_o.report_effective)
        self.assertEqual(acc_o.warnings_issued, plain_o.warnings_issued)
        self.assertEqual(acc_o.corrupt_amount, plain_o.corrupt_amount)

    def test_laundering_is_free_money_when_nobody_reports_him(self):
        """没被举报时，会计和普通人一模一样——做账只在被抓那一刻生效。"""
        acc = PlayerState(id=1, name="会计", origin=Origin.ACCOUNTANT)
        plain = PlayerState(id=2, name="普通")
        out = resolve(
            [acc, plain],
            {1: [Action(Card.CORRUPT, value=40)], 2: [Action(Card.CORRUPT, value=40)]},
            cfg=REAL_CFG,
        )
        self.assertEqual(acc.money, plain.money)
        self.assertEqual(
            out.outcomes[1].corrupt_amount, out.outcomes[2].corrupt_amount
        )

    # ---- 红二代 · 硬保 ----

    def _caught_twice(self, origin):
        """连吃两次查实（刚好攒满降职线），看官职动没动。"""
        wmax = REAL_CFG.warnings_before_demotion
        victim = PlayerState(id=1, name="红", rank=2, origin=origin)
        reporter = PlayerState(id=2, name="举报人")
        last = None
        for _ in range(wmax):
            last = resolve(
                [victim, reporter],
                {1: [Action(Card.CORRUPT, value=40)], 2: [Action(Card.REPORT, 1)]},
                cfg=REAL_CFG,
            )
        return victim, last.outcomes[1]

    def test_red_never_gets_demoted(self):
        red, o = self._caught_twice(Origin.RED)
        plain, po = self._caught_twice(None)
        self.assertEqual(red.rank, 2, "硬保了还被降职")
        self.assertEqual(plain.rank, 1, "对照组该降一级")
        self.assertTrue(o.origin_shielded_demotion)
        self.assertEqual(o.demotion, DemotionKind.NONE)
        self.assertEqual(po.demotion, DemotionKind.MINOR)

    def test_red_still_gets_everything_else(self):
        """硬保只保官职：警告照记、工龄照清、赃款照抄、举报照样查实。"""
        red, o = self._caught_twice(Origin.RED)
        self.assertTrue(o.report_effective)
        self.assertGreater(o.warnings_issued, 0)
        self.assertGreater(o.money_confiscated, 0)
        self.assertEqual(red.tenure, 0)
        self.assertEqual(red.money, 0)

    def test_the_reporter_is_paid_the_same_for_catching_a_red(self):
        """保的是他的官，不是别人的赏钱。"""
        wmax = REAL_CFG.warnings_before_demotion

        def payout(origin):
            victim = PlayerState(id=1, name="红", rank=2, origin=origin)
            reporter = PlayerState(id=2, name="举报人")
            for _ in range(wmax):
                resolve(
                    [victim, reporter],
                    {1: [Action(Card.CORRUPT, value=40)], 2: [Action(Card.REPORT, 1)]},
                    cfg=REAL_CFG,
                )
            return reporter.money

        self.assertEqual(payout(Origin.RED), payout(None))

    def test_the_warning_counter_still_cycles(self):
        """挡掉降职之后警告要清空，不然 UI 上会出现"再记 -1 次就降级"。"""
        red, _ = self._caught_twice(Origin.RED)
        self.assertLess(red.warnings, REAL_CFG.warnings_before_demotion)
        self.assertGreaterEqual(red.warnings, 0)

    def test_the_ai_stops_counting_warnings_against_a_red(self):
        """他降不下来，"离降级越近越值钱"那一项对他应该归零。"""
        import ai as ai_mod

        pool = ai_mod.AgentPool(cfg=REAL_CFG, rng=random.Random(0))
        agent = pool.get(1)
        wmax = REAL_CFG.warnings_before_demotion
        public = {
            "players": [
                {"id": 1, "name": "我", "rank": 2, "merit": 0, "tenure": 0,
                 "origin": None, "warnings": 0},
                {"id": 2, "name": "红", "rank": 2, "merit": 0, "tenure": 0,
                 "origin": "RED", "warnings": wmax - 1},
                {"id": 3, "name": "普通", "rank": 2, "merit": 0, "tenure": 0,
                 "origin": None, "warnings": wmax - 1},
            ],
            "round": 5, "last_result": None,
            "picks_per_round": REAL_CFG.picks_per_round,
            "max_rounds": REAL_CFG.max_rounds,
        }
        private = {"rank": 2, "merit": 0, "money": 0, "origin": None}
        agent.observe(public)
        opp = {o["id"]: o for o in public["players"]}
        red = agent._score_report(public, private, opp[2])[0]
        plain = agent._score_report(public, private, opp[3])[0]
        self.assertLess(red, plain, "两人都差一次降级，举报红二代不该一样值钱")


if __name__ == "__main__":
    unittest.main()
