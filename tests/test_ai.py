"""ai.py 的行为测试：AI 只能看公开信息，而且不能有系统性偏向。"""

from __future__ import annotations

import random
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ai  # noqa: E402
import rules  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402
from game import Game  # noqa: E402
from models import Card, DealtCard, Origin  # noqa: E402

CFG = DEFAULT_CONFIG
# 一纸调令默认已删掉；测调令本身的用例用这个
import dataclasses as _dc  # noqa: E402
FAMILY_CFG = _dc.replace(DEFAULT_CONFIG, origin_red_family_card=True)


def fresh_table(n=6):
    """开局状态：所有人完全一样，所有目标的评分必然打平。"""
    return {
        "players": [
            {"id": i, "name": f"P{i}", "rank": 0, "merit": 0, "tenure": 0} for i in range(1, n + 1)
        ],
        "round": 1,
        "last_result": None,
        "picks_per_round": CFG.picks_per_round,
    }


class TestTargetingIsUnbiased(unittest.TestCase):
    """开局打平时必须随机挑目标。

    这里守的是一个真实踩过的坑：原来用 max() 取第一个最大值，而开局所有目标
    评分精确相等，于是全场 AI 一起选中编号最小的玩家——房主永远是 1 号，
    结果真人每局第一轮都被集体举报。
    """

    def _distribution(self, card, trials=2000):
        pub = fresh_table()
        me = 2
        opponents = [o for o in pub["players"] if o["id"] != me]
        priv = {"rank": 0, "merit": 0, "money": 0, "hand": [card.value], "player_id": me}
        hits = Counter()
        for seed in range(trials):
            agent = ai.SmartAgent(me, cfg=CFG, rng=random.Random(seed))
            agent.observe(pub)
            hits[agent._pick_target(pub, priv, card, opponents, set())] += 1
        return hits, len(opponents), trials

    def _assert_spread(self, card):
        hits, n_opp, trials = self._distribution(card)
        self.assertEqual(len(hits), n_opp, f"{card} 应该会挑到每一个对手，而不是死盯一个")
        expected = trials / n_opp
        for pid, count in hits.items():
            self.assertGreater(
                count, expected * 0.6,
                f"{card} 对 P{pid} 的选中率过低（{count}/{trials}），分布不均",
            )
            self.assertLess(
                count, expected * 1.4,
                f"{card} 对 P{pid} 的选中率过高（{count}/{trials}），有系统性偏向",
            )

    def test_report_spreads_across_opponents(self):
        self._assert_spread(Card.REPORT)

    def test_attack_spreads_across_opponents(self):
        self._assert_spread(Card.ATTACK)

    def test_the_host_is_not_singled_out_in_round_one(self):
        """整桌 AI 在第 1 轮不该一致地扑向 1 号（房主）。"""
        pub = fresh_table()
        priv_tpl = {"rank": 0, "merit": 0, "money": 0, "player_id": 0}
        picked = Counter()
        for seed in range(400):
            for me in (2, 3, 4, 5, 6):
                agent = ai.SmartAgent(me, cfg=CFG, rng=random.Random(seed * 10 + me))
                agent.observe(pub)
                priv = dict(priv_tpl, player_id=me, hand=["REPORT"])
                opponents = [o for o in pub["players"] if o["id"] != me]
                picked[agent._pick_target(pub, priv, Card.REPORT, opponents, set())] += 1
        total = sum(picked.values())
        # 1 号在每个人眼里都是对手，其他人只有 4/5 的场合是，所以基准略高
        self.assertLess(picked[1] / total, 0.30, "1 号被针对得太多了")


class TestMoneyEstimateAfterPublicEvents(unittest.TestCase):
    """对手的钱是暗的，但几条公开事实足够把估计钉住（复盘 5A9J）。"""

    def _observe(self, agent, fact_overrides, top_ids, merit_after=0, rank_after=None):
        before = {"id": 2, "rank": 1, "merit": 0, "origin": None}
        agent.observe({"players": [{"id": 1, "rank": 0, "merit": 0, "origin": None}, before],
                       "last_result": None})
        fact = {"player_id": 2, "attacked": False, "attack_merit_loss": 0,
                "promotion": "NONE", "demotion": "NONE", "warnings_issued": 0,
                "rank_before": 1, "rank_after": 1}
        fact.update(fact_overrides)
        after = dict(before, merit=merit_after, rank=fact["rank_after"])
        agent.observe({
            "players": [{"id": 1, "rank": 0, "merit": 0, "origin": None}, after],
            "last_result": {"round": 1, "wealth_top_ids": top_ids, "player_facts": [
                fact,
                {"player_id": 1, "attacked": False, "attack_merit_loss": 0, "promotion": "NONE",
                 "demotion": "NONE", "warnings_issued": 0, "rank_before": 0, "rank_after": 0},
            ]},
        })
        return agent.models[2].money_est

    def test_buying_a_promotion_keeps_the_change(self):
        """花钱升职只扣门槛，余额全留——以前误按 /5 算，买过官的人存款被估成零头。"""
        agent = ai.SmartAgent(1, cfg=CFG, rng=random.Random(0))
        agent.models[2] = ai.OpponentModel(money_est=CFG.money_cost(1) + 20)
        est = self._observe(agent, {"promotion": "MONEY", "rank_after": 2}, [])
        self.assertGreaterEqual(est, 20)

    def test_being_named_by_the_gossip_counts_as_dirty_money(self):
        """有传闻 = 这轮确实有人捞了没被抓，被点名的就是到手最多的——至少记一笔以权谋私。"""
        quiet = ai.SmartAgent(1, cfg=CFG, rng=random.Random(0))
        named = ai.SmartAgent(1, cfg=CFG, rng=random.Random(0))
        # 政绩涨了（看起来在干活），我自己只拿工资，证明不了他贪了
        base = self._observe(quiet, {}, [], merit_after=5)
        est = self._observe(named, {}, [2], merit_after=5)
        self.assertGreaterEqual(est - base, named._expected_graft_money(1) * 0.5)


class TestAgentOnlySeesPublicInfo(unittest.TestCase):
    def test_choose_runs_off_the_same_payload_a_browser_gets(self):
        game = Game(game_id="t", cfg=CFG, rng=random.Random(5))
        for i in range(4):
            game.add_player(f"P{i + 1}")
        game.start_game()
        game.players[2].money = 9999  # 别人的巨款
        pool = ai.AgentPool(cfg=CFG, rng=random.Random(5))
        picks = ai.choose(game, 1, pool)
        self.assertEqual(len(picks), CFG.picks_per_round)
        agent = pool.get(1)
        # AI 对别人金钱的估计只能来自公开推断，不可能等于真实值
        self.assertNotEqual(agent.models[2].money_est, 9999)

    def test_ai_picks_are_legal(self):
        game = Game(game_id="t", cfg=CFG, rng=random.Random(11))
        for i in range(5):
            game.add_player(f"P{i + 1}")
        game.start_game()
        pool = ai.AgentPool(cfg=CFG, rng=random.Random(11))
        for _ in range(4):
            for pid in sorted(game.players):
                game.select_actions(pid, ai.choose(game, pid, pool))
                game.lock_action(pid)
            game.reveal_event()
            game.resolve()
            if game.is_over:
                break
            game.advance_round()


class TestTheAiUnderstandsOrigins(unittest.TestCase):
    """出身是公开信息，AI 必须把它算进去。

    不算的话会原样重演上一轮那个 bug：AI 用错门槛 -> 算错"他还差多远"
    -> 终局那道"有人下一步就夺冠"的刹车一次都不踩，对手当轮登顶。
    官二代的政绩门槛打了八折，正是这类。
    """

    def _agent(self):
        pool = ai.AgentPool(cfg=CFG, rng=random.Random(0))
        return pool.get(1)

    def test_it_uses_the_discounted_threshold_for_a_patronage_opponent(self):
        agent = self._agent()
        tc = CFG.merit_cost(0)
        cut = rules.merit_cost_at(0, "OFFICIAL", CFG)
        self.assertLess(cut, tc)
        plain = agent._progress(0, 0, cut, None)
        vip = agent._progress(0, 0, cut, "OFFICIAL")
        self.assertEqual(vip, 1.0, "官二代这点政绩已经够线了，AI 该看出来")
        self.assertLess(plain, 1.0)

    def test_a_patronage_opponent_reads_as_more_threatening(self):
        """同样的政绩，官二代离晋升更近，威胁值就该更高。"""
        agent = self._agent()
        merit = CFG.merit_cost(1) * 0.9
        plain = agent._threat(1, agent._progress(1, 0, merit, None))
        vip = agent._threat(1, agent._progress(1, 0, merit, "OFFICIAL"))
        self.assertGreater(vip, plain)

    def test_about_to_win_accounts_for_the_discount(self):
        """终局刹车：官二代够线时门槛更低，不能拿原价去比。"""
        agent = self._agent()
        top = CFG.president_rank - 1
        cut = rules.merit_cost_at(top, "OFFICIAL", CFG)
        self.assertLess(cut, CFG.merit_cost(top))
        model = ai.OpponentModel()
        # 钱一分没有，得靠这一轮现贪 —— 那张生产牌位子就被占了，
        # 政绩必须**已经**够线才算。这样比的正好是"门槛有没有打折"。
        model.money_est = 0
        vip = {"id": 2, "rank": top, "merit": cut, "origin": "OFFICIAL"}
        plain = {"id": 3, "rank": top, "merit": cut, "origin": None}
        self.assertTrue(agent._about_to_win(vip, model), "官二代已经够线了")
        self.assertFalse(agent._about_to_win(plain, model))

    def test_buying_a_promotion_carries_report_risk(self):
        """举报也抓行贿：买官被查实官作废、钱打水漂。AI 以前把买官当零风险，
        钱越多越急着买（富二代开局 10 -> 15，第 2 轮被查实从 7% 涨到 23%）。"""
        import dataclasses
        public = {"players": [{"id": 1, "name": "我", "rank": 0, "merit": 0, "tenure": 0},
                              {"id": 2, "name": "甲", "rank": 0, "merit": 0, "tenure": 0}],
                  "round": 2, "last_result": None, "picks_per_round": 2, "max_rounds": 12}
        private = {"rank": 0, "merit": 0, "money": CFG.money_cost(0) + 5, "origin": None}
        cards, vals = [Card.PROMOTE_MONEY, Card.WORK], [0, 6]
        risky = ai.SmartAgent(1, cfg=CFG)._score_economy(public, private, cards, vals)[0]
        safe_cfg = dataclasses.replace(CFG, report_catches_bribery=False)
        safe = ai.SmartAgent(1, cfg=safe_cfg)._score_economy(public, private, cards, vals)[0]
        self.assertLess(risky, safe)
        # 走政绩那条路不算行贿，不该扣
        merit_private = dict(private, merit=CFG.merit_cost(0), money=0)
        m_cards = [Card.PROMOTE_MERIT, Card.WORK]
        self.assertEqual(
            ai.SmartAgent(1, cfg=CFG)._score_economy(public, merit_private, m_cards, vals),
            ai.SmartAgent(1, cfg=safe_cfg)._score_economy(public, merit_private, m_cards, vals),
        )

    def test_promote_first_then_corrupt_still_counts_the_risk(self):
        """"先升官再贪"那条路以前直接 return，贪污被抓的风险整个没算。"""
        import dataclasses
        public = {"players": [{"id": 1, "name": "我", "rank": 0, "merit": 0, "tenure": 0},
                              {"id": 2, "name": "甲", "rank": 0, "merit": 0, "tenure": 0}],
                  "round": 2, "last_result": None, "picks_per_round": 2, "max_rounds": 12}
        private = {"rank": 0, "merit": CFG.merit_cost(0), "money": 0, "origin": None}
        cards, vals = [Card.PROMOTE_MERIT, Card.CORRUPT], [0, 18]
        fearless = dataclasses.replace(ai.Weights(), caught_dread=0.0)
        brave = ai.SmartAgent(1, cfg=CFG, weights=fearless)._score_economy(public, private, cards, vals)[0]
        normal = ai.SmartAgent(1, cfg=CFG)._score_economy(public, private, cards, vals)[0]
        self.assertLess(normal, brave)

    def test_official_uses_the_tipoff(self):
        """官二代 AI 听到风声要真的用上：反腐风暴不贪，重点项目更愿意埋头工作。"""
        import dataclasses
        public = {"players": [{"id": 1, "name": "我", "rank": 1, "merit": 0, "tenure": 0},
                              {"id": 2, "name": "甲", "rank": 1, "merit": 0, "tenure": 0}],
                  "round": 3, "last_result": None, "picks_per_round": 2, "max_rounds": 12}
        agent = self._agent()
        cards, vals = [Card.CORRUPT, Card.WORK], [18, 6]

        def score(event_id):
            private = {"rank": 1, "merit": 0, "money": 0, "origin": "OFFICIAL"}
            if event_id:
                private["tipoff_event"] = {"id": event_id}
            return agent._score_economy(public, private, cards, vals)[0]

        self.assertLess(score("ANTI_CORRUPTION"), score(None))

        def work_score(event_id):  # 市级：一张工作离门槛还远，不会被进度封顶吃掉
            private = {"rank": 2, "merit": 0, "money": 0, "origin": "OFFICIAL"}
            if event_id:
                private["tipoff_event"] = {"id": event_id}
            return agent._score_economy(public, private, [Card.WORK, Card.PROMOTE_MONEY], [6, 0])[0]

        self.assertGreater(work_score("KEY_PROJECT"), work_score(None))

    def test_red_plays_the_family_card_when_it_has_no_promotion_card(self):
        """够门槛却没摸到晋升卡：红二代 AI 会掏出一纸调令。手里有普通晋升卡时不浪费它。"""
        # 一纸调令默认已删掉，这里测的是开关打开时 AI 会用它
        game = Game(game_id="red", cfg=FAMILY_CFG, rng=random.Random(2))
        for name in ("我", "甲", "乙"):
            game.add_player(name)
        game.players[1].origin = Origin.RED
        game.start_game()
        me = game.players[1]
        me.rank, me.merit, me.money = 1, CFG.merit_cost(1), 0
        pool = ai.AgentPool(cfg=FAMILY_CFG, rng=random.Random(2))
        game.hands[1] = [DealtCard(card=Card.WORK, value=6)] * CFG.hand_size
        picks = [pk["action"] for pk in ai.choose(game, 1, pool)]
        self.assertIn("PROMOTE_FAMILY", picks)
        self.assertEqual(picks[0], "PROMOTE_FAMILY", "开局就够门槛：排最前面先升官")
        self.assertEqual(len([p for p in picks if p != "PROMOTE_FAMILY"]), 2, "不占出牌位：两张手牌照样打满")
        game.hands[1] = [DealtCard(card=Card.PROMOTE_MERIT)] + [DealtCard(card=Card.WORK, value=6)] * 5
        picks = [pk["action"] for pk in ai.choose(game, 1, pool)]
        self.assertNotIn("PROMOTE_FAMILY", picks)
        self.assertIn("PROMOTE_MERIT", picks)

    def test_attacks_go_to_the_highest_rank_not_the_hardest_worker(self):
        """用户原话：基层公务员就算政绩一万也没什么好担心的；省级第一了 AI 还在打我。

        桌上一个政绩堆成山、每轮都在干活的基层，一个省级领跑者——攻击该打省级的。
        """
        game = Game(game_id="focus", cfg=CFG, rng=random.Random(1))
        for name in ("我", "卷王", "省级"):
            game.add_player(name)
        game.start_game()
        me, worker, top = (game.players[i] for i in (1, 2, 3))
        me.rank, me.merit = 1, 5
        worker.rank, worker.merit = 0, 9999
        top.rank, top.merit = CFG.president_rank - 1, 10
        agent = ai.AgentPool(cfg=CFG, rng=random.Random(1)).get(1)
        public, private = game.public_state(), game.private_state(1)
        agent.observe(public, private)
        for pid in (2, 3):  # 两个人都一直在干活
            agent.models[pid].work_rounds = agent.models[pid].observed_rounds = 5
        opp = {o["id"]: o for o in public["players"]}
        on_worker = agent._score_attack(public, private, opp[2])[0]
        on_top = agent._score_attack(public, private, opp[3])[0]
        self.assertGreater(on_top, on_worker, f"打省级 {on_top:.3f} 居然不如打基层 {on_worker:.3f}")
        self.assertLess(agent._target_focus(public, opp[2]), 0.5)
        self.assertAlmostEqual(agent._target_focus(public, opp[3]), 1.0)

    def test_red_places_the_family_card_after_the_work_that_reaches_the_threshold(self):
        """门槛差一点、这一轮干活才够：一纸调令要排在干活**后面**，用这一轮的政绩升上去。

        （它最先结算时是赶不上这一轮产出的——修之前那版 AI 资源不够也照打，94% 白用。）
        """
        # 一纸调令默认已删掉，这里测的是开关打开时 AI 会用它
        game = Game(game_id="red2", cfg=FAMILY_CFG, rng=random.Random(3))
        for name in ("我", "甲", "乙"):
            game.add_player(name)
        game.players[1].origin = Origin.RED
        game.start_game()
        me = game.players[1]
        # 市级：基层升县级用这张每局一次的卡不划算（AI 会留着），市级升省级才值得
        me.rank, me.merit, me.money = 2, CFG.merit_cost(2) - 4, 0
        game.hands[1] = [DealtCard(card=Card.WORK, value=8)] * CFG.hand_size
        pool = ai.AgentPool(cfg=FAMILY_CFG, rng=random.Random(3))
        picks = ai.choose(game, 1, pool)
        actions = [pk["action"] for pk in picks]
        self.assertIn("PROMOTE_FAMILY", actions)
        self.assertEqual(actions[-1], "PROMOTE_FAMILY", "要排在干活后面")
        game.select_actions(1, picks)
        for pid in game.players:
            if pid != 1:
                game.select_actions(pid, [])
            game.lock_action(pid)
        game.reveal_event()
        out = game.resolve()
        self.assertTrue(out.outcomes[1].family_promotion, "用这一轮的政绩升上去了")

    def test_family_card_is_not_tagged_along_with_another_promotion(self):
        """升职靠的是普通晋升卡时，一纸调令配着打只是浪费。修之前 AI 会这么干：
        风险按"官不撤"算轻了，「一纸调令 + 贪污 + 贿赂升职」看起来反而更划算。"""
        game = Game(game_id="red3", cfg=CFG, rng=random.Random(4))
        for name in ("我", "甲", "乙"):
            game.add_player(name)
        game.players[1].origin = Origin.RED
        game.start_game()
        me = game.players[1]
        me.rank, me.merit, me.money = 0, 0, 1
        game.hands[1] = ([DealtCard(card=Card.CORRUPT, value=18), DealtCard(card=Card.PROMOTE_MONEY)]
                         + [DealtCard(card=Card.REPORT)] * 4)
        pool = ai.AgentPool(cfg=CFG, rng=random.Random(4))
        picks = [pk["action"] for pk in ai.choose(game, 1, pool)]
        # 一纸调令和别的晋升卡一起打 = 浪费（一轮只升一级）。
        # 用一纸调令顶替贿赂升职、腾出位子多打一张，是合法而且更好的打法。
        others = [p for p in picks if p.startswith("PROMOTE") and p != "PROMOTE_FAMILY"]
        self.assertFalse("PROMOTE_FAMILY" in picks and others, picks)

    def test_attacking_a_peasant_is_worth_less(self):
        """贫农免疫穿小鞋，所以"挡住他晋升"那份价值不该算进去。"""
        agent = self._agent()
        tc = CFG.merit_cost(1)
        public = {
            "players": [
                {"id": 1, "name": "我", "rank": 1, "merit": 0, "tenure": 0, "origin": None},
                {"id": 2, "name": "贫农", "rank": 1, "merit": tc, "tenure": 0,
                 "origin": "PEASANT"},
                {"id": 3, "name": "普通", "rank": 1, "merit": tc, "tenure": 0,
                 "origin": None},
            ],
            "round": 3, "last_result": None, "picks_per_round": CFG.picks_per_round,
            "max_rounds": CFG.max_rounds,
        }
        private = {"rank": 1, "merit": 0, "money": 0, "origin": None}
        agent.observe(public)
        opp = {o["id"]: o for o in public["players"]}
        peasant = agent._score_attack(public, private, opp[2])[0]
        plain = agent._score_attack(public, private, opp[3])[0]
        self.assertLess(peasant, plain, "打贫农和打普通人一样值钱 = 没认出免疫")

    def test_an_accountant_fears_corruption_less(self):
        """做账能保住一半，所以会计对"被抓"的估损该比别人小。"""
        agent = self._agent()
        self.assertLess(agent._exposed_share("ACCOUNTANT"), 1.0)
        self.assertEqual(agent._exposed_share(None), 1.0)
        self.assertEqual(agent._exposed_share("RICH"), 1.0)

    def test_a_grinder_values_two_works_more(self):
        """卷王单张和普通人一样；连干两张时 AI 要把 ×4 算进去。"""
        agent = self._agent()
        self.assertEqual(
            agent._own_work_merit(6, 2, "GRINDER"), agent._own_work_merit(6, 2, None)
        )
        public = {"players": [{"id": 1, "name": "我", "rank": 1, "merit": 0, "tenure": 0},
                              {"id": 2, "name": "甲", "rank": 1, "merit": 0, "tenure": 0}],
                  "round": 2, "last_result": None, "picks_per_round": 2, "max_rounds": 12}
        cards, vals = [Card.WORK, Card.WORK], [6, 4]

        def score(origin):
            private = {"rank": 1, "merit": 0, "money": 0, "origin": origin}
            return agent._score_economy(public, private, cards, vals)[0]

        # 进度单位，封顶 1.0（政绩够了门槛之后再多也不算进度）
        two = (rules.work_merit(6, 1, None, CFG) + rules.work_merit(4, 1, None, CFG)) / CFG.merit_cost(1)
        k = CFG.origin_grinder_overtime_multiplier
        self.assertAlmostEqual(score("GRINDER") - score(None), min(1.0, k * two) - two, places=6)

    def test_the_master_switch_makes_the_ai_blind_to_origins_too(self):
        """关掉总开关时技能不生效，AI 的估值也得跟着回到原样，
        否则平衡对照组测出来的是"规则关了但 AI 还按开着算"。"""
        import dataclasses

        off = dataclasses.replace(CFG, origins_enabled=False)
        pool = ai.AgentPool(cfg=off, rng=random.Random(0))
        agent = pool.get(1)
        self.assertEqual(
            agent._progress(0, 0, 10, "OFFICIAL"), agent._progress(0, 0, 10, None)
        )
        self.assertEqual(agent._exposed_share("ACCOUNTANT"), 1.0)
        self.assertEqual(
            agent._own_work_merit(0, 0, "GRINDER"), agent._own_work_merit(0, 0, None)
        )


if __name__ == "__main__":
    unittest.main()


class TestTieBreakDoesNotSwallowRealDifferences(unittest.TestCase):
    """打平容差必须是**相对**的，不能是绝对值。

    真实对局里踩到过：中局各目标的攻击分普遍只有 0.006~0.016，
    整个区间比原来的绝对阈值 0.02 还小，于是所有人都被判为"打平"，
    "谁快升职了"这套威胁模型算完就被扔掉，变成随机挑人——
    一个离胜利很远的基层玩家被打，而真正的领先者没人管。
    """

    # 那一局第 6 轮开局时的真实桌面（全是公开信息）
    BOARD = [("胡郁博", 0, 24, 1), ("老张", 0, 4, 7), ("老李", 1, 14, 14),
             ("小王", 1, 11, 43), ("老陈", 2, 2, 1), ("小刘", 0, 2, 8)]
    ME = 4  # 小王

    def _board(self, seed):
        game = Game(game_id="tie", cfg=CFG, rng=random.Random(seed))
        for name, *_ in self.BOARD:
            game.add_player(name)
        game.start_game()
        for pid, (_, rank, merit, money) in zip(sorted(game.players), self.BOARD):
            p = game.players[pid]
            p.rank, p.merit, p.money = rank, merit, money
        return game

    def test_the_best_target_actually_gets_picked(self):
        counts = Counter()
        best_ids = set()
        for seed in range(120):
            game = self._board(seed)
            pool = ai.AgentPool(cfg=CFG, rng=random.Random(seed))
            agent = pool.get(self.ME)
            public, private = game.public_state(), game.private_state(self.ME)
            agent.observe(public)
            opponents = [o for o in public["players"] if o["id"] != self.ME]
            scored = [
                (agent._score_attack(public, private, o)[0], o["id"]) for o in opponents
            ]
            top = max(sc for sc, _ in scored)
            best_ids |= {pid for sc, pid in scored if sc >= top - 1e-9}
            counts[agent._pick_target(public, private, Card.ATTACK, opponents, set())] += 1

        # 分数差距虽小但真实存在，选中的必须始终是最高分那一档
        self.assertTrue(
            set(counts) <= best_ids,
            f"挑中了不是最高分的目标：选中 {dict(counts)}，最高分是 {best_ids}",
        )

    def test_scores_here_are_all_tiny_which_is_the_whole_point(self):
        """守住前提：这一局的分数区间确实比旧的绝对阈值还小。"""
        game = self._board(0)
        pool = ai.AgentPool(cfg=CFG, rng=random.Random(0))
        agent = pool.get(self.ME)
        public, private = game.public_state(), game.private_state(self.ME)
        agent.observe(public)
        scores = [
            agent._score_attack(public, private, o)[0]
            for o in public["players"] if o["id"] != self.ME
        ]
        self.assertLess(max(scores) - min(scores), 0.02)
        self.assertGreater(max(scores) - min(scores), 0.0)


class TestStoppingAnImminentWinner(unittest.TestCase):
    """有人下一步就夺冠时，AI 必须把干扰火力压在他身上。

    真实对局里踩过：对手已经站在主席门口，AI 还在到处乱打，
    甚至手上没干扰牌也不肯花钱重抽。
    """

    TOP = CFG.president_rank - 1

    def _table(self, seed, hand, my_money=40, know_his_money=True):
        game = Game(game_id="win", cfg=CFG, rng=random.Random(seed))
        for name in ("我", "快赢的", "路人甲", "路人乙"):
            game.add_player(name)
        game.start_game()
        ids = sorted(game.players)
        winner = game.players[ids[1]]
        winner.rank = self.TOP
        winner.merit = CFG.merit_cost(self.TOP) + 5
        winner.money = CFG.money_cost(self.TOP) + 5
        for pid in ids[2:]:
            game.players[pid].rank = 1
        me = game.players[ids[0]]
        me.rank, me.merit, me.money = 1, 5, my_money
        game.hands[ids[0]] = [DealtCard(card=Card(c), value=10) for c in hand]
        pool = ai.AgentPool(cfg=CFG, rng=random.Random(seed))
        agent = pool.get(ids[0])
        agent.observe(game.public_state())
        if know_his_money:
            agent.models.setdefault(ids[1], ai.OpponentModel()).money_est = winner.money
        return game, ids, pool

    def _targets(self, hand, **kw):
        hits = Counter()
        for seed in range(60):
            game, ids, pool = self._table(seed, hand, **kw)
            for pk in ai.choose(game, ids[0], pool):
                if pk["action"] in ("REPORT", "ATTACK"):
                    hits[(pk["action"], game.players[pk["target"]].name)] += 1
        return hits

    def test_report_goes_to_the_one_about_to_win(self):
        hits = self._targets(["REPORT"] + ["WORK"] * 5)
        self.assertEqual(hits[("REPORT", "快赢的")], 60)

    def test_attack_goes_to_the_one_about_to_win(self):
        hits = self._targets(["ATTACK"] + ["WORK"] * 5)
        self.assertEqual(hits[("ATTACK", "快赢的")], 60)

    def test_both_cards_stack_on_him_instead_of_spreading_out(self):
        """分散开来谁也拦不住，而他赢了桌上每个人都输。"""
        hits = self._targets(["REPORT", "ATTACK"] + ["WORK"] * 4)
        self.assertEqual(hits[("REPORT", "快赢的")], 60)
        self.assertEqual(hits[("ATTACK", "快赢的")], 60)

    def test_rich_uses_the_free_redraw_on_a_weak_hand(self):
        """富二代换牌不花钱：够门槛却没有晋升卡、或者一张能推进的牌都没有，就换。"""
        game = Game(game_id="rich", cfg=CFG, rng=random.Random(1))
        for name in ("我", "甲", "乙"):
            game.add_player(name)
        game.players[1].origin = Origin.RICH
        game.start_game()
        pool = ai.AgentPool(cfg=CFG, rng=random.Random(1))
        me = game.players[1]

        me.rank, me.merit, me.money = 0, CFG.merit_cost(0), 0
        game.hands[1] = [DealtCard(card=Card.WORK, value=6)] * CFG.hand_size
        self.assertTrue(ai.wants_redraw(game, 1, pool), "政绩够了手里却没晋升卡")

        me.merit = 0
        game.hands[1] = [DealtCard(card=Card.REPORT)] * CFG.hand_size
        self.assertTrue(ai.wants_redraw(game, 1, pool), "一张生产牌都没有")

        game.hands[1] = [DealtCard(card=Card.WORK, value=6)] * CFG.hand_size
        self.assertFalse(ai.wants_redraw(game, 1, pool), "手牌正常就别换")

        game.players[2].origin = None
        me.origin = None
        me.merit = CFG.merit_cost(0)
        me.money = 50
        self.assertFalse(ai.wants_redraw(game, 1, pool), "不是富二代就得花钱，没人要赢时不换")

    def test_redraws_when_holding_no_interference_card(self):
        game, ids, pool = self._table(0, ["WORK"] * 6)
        self.assertTrue(ai.wants_redraw(game, ids[0], pool))

    def test_does_not_waste_money_when_it_already_has_one(self):
        game, ids, pool = self._table(0, ["REPORT"] + ["WORK"] * 5)
        self.assertFalse(ai.wants_redraw(game, ids[0], pool))

    def test_does_not_redraw_when_it_cannot_afford_it(self):
        game, ids, pool = self._table(0, ["WORK"] * 6, my_money=0)
        self.assertFalse(ai.wants_redraw(game, ids[0], pool))

    def test_underestimating_his_hidden_cash_still_counts_as_about_to_win(self):
        """钱是暗的，估出来只会偏低——刹车不能等估计值过线才踩。

        复盘 F7KF 第 7 轮：老张真有 45 块（门槛 37），AI 估 26，
        于是「他下一步就夺冠」判 False，谁都没去拦，他当轮登顶。
        """
        game, ids, pool = self._table(0, ["WORK"] * 6, know_his_money=False)
        agent = pool.get(ids[0])
        model = agent.models.setdefault(ids[1], ai.OpponentModel())
        model.money_est = CFG.money_cost(self.TOP) * 0.7  # 估低三成
        winner = [o for o in game.public_state()["players"] if o["id"] == ids[1]][0]
        self.assertTrue(agent._about_to_win(winner, model))
        self.assertTrue(ai.wants_redraw(game, ids[0], pool))

    def test_a_hopeless_estimate_still_does_not_trigger_it(self):
        """留余量不等于见谁都当大敌——差得远的还是不该触发。

        "差得远" = 钱和政绩**两样都差**：一轮只有一个生产牌位子（另一张得是晋升卡），
        补不了两头。（只差钱的那种不算差得远——省级一张贪污期望 45，
        比门槛 37 还多，政绩够线的人当轮现贪现买就能登顶。）
        """
        game, ids, pool = self._table(0, ["WORK"] * 6, know_his_money=False)
        agent = pool.get(ids[0])
        model = agent.models.setdefault(ids[1], ai.OpponentModel())
        model.money_est = CFG.money_cost(self.TOP) * 0.2
        winner = dict(
            [o for o in game.public_state()["players"] if o["id"] == ids[1]][0],
            merit=CFG.merit_cost(self.TOP) - 1,
        )
        self.assertFalse(agent._about_to_win(winner, model))

    def test_one_production_card_short_still_counts(self):
        """复盘 QN78 第 9 轮：省级、钱够、政绩 30/43，一张埋头工作（省级 +15）就够。

        「生产牌 + 晋升卡」当轮补上当轮升，人类一眼看得出来；只看"现在够不够"的话
        AI 会判"不危险"，拿着攻击牌眼睁睁看他登顶。
        """
        game, ids, pool = self._table(0, ["WORK"] * 6)
        agent = pool.get(ids[0])
        model = agent.models[ids[1]]
        base = [o for o in game.public_state()["players"] if o["id"] == ids[1]][0]
        tc = CFG.merit_cost(self.TOP)
        one_work = agent._own_work_merit(0, self.TOP, None)
        self.assertTrue(agent._about_to_win(dict(base, merit=tc - 1), model))
        self.assertTrue(agent._about_to_win(dict(base, merit=int(tc - one_work)), model))
        self.assertFalse(agent._about_to_win(dict(base, merit=int(tc - one_work) - 1), model))

    def test_lookahead_needs_a_second_pick(self):
        """一轮只能打一张牌时没法「生产 + 晋升」，前瞻就不该开。"""
        import dataclasses
        cfg = dataclasses.replace(CFG, picks_per_round=1)
        agent = ai.SmartAgent(1, cfg=cfg)
        model = ai.OpponentModel()
        model.money_est = cfg.money_cost(self.TOP) * 10
        opp = {"id": 2, "rank": self.TOP, "merit": cfg.merit_cost(self.TOP) - 1, "origin": None}
        self.assertFalse(agent._about_to_win(opp, model))

    def test_it_keeps_redrawing_if_the_new_hand_is_also_useless(self):
        """换一次没摸到就放弃等于没救——他赢了，省下的钱一分也花不掉。"""
        game, ids, pool = self._table(0, ["WORK"] * 6)
        tries = 0
        for _ in range(ai.MAX_PANIC_REDRAWS):
            if not ai.wants_redraw(game, ids[0], pool):
                break
            game.hands[ids[0]] = [DealtCard(card=Card.WORK, value=10)] * CFG.hand_size
            game.redraw(ids[0])
            game.hands[ids[0]] = [DealtCard(card=Card.WORK, value=10)] * CFG.hand_size
            tries += 1
        self.assertGreater(tries, 1, "只换一次就认命了")

    def test_seized_cash_is_worthless_when_someone_is_one_step_from_winning(self):
        """用户原话：就算抄到钱了也没用啊 —— 他一登顶，游戏当场结束。

        这是举报选错人的根因：领跑者刚砸钱升完级，身上估着 0 块，
        「举报他」看起来一文不值，AI 转头去抄一个兜里有钱但毫无威胁的人。
        """
        game, ids, pool = self._table(0, ["REPORT"] + ["WORK"] * 5)
        agent = pool.get(ids[0])
        public, private = game.public_state(), game.private_state(ids[0])
        agent.observe(public)
        # 领跑者刚买完官，现钱估成 0；一个路人却攒了一大笔
        agent.models[ids[1]].money_est = 0.0
        agent.models[ids[2]].money_est = CFG.money_cost(1) * 3
        opp = {o["id"]: o for o in public["players"]}
        lead = agent._score_report(public, private, opp[ids[1]])[0]
        fat = agent._score_report(public, private, opp[ids[2]])[0]
        self.assertGreater(
            lead, fat, f"举报快赢的 {lead:.3f} 居然不如举报有钱的路人 {fat:.3f}"
        )

    def test_own_promotion_does_not_outweigh_stopping_the_winner(self):
        """复盘 QN78 第 10 轮：真人站在主席门口，老张手里有举报牌，
        却选了"贪污 + 自己凭政绩升一级"——那一级在他登顶之后一文不值。

        原样复现老张那一手：基层、政绩正好够升县级、手里两张大额贪污。
        修之前 60 局全是"贪污 + 通用升职"，一次都没举报。
        """
        hand = [("CORRUPT", 18), ("CORRUPT", 19), ("PROMOTE_ANY", 0),
                ("REPORT", 0), ("PROMOTE_MONEY", 0), ("WORK", 6)]
        reported = 0
        for seed in range(60):
            game, ids, pool = self._table(seed, [c for c, _ in hand])
            me = game.players[ids[0]]
            me.rank, me.merit, me.money = 0, CFG.merit_cost(0), 2
            game.hands[ids[0]] = [DealtCard(card=Card(c), value=v) for c, v in hand]
            pool.get(ids[0]).models[ids[1]].money_est = 33
            picks = ai.choose(game, ids[0], pool)
            reported += any(
                pk["action"] == "REPORT" and pk["target"] == ids[1] for pk in picks
            )
        self.assertGreaterEqual(reported, 55, f"60 局里只举报了快赢的 {reported} 次")

    def test_holding_both_it_throws_both_at_the_winner(self):
        """用户原话：同时有攻击和举报的话，真人会两张一起砸向要登顶的人，求稳妥。

        主席那一级攻击只按得住政绩升职、举报只抓得住贿赂升职，
        他两张晋升卡一起打（或者打通用升职）的话，单张牌拦不住。
        """
        both = 0
        for seed in range(60):
            game, ids, pool = self._table(seed, ["ATTACK", "REPORT", "WORK", "WORK", "WORK", "WORK"])
            picks = ai.choose(game, ids[0], pool)
            both += sorted((pk["action"], pk["target"]) for pk in picks) == [
                ("ATTACK", ids[1]), ("REPORT", ids[1])
            ]
        self.assertGreaterEqual(both, 55, f"60 局里只有 {both} 局两张一起压上去")

    def test_a_second_attack_adds_nothing(self):
        """第一刀已经按住政绩那条路了，第二刀不会多拦下什么——别把两张攻击都扔出去。

        （我身上没钱，埋头工作对我才有用；不然两张牌都是零分，比的就只是噪声。）
        """
        doubled = 0
        for seed in range(60):
            game, ids, pool = self._table(
                seed, ["ATTACK", "ATTACK", "WORK", "WORK", "WORK", "WORK"], my_money=0
            )
            picks = ai.choose(game, ids[0], pool)
            doubled += sum(pk["action"] == "ATTACK" for pk in picks) > 1
        self.assertEqual(doubled, 0)

    def test_cover_is_route_aware(self):
        """单张只拦一条路；两张一起才稳。贫农攻击挡不住，现贪的人一张举报就够。"""
        agent = ai.SmartAgent(1, cfg=CFG)
        rich = ai.OpponentModel()
        rich.money_est = CFG.money_cost(self.TOP) * 2
        broke = ai.OpponentModel()
        opp = {"id": 2, "rank": self.TOP, "merit": CFG.merit_cost(self.TOP), "origin": None}
        a, r, b = agent._endgame_cover(opp, rich)
        self.assertLess(a + r, b)
        _, r_broke, _ = agent._endgame_cover(opp, broke)
        self.assertGreater(r_broke, r, "他得现贪才凑得齐钱，举报单张就该几乎稳拦")
        # 默认（旧版贫农）：几个人攻击都按不住他升职，攻击把握为 0，只能靠举报
        a_p, r_p, b_p = agent._endgame_cover(dict(opp, origin="PEASANT"), rich)
        self.assertEqual(a_p, 0.0)
        self.assertEqual(r_p, r)
        self.assertAlmostEqual(b_p, r_p)
        # 只挡一个人的版本：攻击要靠别人也一起打才管用，把握打折但不是 0
        one = ai.SmartAgent(1, cfg=_dc.replace(CFG, origin_peasant_max_attackers=1))
        a1, r1, b1 = one._endgame_cover(dict(opp, origin="PEASANT"), rich)
        k = one.w.peasant_second_attacker
        self.assertAlmostEqual(a1, k * a)
        self.assertTrue(r1 < b1 < b)

    def test_decide_plays_the_top_of_its_own_scores(self):
        """decide 和复盘工具共用 score_combos。零噪声时 decide 挑的必须就是 last_scores
        里第一个最高分——两边要是又各算各的，复盘看到的打分就不是 AI 真用的那份了。"""
        import dataclasses
        weights = dataclasses.replace(ai.Weights(), noise=0.0)
        for seed in range(20):
            game, ids, _ = self._table(seed, ["ATTACK", "REPORT", "CORRUPT", "WORK", "WORK", "PROMOTE_ANY"])
            pool = ai.AgentPool(cfg=CFG, weights=weights, rng=random.Random(seed))
            picks = ai.choose(game, ids[0], pool)
            scores = pool.get(ids[0]).last_scores
            best = max(sc for sc, _ in scores)
            first_best = next(cards for sc, cards in scores if sc == best)
            self.assertEqual(
                sorted(pk["action"] for pk in picks), sorted(c.value for c in first_best)
            )

    def test_mid_game_cash_is_not_discounted(self):
        """这一折只能在终局生效。要是 payload 里少了 max_rounds 之类的字段，
        它会静默退化成"钱永远不值钱"，AI 从此不敢再举报捞钱——
        这种错不会抛异常，只能靠断言中局那一档必须是满值。"""
        game = Game(game_id="mid", cfg=CFG, rng=random.Random(5))
        for name in ("我", "甲", "乙"):
            game.add_player(name)
        game.start_game()
        ids = sorted(game.players)
        pool = ai.AgentPool(cfg=CFG, rng=random.Random(5))
        agent = pool.get(ids[0])
        public = game.public_state()
        agent.observe(public)
        self.assertEqual(agent._cash_horizon(public), 1.0)

    def test_does_not_redraw_when_nobody_is_close(self):
        game = Game(game_id="calm", cfg=CFG, rng=random.Random(3))
        for name in ("我", "甲", "乙"):
            game.add_player(name)
        game.start_game()
        ids = sorted(game.players)
        game.players[ids[0]].money = 40
        game.hands[ids[0]] = [DealtCard(card=Card.WORK, value=10)] * CFG.hand_size
        pool = ai.AgentPool(cfg=CFG, rng=random.Random(3))
        self.assertFalse(ai.wants_redraw(game, ids[0], pool))


class TestAuditFixes(unittest.TestCase):
    """audit.py 体检查出来的几处 AI 毛病，每处一条回归测试。"""

    def _game(self, n=3, seed=1):
        game = Game(game_id="fix", cfg=CFG, rng=random.Random(seed))
        for i in range(n):
            game.add_player(f"P{i + 1}")
        game.start_game()
        return game

    def test_noise_never_swaps_work_for_a_dead_promotion_card(self):
        """噪声只决定打什么干扰；同一种干扰搭配里，经济牌永远挑最好的。"""
        game = self._game()
        pool = ai.AgentPool(cfg=CFG, rng=random.Random(0),
                            weights=ai.Weights(noise=0.05))
        agent = pool.get(1)
        game.hands[1] = [DealtCard(Card.REPORT, 0), DealtCard(Card.WORK, 6),
                         DealtCard(Card.PROMOTE_MERIT, 0)] + [DealtCard(Card.WORK, 5)] * 3
        fixed = [(0.300, [Card.REPORT, Card.WORK]), (0.295, [Card.REPORT, Card.PROMOTE_MERIT])]
        agent.score_combos = lambda public, private: list(fixed)
        for _ in range(50):
            picks = agent.decide(game.public_state(), game.private_state(1))
            self.assertNotIn("PROMOTE_MERIT", [c for c, _ in picks])

    def test_second_report_is_scored_on_the_second_best_target(self):
        """第二张举报会换一个人打，只能算第二好的目标，不能再按最好的算一遍。"""
        game = self._game(n=4)
        agent = ai.AgentPool(cfg=CFG, rng=random.Random(0)).get(1)
        game.hands[1] = [DealtCard(Card.REPORT, 0)] * 2 + [DealtCard(Card.PASS, 0)] * 4 \
            if hasattr(Card, "PASS") else [DealtCard(Card.REPORT, 0)] * 2 + [DealtCard(Card.WORK, 0)] * 4
        pub, priv = game.public_state(), game.private_state(1)
        agent.observe(pub, priv)
        targets = sorted((agent._score_report(pub, priv, o)[0]
                          for o in pub["players"] if o["id"] != 1), reverse=True)
        scored = dict((tuple(c.value for c in cards), s)
                      for s, cards in agent.score_combos(pub, priv))
        both = scored[("REPORT", "REPORT")]
        econ, _ = agent._score_economy(pub, priv, [Card.REPORT, Card.REPORT], [0, 0])
        self.assertAlmostEqual(both - econ, targets[0] + targets[1], places=6)

    def test_work_still_counts_when_money_is_ready_but_buying_is_risky(self):
        """钱够了不等于能升（买官会被查实）：以前进度取 max(钱, 政绩)，埋头工作被估成 0 分。"""
        game = self._game()
        me = game.players[1]
        me.rank, me.money, me.merit = 1, CFG.money_cost(1) + 5, 0
        agent = ai.AgentPool(cfg=CFG, rng=random.Random(0)).get(1)
        pub, priv = game.public_state(), game.private_state(1)
        work, _ = agent._score_economy(pub, priv, [Card.WORK], [6])
        self.assertGreater(work, 0.05)

    def test_a_persistent_corruptor_reads_as_more_likely_to_corrupt(self):
        """个人档案：每轮都被传闻点名的人，比一般人更可能这轮还在贪。"""
        game = self._game()
        agent = ai.AgentPool(cfg=CFG, rng=random.Random(0)).get(1)
        pub = game.public_state()
        opp = next(o for o in pub["players"] if o["id"] == 2)
        clean = ai.OpponentModel(observed_rounds=6, dirty_rounds=0)
        dirty = ai.OpponentModel(observed_rounds=6, dirty_rounds=6)  # 轮轮被点名
        # 学习型 AI 被盯上会收手，档案只在最高那档往上翘（真人里有一路贪到底的）
        self.assertGreater(agent._read(pub, opp, dirty)[0], 1.1 * agent._read(pub, opp, clean)[0])

    def test_i_know_when_i_look_suspicious(self):
        """别人眼里的我比全桌平均可疑，我自己被查实的风险就该更高。"""
        game = self._game()
        agent = ai.AgentPool(cfg=CFG, rng=random.Random(0)).get(1)
        pub, priv = game.public_state(), game.private_state(1)
        agent.observe(pub, priv)
        base = agent._report_pressure(pub, priv)
        agent.self_model = ai.OpponentModel(observed_rounds=6, dirty_rounds=6)
        for o in pub["players"]:
            if o["id"] != 1:
                agent.models[o["id"]] = ai.OpponentModel(observed_rounds=6, dirty_rounds=0)
        self.assertGreater(agent._report_pressure(pub, priv), base * 1.15)

    def test_the_one_who_is_already_ready_is_stopped_first(self):
        """几个人同时可能登顶：两样都已经够了的人，比还要靠一张好牌的人更该先按住。"""
        top = CFG.president_rank - 1
        agent = ai.AgentPool(cfg=CFG, rng=random.Random(0)).get(1)
        mc, tc = CFG.money_cost(top), CFG.merit_cost(top)
        ready = {"id": 2, "rank": top, "merit": tc, "origin": None}
        close = {"id": 3, "rank": top, "merit": tc - 12, "origin": None}
        model = ai.OpponentModel(money_est=mc)
        self.assertGreater(agent._p_reach(ready, model), agent._p_reach(close, model))

    def test_the_runner_up_blocks_harder_than_the_backmarker(self):
        """拦下快登顶的人，对紧跟着的第二名值一整局；远远落后的人拦下来自己也赢不了。"""
        top = CFG.president_rank - 1
        game = self._game(n=3)
        leader, me = game.players[2], game.players[1]
        leader.rank, leader.merit = top, CFG.merit_cost(top)
        game.players[3].rank = top - 1
        values = {}
        for my_rank in (top, 0):  # 我是第二名 / 我在基层
            me.rank = my_rank
            agent = ai.AgentPool(cfg=CFG, rng=random.Random(0)).get(1)
            agent.observe(game.public_state(), game.private_state(1))
            values[my_rank] = agent._contender(2)
        self.assertAlmostEqual(values[top], 1.0)
        self.assertLess(values[0], 0.1)


class TestLearnedAgent(unittest.TestCase):
    """学习型 AI：眼睛和手写 AI 共用，判断按学到的权重。"""

    def _game(self, seed=3):
        game = Game(game_id="learn", cfg=CFG, rng=random.Random(seed))
        for i in range(4):
            game.add_player(f"P{i + 1}")
        game.start_game()
        return game

    def test_features_do_not_change_the_handwritten_scores(self):
        """打开特征只是顺便记录，手写 AI 的打分一分不变；"hand" 特征就是手写总分。"""
        game = self._game()
        pub, priv = game.public_state(), game.private_state(1)
        plain = ai.SmartAgent(1, cfg=CFG, rng=random.Random(0))
        plain.observe(pub, priv)
        a = plain.score_combos(pub, priv)
        feat = ai.SmartAgent(1, cfg=CFG, rng=random.Random(0))
        feat.want_features = True
        feat.observe(pub, priv)
        b = feat.score_combos(pub, priv)
        self.assertEqual([round(s, 9) for s, _ in a], [round(s, 9) for s, _ in b])
        self.assertEqual(len(feat.last_features), len(b))
        for (s, _), f in zip(b, feat.last_features):
            self.assertAlmostEqual(f["hand"], s)

    def test_learned_agent_plays_legal_cards_and_records_its_choices(self):
        game = self._game()
        pool = ai.AgentPool(cfg=CFG, rng=random.Random(0), policy={"hand": 50.0}, record=True)
        for _ in range(4):
            for pid in sorted(game.players):
                picks = ai.turn(game, pid, pool)
                game.select_actions(pid, picks)  # 不合法会抛 GameError
                game.lock_action(pid)
            game.reveal_event()
            game.resolve()
            if game.is_over:
                break
            game.advance_round()
        agent = pool.get(1)
        self.assertTrue(agent.trace, "record=True 时每次决策都要记下 ∇log π")
        self.assertIn("dirty", agent.trace[0])

    def test_table_mood_tracks_what_people_actually_do(self):
        """这桌最近举报多凶：每轮都有人被举报，估计就往上走。"""
        agent = ai.SmartAgent(1, cfg=CFG, rng=random.Random(0))
        before = agent.table["rep"]
        facts = {pid: {"player_id": pid, "attacked": False, "attack_merit_loss": 0,
                       "reported_by_player": True, "warnings_issued": 0, "demotion": "NONE"}
                 for pid in (1, 2, 3)}
        for _ in range(3):
            agent._update_table(facts, {}, {"wealth_top_ids": []})
        self.assertGreater(agent.table["rep"], before + 0.3)

    def test_read_calibrates_itself_to_the_table(self):
        """被举报的人实际查实得比我估的多，我的读法就往上调；少，就往下调。"""
        agent = ai.SmartAgent(1, cfg=CFG, rng=random.Random(0))
        self.assertAlmostEqual(agent._read_calibration(), 1.0)
        agent._cal_expect, agent._cal_hits = 10.0, 2.0   # 估了 10 个、只查实 2 个
        low = agent._read_calibration()
        agent._cal_expect, agent._cal_hits = 4.0, 9.0    # 估了 4 个、查实 9 个
        high = agent._read_calibration()
        self.assertLess(low, 0.5)
        self.assertGreater(high, 1.5)


    def test_sparse_features_do_not_break_the_gradient(self):
        """只有部分组合才有的特征（比如"举报打在会计身上"）：缺的当 0，梯度照算。"""
        game = Game(game_id="sparse", cfg=CFG, rng=random.Random(2))
        for i in range(6):
            game.add_player(f"P{i + 1}")
        for pid, oid in zip(sorted(game.players), CFG.origin_ids()):
            game.players[pid].origin = Origin(oid)
        game.start_game()
        game.hands[1] = [DealtCard(Card.REPORT, 0), DealtCard(Card.ATTACK, 0),
                         DealtCard(Card.WORK, 6), DealtCard(Card.CORRUPT, 18),
                         DealtCard(Card.PROMOTE_ANY, 0), DealtCard(Card.WORK, 5)]
        pool = ai.AgentPool(cfg=CFG, rng=random.Random(0), policy={"hand": 50.0}, record=True)
        ai.turn(game, 1, pool)
        trace = pool.get(1).trace[-1]
        self.assertTrue(any(k.startswith("rep_on_") for k in trace))


class TestDisallowedCardsAreReallyDisallowed(unittest.TestCase):
    def test_learned_agent_never_plays_a_disallowed_card(self):
        """消融"禁用攻击"的座位：手里全是攻击也不许打（宁可这轮不出牌）。"""
        game = Game(game_id="ban", cfg=CFG, rng=random.Random(1))
        for i in range(4):
            game.add_player(f"P{i + 1}")
        game.start_game()
        game.hands[1] = [DealtCard(Card.ATTACK, 0)] * 5 + [DealtCard(Card.WORK, 6)]
        for policy in (None, {"hand": 50.0, "n_attack": 100.0}):
            pool = ai.AgentPool(cfg=CFG, rng=random.Random(0), policy=policy, allow_attack=False)
            picks = ai.choose(game, 1, pool)
            self.assertNotIn("ATTACK", [p["action"] for p in picks])


class TestPolicyByOrigin(unittest.TestCase):
    def test_each_origin_uses_its_own_weights(self):
        """按出身选权重：会计那份权重偏爱贪污，会计就贪；别的出身用默认权重。"""
        game = Game(game_id="bo", cfg=CFG, rng=random.Random(1))
        for i in range(3):
            game.add_player(f"P{i + 1}")
        game.players[1].origin = Origin.ACCOUNTANT
        game.players[2].origin = Origin.GRINDER
        game.start_game()
        hand = [DealtCard(Card.CORRUPT, 18), DealtCard(Card.WORK, 6)] + [DealtCard(Card.WORK, 5)] * 4
        game.hands[1] = list(hand)
        game.hands[2] = list(hand)
        pool = ai.AgentPool(cfg=CFG, rng=random.Random(0), policy={"dirty": -100.0},
                            policy_by_origin={"ACCOUNTANT": {"dirty": 100.0}})
        acc = [p["action"] for p in ai.choose(game, 1, pool)]
        grind = [p["action"] for p in ai.choose(game, 2, pool)]
        self.assertIn("CORRUPT", acc)
        self.assertNotIn("CORRUPT", grind)


class TestDogpileNorm(unittest.TestCase):
    """AI_DOGPILE：有人在省级、政绩快够了，全桌一起按住他；没攻击牌就花钱重抽。"""

    def _game(self, cfg):
        game = Game(game_id="dog", cfg=cfg, rng=random.Random(2))
        for i in range(4):
            game.add_player(f"P{i + 1}")
        game.start_game()
        top = cfg.president_rank - 1
        leader = game.players[2]
        leader.rank, leader.merit = top, cfg.merit_cost(top) - 2
        return game

    def test_everyone_attacks_the_leader_near_the_line(self):
        import dataclasses

        cfg = dataclasses.replace(CFG, ai_dogpile=True)
        game = self._game(cfg)
        game.hands[1] = [DealtCard(Card.ATTACK, 0), DealtCard(Card.CORRUPT, 18)] + \
            [DealtCard(Card.WORK, 7)] * 4
        for policy in (None, {"hand": 50.0, "n_attack": -100.0}):
            pool = ai.AgentPool(cfg=cfg, rng=random.Random(0), policy=policy)
            picks = ai.choose(game, 1, pool)
            self.assertIn(("ATTACK", 2), [(p["action"], p["target"]) for p in picks])

    def test_no_attack_card_means_paying_to_redraw(self):
        import dataclasses

        cfg = dataclasses.replace(CFG, ai_dogpile=True)
        game = self._game(cfg)
        game.players[1].money = 50
        game.hands[1] = [DealtCard(Card.WORK, 7)] * 6
        pool = ai.AgentPool(cfg=cfg, rng=random.Random(0))
        self.assertTrue(ai.wants_redraw(game, 1, pool))
        no_dog = dataclasses.replace(CFG, ai_dogpile=False)
        off = ai.AgentPool(cfg=no_dog, rng=random.Random(0))
        game.cfg = no_dog
        self.assertFalse(ai.wants_redraw(game, 1, off))


class TestLearnedTargeting(unittest.TestCase):
    def test_target_choice_follows_learned_origin_weights(self):
        """学"打谁"：权重偏爱攻击官二代，它就去打官二代，而且这次选择会记进梯度。"""
        game = Game(game_id="tgt", cfg=CFG, rng=random.Random(3))
        for i in range(4):
            game.add_player(f"P{i + 1}")
        for pid, o in zip((2, 3, 4), (Origin.OFFICIAL, Origin.GRINDER, Origin.RICH)):
            game.players[pid].origin = o
        game.start_game()
        game.hands[1] = [DealtCard(Card.ATTACK, 0)] + [DealtCard(Card.WORK, 6)] * 5
        theta = {"hand": 50.0, "n_attack": 100.0, "A:o_OFFICIAL": 100.0}
        pool = ai.AgentPool(cfg=CFG, rng=random.Random(0), policy=theta, record=True)
        picks = ai.choose(game, 1, pool)
        self.assertIn(("ATTACK", 2), [(p["action"], p["target"]) for p in picks])
        self.assertTrue(any(k.startswith("A:") for g in pool.get(1).trace for k in g))
