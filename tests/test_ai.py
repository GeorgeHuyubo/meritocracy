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
        model.money_est = CFG.money_cost(top) * 10  # 钱那侧管够，只看政绩
        vip = {"id": 2, "rank": top, "merit": cut, "origin": "OFFICIAL"}
        plain = {"id": 3, "rank": top, "merit": cut, "origin": None}
        self.assertTrue(agent._about_to_win(vip, model), "官二代已经够线了")
        self.assertFalse(agent._about_to_win(plain, model))

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

    def test_a_grinder_values_work_more(self):
        agent = self._agent()
        self.assertGreater(
            agent._own_work_merit(0, 0, "GRINDER"), agent._own_work_merit(0, 0, None)
        )
        self.assertGreater(
            agent._own_work_merit(10, 2, "GRINDER"), agent._own_work_merit(10, 2, None)
        )

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
        """留余量不等于见谁都当大敌——差得远的还是不该触发。"""
        game, ids, pool = self._table(0, ["WORK"] * 6, know_his_money=False)
        agent = pool.get(ids[0])
        model = agent.models.setdefault(ids[1], ai.OpponentModel())
        model.money_est = CFG.money_cost(self.TOP) * 0.2
        winner = [o for o in game.public_state()["players"] if o["id"] == ids[1]][0]
        self.assertFalse(agent._about_to_win(winner, model))

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
