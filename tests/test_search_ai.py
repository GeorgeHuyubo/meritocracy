"""搜索型 AI（search_ai.py）、焦点座位框架（telemetry.py）、探针（probe.py）、统计小工具的测试。

最要紧的一条是**信息泄露**：搜索 AI 的决定只能取决于它看得到的东西（公开状态 + 自己的私密信息），
对手的手牌、存款、本轮事件、随机数状态换成别的，结果必须一模一样。
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import random
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ai  # noqa: E402
import balance_stats as bs  # noqa: E402
import probe  # noqa: E402
import rules  # noqa: E402
import search_ai  # noqa: E402
import telemetry as T  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402
from models import Card, DealtCard  # noqa: E402

CFG = DEFAULT_CONFIG
SMALL = search_ai.SearchConfig(budget=8, k=3, n_samples=8, money_model="lognormal")


def table(g=0, seed=1):
    fo, fs, seats = T.focal_assignment(g, seed, CFG)
    gs = T.game_seed(seed, g)
    return T.new_game(CFG, seats, gs), fs + 1, gs


class TestFocalFramework(unittest.TestCase):
    def test_36_games_cover_every_origin_seat_once(self):
        cells = Counter()
        for g in range(36):
            fo, fs, seats = T.focal_assignment(g, 1, CFG)
            self.assertEqual(seats[fs], fo)
            self.assertEqual(sorted(seats), sorted(CFG.origin_ids()))
            cells[(fo, fs)] += 1
        self.assertEqual(len(cells), 36)
        self.assertTrue(all(v == 1 for v in cells.values()))

    def test_reseed_gives_same_hands_despite_different_play(self):
        """两局前面打法不同（不同 AI 种子），每一轮发牌前重新播种，第 r 轮的手牌仍然相同（没换牌的人）。"""
        hands = []
        for pseed in (1, 2):
            game, _, gs = table(4)
            pool = ai.make_pool(CFG, random.Random(pseed))
            seen = {}

            def dec(gm, pid, pool=pool, seen=seen):
                if gm.round_number not in seen:
                    seen[gm.round_number] = {p: [(d.card.value, d.value) for d in h]
                                             for p, h in gm.hands.items()}
                return ai.choose(gm, pid, pool)  # 不换牌，手牌只取决于种子

            T.drive_game(game, {p: dec for p in game.players}, gs)
            hands.append(seen)
        common = set(hands[0]) & set(hands[1])
        self.assertGreaterEqual(len(common), 3)
        for r in common:
            self.assertEqual(hands[0][r], hands[1][r], f"第 {r} 轮")

    def test_round_record_matches_game_and_is_json(self):
        game, _, gs = table(1)
        pool = ai.make_pool(CFG, random.Random(0))
        tel = T.drive_game(game, {p: (lambda gm, pid: ai.turn(gm, pid, pool)) for p in game.players}, gs)
        self.assertEqual(len(tel), game.round_number)
        last = {p["pid"]: p for p in tel[-1]["players"]}
        for pid, pl in game.players.items():
            self.assertEqual(last[pid]["rank_after"], pl.rank)
            self.assertEqual(last[pid]["merit_after"], pl.merit)
        self.assertEqual(json.loads(json.dumps(tel)), tel)
        summ = T.game_summary(game)
        self.assertEqual(summ["final_standing"][0] in game.winners, True)


class TestDeterminize(unittest.TestCase):
    def setUp(self):
        self.game, self.me, gs = table(0)
        self.s = search_ai.Searcher(self.me, CFG, SMALL, base_rng_seed=gs)
        self.s.base_pool.get(self.me).observe(self.game.public_state(), self.game.private_state(self.me))

    def test_keeps_public_and_my_private(self):
        g2 = self.s.determinize(self.game, self.me, random.Random(3))
        self.assertEqual(g2.public_state(), self.game.public_state())
        a, b = g2.private_state(self.me), self.game.private_state(self.me)
        self.assertEqual(a["hand"], b["hand"])
        self.assertEqual(a["money"], b["money"])

    def test_resamples_hidden(self):
        hands, events = set(), set()
        grinder = next(p for p, pl in self.game.players.items() if pl.origin.value == "GRINDER")
        for i in range(60):
            g2 = self.s.determinize(self.game, self.me, random.Random(i))
            opp = next(o for o in g2.players if o != self.me)
            hands.add(tuple((d.card.value, d.value) for d in g2.hands[opp]))
            events.add(g2.next_event.id)
            for o, pl in g2.players.items():
                self.assertGreaterEqual(pl.money, 0)
            if grinder != self.me:
                self.assertGreaterEqual(sum(d.card is Card.WORK for d in g2.hands[grinder]),
                                        CFG.origin_grinder_min_work)
            self.assertTrue(all(not g2.selections[o].picks for o in g2.players if o != self.me))
        self.assertGreater(len(hands), 10)
        self.assertGreater(len(events), 1)


class TestInformationSetInvariance(unittest.TestCase):
    """同样的公开状态 + 同样的我的私密信息，对手的手牌 / 存款 / 本轮事件 / 随机数 / 已选的牌都不同：
    搜索结果必须一模一样。"""

    def test_hidden_info_does_not_leak(self):
        game_a, me, gs = table(3)
        game_b = copy.deepcopy(game_a)
        rng = random.Random(99)
        game_b.rng = random.Random(12345)
        for o, pl in game_b.players.items():
            if o == me:
                continue
            game_b.hands[o] = rules.deal_hand_for(pl, rng, CFG)
            pl.money += 23
        game_b.next_event = rules.pick_event(rng, CFG)
        # 对手已经出过牌（探针里陪练先出）——搜索 AI 不能看见
        crowd = ai.make_pool(CFG, random.Random(5))
        for o in game_b.players:
            if o != me:
                game_b.select_actions(o, ai.choose(game_b, o, crowd))
        self.assertEqual(game_a.public_state(), game_b.public_state())
        self.assertEqual(game_a.private_state(me), game_b.private_state(me))
        scfg = search_ai.SearchConfig(budget=12, k=3, n_samples=8, money_model="lognormal",
                                      include_redraw=False, skip_top_prob=1.1)
        out = []
        for g in (game_a, game_b):
            s = search_ai.Searcher(me, CFG, scfg, base_rng_seed=gs)
            picks = s.turn(g, me)
            out.append((picks, s.log))
        self.assertEqual(out[0][0], out[1][0])
        self.assertEqual(out[0][1], out[1][1])
        self.assertTrue(out[0][1][0]["searched"])


class TestSearcherBehaviour(unittest.TestCase):
    def test_budget_zero_equals_base(self):
        for g in (0, 2, 5):
            a = probe.play_probe_game(CFG, g, 1, "learned", "learned", SMALL)
            b = probe.play_probe_game(CFG, g, 1, "search", "learned",
                                      search_ai.SearchConfig(budget=0))
            for k in ("winners", "place", "rounds", "focal_cards"):
                self.assertEqual(a[k], b[k], f"g={g} {k}")

    def test_full_games_are_legal(self):
        """富二代（免费换牌）、红二代、会计各打一整局，小预算搜索，不能抛异常、出牌必须合法。"""
        for g in (0, 2, 5):
            r = probe.play_probe_game(CFG, g, 2, "search", "learned", SMALL)
            self.assertGreater(r["rounds"], 0)
            self.assertGreater(r["search_stats"]["searched"], 0)

    def test_common_random_numbers(self):
        game, me, gs = table(1)
        s = search_ai.Searcher(me, CFG, SMALL, base_rng_seed=gs)
        pub, priv = game.public_state(), game.private_state(me)
        s.base_pool.get(me).observe(pub, priv)
        for o in game.players:
            if o != me:
                s.shadow.get(o).observe(pub, None)
        c = s._candidates(pub, priv)[0]
        twin = search_ai.Candidate(c.key, c.picks, c.prior)
        s._search(game, me, [c, twin])
        self.assertEqual(c.rewards, twin.rewards)

    def test_takes_the_immediate_win(self):
        game, me, gs = table(0)
        top = CFG.president_rank - 1
        p = game.players[me]
        p.rank, p.merit, p.money = top, 200, 200
        game.hands[me] = [DealtCard(Card.PROMOTE_ANY, 0), DealtCard(Card.WORK, 6),
                          DealtCard(Card.REPORT, 0), DealtCard(Card.CORRUPT, 12),
                          DealtCard(Card.ATTACK, 0), DealtCard(Card.WORK, 6)]
        s = search_ai.Searcher(me, CFG, search_ai.SearchConfig(
            budget=16, k=4, n_samples=8, money_model="lognormal", skip_top_prob=1.1), base_rng_seed=gs)
        picks = s.turn(game, me)
        self.assertIn(Card.PROMOTE_ANY, [x["action"] for x in picks])

    def test_money_calibration_shape(self):
        res = search_ai.calibrate_money(3)
        self.assertIn("residuals", res)
        self.assertTrue(any(k.endswith("|*|*") for k in res["residuals"]))
        m = search_ai.sample_money(20.0, 1, 2, None, random.Random(0),
                                   search_ai.SearchConfig(), res["residuals"])
        self.assertGreaterEqual(m, 0)


class TestProbeSummary(unittest.TestCase):
    def test_aa_run_has_zero_uplift_and_resume_skips(self):
        with tempfile.TemporaryDirectory() as d:
            with contextlib.redirect_stdout(io.StringIO()):
                probe.main(["--aa", "--games", "6", "--workers", "1", "--out", d])
            rows = probe.load_rows(Path(d))
            self.assertEqual(len(rows), 12)
            s = json.loads((Path(d) / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(s["uplift"]["search-learned"]["ALL"]["delta"], 0.0)
            with contextlib.redirect_stdout(io.StringIO()):
                probe.main(["--resume", d, "--workers", "1"])
            self.assertEqual(len(probe.load_rows(Path(d))), 12)
            src = json.loads((Path(d) / "source_search.json").read_text(encoding="utf-8"))
            self.assertEqual(src["design"], "focal")
            self.assertEqual(sum(v["n"] for v in src["per_origin"].values()), 6)


class TestBalanceStats(unittest.TestCase):
    def test_basics(self):
        m, se = bs.mean_se([0, 1, 0, 1])
        self.assertAlmostEqual(m, 0.5)
        self.assertGreater(se, 0)
        d, dse = bs.paired_diff([1, 1, 0], [0, 1, 0])
        self.assertAlmostEqual(d, 1 / 3)
        r, rse = bs.ratio_se([1, 2, 3], [2, 4, 6])
        self.assertAlmostEqual(r, 0.5)
        self.assertAlmostEqual(rse, 0.0)
        self.assertEqual(bs.significant(0.1, 0.01), 1)
        self.assertEqual(bs.significant(-0.1, 0.01), -1)
        self.assertEqual(bs.significant(0.01, 0.1), 0)


if __name__ == "__main__":
    unittest.main()


class TestReportPressureCache(unittest.TestCase):
    def test_cached_scores_equal_uncached(self):
        """score_combos 里的举报风险缓存只是提速：打分必须和不缓存时一模一样。"""
        game, me, gs = table(2)
        pool = ai.make_pool(CFG, random.Random(1))
        T.drive_game(game, {p: (lambda gm, pid: ai.turn(gm, pid, pool)) for p in game.players}, gs)
        # 拿一局打完的中途状态不方便，直接用新开一局第 1 轮 + 一个有记忆的 agent
        game2, me2, _ = table(3)
        agent = pool.get(me2)
        pub, priv = game2.public_state(), game2.private_state(me2)
        agent.observe(pub, priv)
        cached = agent.score_combos(pub, priv)
        orig = agent._report_pressure
        try:
            agent._report_pressure = agent._report_pressure_uncached
            plain = agent.score_combos(pub, priv)
        finally:
            agent._report_pressure = orig
        self.assertEqual(cached, plain)
