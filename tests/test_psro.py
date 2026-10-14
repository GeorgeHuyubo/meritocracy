"""psro.py（逼近纳什均衡）和 AgentPool 混合打法的测试。"""

from __future__ import annotations

import contextlib
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
import psro  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class TestReplicator(unittest.TestCase):
    def test_rock_paper_scissors_averages_to_one_third(self):
        """剪刀石头布：σ 本身绕圈，平均值收敛到 (1/3, 1/3, 1/3)。"""
        def payoff(s):  # 0=石头 1=布 2=剪刀：石头赢剪刀、布赢石头、剪刀赢布
            return [s[2] - s[1], s[0] - s[2], s[1] - s[0]]

        avg, trace = psro.replicator(payoff, [0.6, 0.3, 0.1], rounds=3000, eta=0.2)
        for x in avg:
            self.assertAlmostEqual(x, 1 / 3, delta=0.05)
        self.assertEqual(len(trace), 3000)

    def test_dominant_strategy_takes_over(self):
        avg, _ = psro.replicator(lambda s: [0.30, 0.15, 0.10], [1 / 3] * 3, rounds=200, eta=5.0)
        self.assertGreater(avg[0], 0.8)


class TestMixturePool(unittest.TestCase):
    def setUp(self):
        self.a = {"hand": 50.0}
        self.b = {"hand": 10.0}

    def test_weight_one_always_picks_it(self):
        pool = ai.AgentPool(cfg=DEFAULT_CONFIG, rng=random.Random(0), policy=self.a,
                            mixture=[(self.a, 1.0), (self.b, 0.0)])
        for pid in range(1, 30):
            self.assertIs(pool.get(pid).theta, self.a)

    def test_frequencies_follow_weights(self):
        pool = ai.AgentPool(cfg=DEFAULT_CONFIG, rng=random.Random(1), policy=self.a,
                            mixture=[(self.a, 0.7), (self.b, 0.3)])
        picks = Counter(id(pool.get(pid).theta) for pid in range(1, 2001))
        self.assertAlmostEqual(picks[id(self.a)] / 2000, 0.7, delta=0.04)

    def test_mixture_file_is_loaded_by_make_pool(self):
        import dataclasses
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "mix.json"
            path.write_text(json.dumps({"mixture": [{"name": "x", "weight": 0.25, "theta": self.a},
                                                    {"name": "y", "weight": 0.75, "theta": self.b}]}))
            theta, by_origin, mix = ai.load_policy_file(str(path))
            self.assertEqual(theta, self.b)  # 主权重 = 比例最大的那一份
            self.assertEqual([w for _, w in mix], [0.25, 0.75])
            self.assertEqual(ai.load_policy(str(path)), self.b)
            pool = ai.make_pool(dataclasses.replace(DEFAULT_CONFIG, ai_policy=str(path)), random.Random(0))
            self.assertEqual(len(pool.mixture), 2)


class TestPsroSmoke(unittest.TestCase):
    def test_one_tiny_iteration_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            pop = Path(d) / "pop"
            argv = ["run", "--pop", str(pop), "--start", str(ROOT / "policies" / "best.json"),
                    "--iters", "1", "--br-minutes", "0.02", "--solve-rounds", "2", "--games", "6",
                    "--exploit-games", "6", "--workers", "2"]
            with contextlib.redirect_stdout(io.StringIO()):
                psro.main(argv)
            meta = psro.load_meta(pop)
            self.assertEqual(meta["iter"], 1)
            self.assertEqual(len(meta["policies"]), 2)
            self.assertAlmostEqual(sum(meta["weights"]), 1.0, places=4)
            h = meta["history"][0]
            self.assertIn("exploitability", h)
            self.assertEqual(len(h["solve"]), 2)
            with contextlib.redirect_stdout(io.StringIO()):
                psro.main(argv + ["--resume"])  # 已经跑完 1 轮：什么也不做
            self.assertEqual(len(psro.load_meta(pop)["policies"]), 2)
            self.assertIn("可被剥削度", psro.status_text(pop))


if __name__ == "__main__":
    unittest.main()
