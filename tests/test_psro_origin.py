"""psro_origin.py（按身份的 PSRO）和 AI 按身份抽打法的测试。"""

from __future__ import annotations

import contextlib
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ai  # noqa: E402
import psro_origin  # noqa: E402
import telemetry as T  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class TestReplicatorJoint(unittest.TestCase):
    def test_each_origin_converges_on_its_own_game(self):
        def payoff(sig):
            a = sig["A"]  # A：剪刀石头布
            return {"A": [a[2] - a[1], a[0] - a[2], a[1] - a[0]], "B": [0.1, 0.3]}  # B：第二个占优

        avg, trace = psro_origin.replicator_joint(payoff, {"A": [0.6, 0.3, 0.1], "B": [0.5, 0.5]},
                                                  rounds=3000, eta=0.2)
        for x in avg["A"]:
            self.assertAlmostEqual(x, 1 / 3, delta=0.05)
        self.assertGreater(avg["B"][1], 0.9)
        self.assertEqual(len(trace), 3000)


class TestMixtureByOrigin(unittest.TestCase):
    def test_each_seat_samples_from_its_own_origin(self):
        a, b = {"hand": 50.0}, {"hand": 10.0}
        prof = {o: [(a, 1.0), (b, 0.0)] if o == "RICH" else [(a, 0.0), (b, 1.0)]
                for o in DEFAULT_CONFIG.origin_ids()}
        seats = list(DEFAULT_CONFIG.origin_ids())
        game = T.new_game(DEFAULT_CONFIG, seats, 5)
        pool = ai.AgentPool(cfg=DEFAULT_CONFIG, rng=random.Random(0), mixture_by_origin=prof)
        for pid in game.players:
            ai.turn(game, pid, pool)
        for pid, pl in game.players.items():
            want = a if pl.origin.value == "RICH" else b
            self.assertIs(pool.agents[pid].theta_by_origin[pl.origin.value], want)

    def test_profile_file_loads_through_make_pool(self):
        import dataclasses
        a = {"hand": 50.0}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "prof.json"
            path.write_text(json.dumps({"profile": {o: [{"name": "x", "weight": 1.0, "theta": a}]
                                                    for o in DEFAULT_CONFIG.origin_ids()}}))
            self.assertEqual(set(ai.load_profile(str(path))), set(DEFAULT_CONFIG.origin_ids()))
            self.assertEqual(ai.load_policy(str(path)), a)
            pool = ai.make_pool(dataclasses.replace(DEFAULT_CONFIG, ai_policy=str(path)), random.Random(0))
            self.assertIsNotNone(pool.mixture_by_origin)


class TestPsroOriginSmoke(unittest.TestCase):
    def test_one_iteration_with_seeded_best_responses(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            theta = ai.load_policy(str(ROOT / "policies" / "best.json"))
            for o in DEFAULT_CONFIG.origin_ids():
                (d / f"seed_{o}.json").write_text(json.dumps({"theta": theta}))
            pop = d / "pop"
            argv = ["run", "--pop", str(pop), "--seed-brs", str(d / "seed_{origin}.json"),
                    "--iters", "1", "--solve-rounds", "1", "--games", "3", "--exploit-games", "3",
                    "--workers", "2"]
            with contextlib.redirect_stdout(io.StringIO()):
                psro_origin.main(argv)
            meta = psro_origin.load_meta(pop)
            self.assertEqual(meta["iter"], 1)
            for o in DEFAULT_CONFIG.origin_ids():
                self.assertEqual(meta["pools"][o][-1]["name"], f"{o}_br0")
                self.assertAlmostEqual(sum(meta["weights"][o]), 1.0, places=4)
            self.assertIn("gains", meta["history"][0])
            self.assertIn("可被剥削度", psro_origin.status_text(pop))


if __name__ == "__main__":
    unittest.main()
