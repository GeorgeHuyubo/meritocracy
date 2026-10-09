"""crosscheck.py（大模型局 vs Python 局同口径对比）和 scorecard.py（综合记分卡）的测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import balance_stats as bs  # noqa: E402
import crosscheck  # noqa: E402
import llm_play  # noqa: E402
import scorecard  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402

ORIGINS = list(DEFAULT_CONFIG.origin_ids())


class TestCrosscheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp())
        opts = {"llm_seats": 1, "crowd": "learned", "control": 1, "base_seed": 4}
        for g in range(12):
            llm_play.play_one((g, 4000 + g, "sonnet", "mock", cls.dir, [], "base", False, 0.0, opts))

    def test_mock_llm_equals_control_metric_by_metric(self):
        """mock 的"大模型"= 同种子学习型 AI：和对照局走同一套指标代码，每一项都必须完全相同。"""
        _, a = crosscheck.load_source(str(self.dir))
        _, b = crosscheck.load_source(f"control:{self.dir}")
        self.assertEqual(len(a), 12)
        self.assertEqual(len(b), 12)
        rows = crosscheck.compare(crosscheck.metrics(a), crosscheck.metrics(b))
        self.assertTrue(any(r["metric"] == "route_MONEY" for r in rows))
        for r in rows:
            if r["metric"] == "agree_with_python":
                continue
            self.assertAlmostEqual(r["diff"], 0.0, msg=f"{r['origin']} {r['metric']}")
        # 对照局里每个身份正好两局
        units = crosscheck.unit_rows(b)
        self.assertEqual(sorted(u["origin"] for u in units), sorted(ORIGINS * 2))

    def test_agreement_metric_from_base_picks(self):
        _, a = crosscheck.load_source(str(self.dir))
        m = crosscheck.metrics(a)
        self.assertAlmostEqual(m["ALL"]["agree_with_python"][0], 1.0)  # mock 本来就是程序 AI

    def test_legacy_adapter_reads_history_text(self):
        gm = {
            "game": 0, "rounds": 2, "president": False,
            "players": [{"pid": i, "name": n, "origin": o, "rank": r, "money": 10, "merit": 0, "winner": i == 1}
                        for i, n, o, r in ((1, "老张", "RED", 1), (2, "老李", "RICH", 1))],
            "history": ["第 1 轮（事件：经济下行）：老张 四处打点，晋升为县级干部。。结束后：老张 县级干部、老李 基层公务员",
                        "第 2 轮（事件：风平浪静）：老李 政绩卓著，晋升为县级干部。。结束后：老张 县级干部、老李 县级干部"],
            "decisions": [{"round": 1, "pid": 2, "picks": [{"action": "ATTACK", "target": 1}], "reason": "", "redraws": 0},
                          {"round": 2, "pid": 1, "picks": [{"action": "WORK", "target": None}], "reason": "", "redraws": 0}],
        }
        ng = crosscheck.normalize_llm(gm)
        self.assertTrue(ng["legacy"])
        rd = ng["rounds_data"]
        self.assertEqual(rd[0][1]["promotion"], "MONEY")
        self.assertEqual(rd[1][2]["promotion"], "MERIT")
        self.assertEqual(rd[0][1]["attacked_n"], 1)
        self.assertEqual(rd[1][2]["rank_after"], 1)
        rows = crosscheck.unit_rows([ng])
        self.assertEqual({r["origin"]: r["win"] for r in rows}, {"RED": 1.0, "RICH": 0.0})


class TestScorecard(unittest.TestCase):
    def _src(self, name, family, design, means, se=0.01, n=1000):
        return {"source": name, "family": family, "design": design, "rules_fp": None, "n_games": n,
                "per_origin": {o: {"n": n, "mean": m, "se": se} for o, m in zip(ORIGINS, means)}}

    def test_focal_effect_removes_skill_offset(self):
        s = self._src("搜索", "search", "focal", [0.30] * 6)
        for d, _, _ in scorecard.effects(s).values():
            self.assertAlmostEqual(d, 0.0)  # 搜索 AI 整体强，不能算成每个身份都强
        sym = self._src("自对弈", "py_learned", "symmetric", [0.30] * 6)
        self.assertAlmostEqual(scorecard.effects(sym)[ORIGINS[0]][0], 0.30 - 1 / 6)

    def test_verdict_rules(self):
        self.assertEqual(scorecard.verdict([(1, "py_learned"), (1, "search"), (1, "llm")]), "确认强")
        self.assertEqual(scorecard.verdict([(1, "py_learned"), (1, "py_learned"), (1, "py_learned")]), "偏强")
        self.assertEqual(scorecard.verdict([(-1, "llm")]), "偏弱")
        self.assertEqual(scorecard.verdict([(1, "llm"), (-1, "search")]), "有争议")
        self.assertEqual(scorecard.verdict([]), "无证据")

    def test_build_table(self):
        hi = [0.25, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 0.08]
        srcs = [self._src("A", "py_learned", "symmetric", hi), self._src("B", "search", "symmetric", hi),
                self._src("C", "llm", "symmetric", hi), self._src("D", "llm", "symmetric", hi[::-1])]
        md, data = scorecard.build(srcs, {"D"}, None, 3, 2)
        self.assertEqual(data["table"][ORIGINS[0]]["verdict"], "确认强")
        self.assertEqual(data["table"][ORIGINS[-1]]["verdict"], "确认弱")
        self.assertEqual(data["table"][ORIGINS[2]]["verdict"], "无证据")
        self.assertIn("仅展示", md)


if __name__ == "__main__":
    unittest.main()
