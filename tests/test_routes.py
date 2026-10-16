"""routes.py（政绩 / 金钱路线报告）的测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import routes  # noqa: E402


class TestRoutes(unittest.TestCase):
    def test_group_of(self):
        self.assertEqual(routes.group_of({"MERIT": 3, "MONEY": 1}), "政绩为主")
        self.assertEqual(routes.group_of({"MERIT": 1, "MONEY": 1}), "两边都有")
        self.assertEqual(routes.group_of({"MONEY": 2, "BOTH": 1}), "金钱为主")
        self.assertIsNone(routes.group_of({"TENURE": 2}))

    def test_tiny_run_renders(self):
        s = routes.summarize(routes.run("", games=6, workers=1))
        self.assertEqual(s["games"], 6)
        md = routes.render(s)
        for key in ("按主要升职路线分组", "各身份", "只走政绩", "卷王升不了职时双埋头"):
            self.assertIn(key, md)
        self.assertAlmostEqual(sum(v["share"] for v in s["groups"].values()), 1.0, delta=0.2)


if __name__ == "__main__":
    unittest.main()
