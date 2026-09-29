"""平衡分析工具自身的测试。

这些工具是用来**决定改不改游戏数值**的，所以它们错了比游戏错了更糟：
一个有偏的测量会让你信心十足地把规则改坏。真踩过两次 ——
一次是消融测试禁半桌，系统性低估"法不责众"型的牌；
一次是规则 A/B 靠一次性脚本 monkeypatch，脚本丢了、结论没法复核。
"""

from __future__ import annotations

import sys
import unittest
from fractions import Fraction
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

import analysis  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402

CFG = DEFAULT_CONFIG


class TestConfigOverride(unittest.TestCase):
    """`--set key=value`：规则 A/B 的入口，必须可信。"""

    def test_bool_accepts_the_usual_spellings(self):
        for raw, want in (("false", False), ("False", False), ("0", False),
                          ("no", False), ("off", False),
                          ("true", True), ("1", True), ("yes", True)):
            cfg = analysis.apply_overrides(
                CFG, [f"report_catches_bribery={raw}"]
            )
            self.assertIs(cfg.report_catches_bribery, want, raw)

    def test_int_float_fraction_and_str(self):
        cfg = analysis.apply_overrides(CFG, [
            "warnings_before_demotion=3",
            "report_reward_ratio=1/4",
            "attack_mode=denial",
            "reveal_event_seconds=0.5",
        ])
        self.assertEqual(cfg.warnings_before_demotion, 3)
        self.assertEqual(cfg.report_reward_ratio, Fraction(1, 4))
        self.assertEqual(cfg.attack_mode, "denial")
        self.assertAlmostEqual(cfg.reveal_event_seconds, 0.5)

    def test_it_does_not_mutate_the_original(self):
        """跑 A/B 时两个配置必须互不影响。"""
        before = CFG.report_catches_bribery
        analysis.apply_overrides(CFG, ["report_catches_bribery=false"])
        self.assertIs(CFG.report_catches_bribery, before)

    def test_no_settings_returns_the_same_config(self):
        self.assertIs(analysis.apply_overrides(CFG, []), CFG)

    def test_unknown_field_is_a_hard_error(self):
        """静默忽略会让整轮实验白跑，所以必须当场炸。"""
        with self.assertRaises(SystemExit) as caught:
            analysis.apply_overrides(CFG, ["report_catches_briberyy=false"])
        self.assertIn("report_catches_bribery", str(caught.exception))  # 给出拼写提示

    def test_bad_value_is_a_hard_error(self):
        with self.assertRaises(SystemExit):
            analysis.apply_overrides(CFG, ["report_catches_bribery=maybe"])
        with self.assertRaises(SystemExit):
            analysis.apply_overrides(CFG, ["warnings_before_demotion=两次"])

    def test_missing_equals_sign_is_a_hard_error(self):
        with self.assertRaises(SystemExit):
            analysis.apply_overrides(CFG, ["report_catches_bribery"])

    def test_compound_fields_say_why_they_are_unsupported(self):
        """list/dict 这种改不了，但要说清楚原因，而不是抛个看不懂的异常。"""
        with self.assertRaises(SystemExit) as caught:
            analysis.apply_overrides(CFG, ["rank_salary=1"])
        self.assertIn("config.py", str(caught.exception))

    def test_the_override_actually_reaches_the_rules(self):
        """光改 Config 对象没用，得确认结算真的按新规则走。"""
        import random

        import rules
        from models import Action, Card, PlayerState

        calm = rules.event_by_id("CALM", CFG)

        def bribe_under(cfg):
            mc = cfg.money_cost(0)
            target = PlayerState(id=1, name="T", rank=0, money=mc + 5)
            out = rules.resolve_round(
                [target, PlayerState(id=2, name="R", rank=0)],
                {1: [Action(Card.PROMOTE_MONEY)], 2: [Action(Card.REPORT, 1)]},
                calm, random.Random(1), cfg=cfg,
            )
            return out.outcomes[1]

        on = bribe_under(analysis.apply_overrides(
            CFG, ["report_catches_bribery=true"]))
        off = bribe_under(analysis.apply_overrides(
            CFG, ["report_catches_bribery=false"]))
        self.assertTrue(on.report_effective)   # 开着：光买官也抓得到
        self.assertGreater(on.bribe_lost, 0)
        self.assertFalse(off.report_effective)  # 关掉：这轮没贪就抓不到
        self.assertEqual(off.bribe_lost, 0)


class TestAblationMethodology(unittest.TestCase):
    """消融测试的默认设置要守住，这是上次踩坑的地方。"""

    def test_defaults_to_one_muted_seat(self):
        """禁半桌会给"人多才安全"的牌定错价（贪污：禁 1 席 -0.80，禁 3 席 +3.05）。"""
        import inspect

        sig = inspect.signature(analysis.analyse_ablation)
        self.assertEqual(sig.parameters["muted_seats"].default, 1)

    def test_it_reports_a_confidence_interval(self):
        """不给区间就会把噪声当结论 —— 同一配置换种子测出过 -4.24 和 -2.38。"""
        import random

        out = analysis.analyse_ablation(
            4, 12, CFG, random.Random(0), muted_seats=1
        )
        for label, row in out["cards"].items():
            self.assertIn("ci_half_width", row, label)
            self.assertGreater(row["ci_half_width"], 0, label)
            self.assertEqual(row["muted_seats"], 1)


class TestFunnel(unittest.TestCase):
    """金钱/政绩两条路线的漏斗 —— 比胜率消融精确得多的那把尺子。"""

    def test_it_measures_both_routes_end_to_end(self):
        import random

        out = analysis.analyse_funnel(4, 20, CFG, random.Random(3))
        for route in ("money", "merit"):
            r = out[route]
            self.assertGreater(r["plays"], 0, route)
            for key in ("survived_pct", "kept_pct", "end_to_end_pct"):
                self.assertGreaterEqual(r[key], 0.0)
                self.assertLessEqual(r[key], 100.0)

    def test_turning_off_bribery_catching_shows_up_in_the_funnel(self):
        """这正是这把尺子要回答的问题：关掉开关，金钱路线该变好走。"""
        import random

        off_cfg = analysis.apply_overrides(CFG, ["report_catches_bribery=false"])
        on = analysis.analyse_funnel(6, 120, CFG, random.Random(7))
        off = analysis.analyse_funnel(6, 120, off_cfg, random.Random(7))
        self.assertGreater(
            off["money"]["bribe_success_pct"], on["money"]["bribe_success_pct"],
            "关掉之后买官成功率应该明显上升",
        )


if __name__ == "__main__":
    unittest.main()
