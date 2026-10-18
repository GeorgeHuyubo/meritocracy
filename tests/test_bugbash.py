"""bugbash.py（规则裁判）的自检：故意把引擎改坏几处，裁判得抓得到；正常的局面不能乱报。"""

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bugbash  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402
import models  # noqa: E402
import rules  # noqa: E402
from models import DemotionKind  # noqa: E402


def kinds(bash) -> set[str]:
    return {f["kind"] for f in bash.findings if not f["kind"].startswith("spec:")}


class TestRefereeCatchesBrokenEngine(unittest.TestCase):
    def test_clean_rounds_have_no_findings(self):
        cases = [
            # 贫农挨一刀照样凭政绩升
            ([{"origin": "PEASANT", "merit": 20}, {"origin": "RICH"}],
             {1: [("WORK", 6, None), ("PROMOTE_MERIT", 0, None)], 2: [("ATTACK", 0, 1)]}, "CALM"),
            # 贪了被举报：没收、分赃、记警告
            ([{"origin": "OFFICIAL", "money": 3}, {"origin": "GRINDER"}],
             {1: [("CORRUPT", 18, None)], 2: [("REPORT", 0, 1), ("WORK", 6, None)]}, "BOOM"),
            # 卷王双埋头被抢一刀，再 ×4
            ([{"origin": "GRINDER"}, {"origin": "RED"}],
             {1: [("WORK", 6, None), ("WORK", 7, None)], 2: [("ATTACK", 0, 1)]}, "KEY_PROJECT"),
            # 先买官、再按新官职贪；没人查
            ([{"origin": "RICH", "money": 20}, {"origin": "ACCOUNTANT"}],
             {1: [("PROMOTE_MONEY", 0, None), ("CORRUPT", 17, None)], 2: [("CORRUPT", 16, None)]}, "RECESSION"),
            # 戴帽子：没干正事的人挨攻击
            ([{"origin": "RED", "merit": 10, "rank": 2}, {"origin": "OFFICIAL", "rank": 1}],
             {1: [("CORRUPT", 18, None)], 2: [("ATTACK", 0, 1)]}, "CALM"),
            # 红二代警告攒满交打点费保官
            ([{"origin": "RED", "money": 30, "rank": 2, "warnings": 1}, {"origin": "PEASANT"}],
             {1: [("CORRUPT", 16, None)], 2: [("REPORT", 0, 1)]}, "CALM"),
            # 举报 -> 一纸调令 -> 埋头工作：调令等赃款，只能走钱（后面挣的政绩用不上）
            ([{"origin": "RED", "merit": 17, "money": 1}, {"rank": 2}],
             {1: [("REPORT", 0, 2), ("PROMOTE_FAMILY", 0, None), ("WORK", 6, None)],
              2: [("CORRUPT", 20, None), ("CORRUPT", 20, None)]}, "CALM"),
        ]
        for players, picks, ev in cases:
            with self.subTest(picks=picks):
                bash, _ = bugbash.scenario(players, picks, ev)
                self.assertEqual(kinds(bash), set(), bash.findings)

    def test_waiting_card_needs_current_money_too(self):
        """等赃款的晋升卡：被查实交了打点费、现在的钱不够，就升不了（裁判也得这么判）。"""
        cfg = dataclasses.replace(DEFAULT_CONFIG, origin_red_shield_fee_strict=False)
        bash, _ = bugbash.scenario(
            [{"origin": "RED", "money": 5, "rank": 1, "warnings": 1}, {}, {}],
            {1: [("GRAFT", 8, None), ("REPORT", 0, 2), ("PROMOTE_FAMILY", 0, None)],
             2: [("CORRUPT", 20, None), ("CORRUPT", 20, None)], 3: [("REPORT", 0, 1)]}, cfg=cfg)
        self.assertEqual(kinds(bash), set(), bash.findings)

    def test_waiting_card_using_later_merit_is_caught(self):
        real = rules._resolve_promotion_card
        with mock.patch.object(rules, "_resolve_promotion_card",
                               lambda *a, merit_at=None, **k: real(*a, **k)):
            bash, _ = bugbash.scenario(
                [{"origin": "RED", "merit": 17, "money": 1}, {"rank": 2}],
                {1: [("REPORT", 0, 2), ("PROMOTE_FAMILY", 0, None), ("WORK", 6, None)],
                 2: [("CORRUPT", 20, None), ("CORRUPT", 20, None)]})
        self.assertIn("resolve:promotion", kinds(bash))

    def test_peasant_immunity_broken(self):
        with mock.patch.object(rules, "smear_resisted", lambda *a, **k: False):
            bash, _ = bugbash.scenario([{"origin": "PEASANT", "merit": 20}, {"origin": "RICH"}],
                                       {1: [("WORK", 6, None), ("PROMOTE_MERIT", 0, None)],
                                        2: [("ATTACK", 0, 1)]})
        # 第一处对不上的是抢功（政绩没花掉，被抢的就多），晋升那一项也一起列出来
        diffs = [f["detail"]["diffs"] for f in bash.findings if f["kind"].startswith("resolve:")]
        self.assertTrue(any("promotion" in d for d in diffs), bash.findings)

    def test_wrong_overflow_divisor(self):
        real = rules.overflow_after_promotion
        with mock.patch.object(rules, "overflow_after_promotion",
                               lambda remaining, cfg=None, divisor=None: real(remaining, cfg, 3 if divisor == 5 else divisor)):
            bash, _ = bugbash.scenario([{"merit": 30}, {}], {1: [("PROMOTE_MERIT", 0, None)]})
        self.assertIn("resolve:promotion", kinds(bash))

    def test_clean_player_caught(self):
        with mock.patch.object(rules, "classify_report", lambda *a, **k: DemotionKind.MINOR):
            bash, _ = bugbash.scenario([{}, {}], {1: [("WORK", 6, None)], 2: [("REPORT", 0, 1)]})
        self.assertIn("resolve:report", kinds(bash))

    def test_gossip_names_wrong_person(self):
        with mock.patch.object(rules, "wealth_broadcast_detail", lambda income, cfg, rng=None: (["x"], ["老李"])):
            bash, _ = bugbash.scenario([{}, {}], {1: [("CORRUPT", 18, None)], 2: [("WORK", 6, None)]})
        self.assertIn("resolve:gossip", kinds(bash))

    def test_ledger_row_missing(self):
        real = models.PlayerRoundOutcome.ledger_lines

        def drop_salary(self, rank_name):
            return [r for r in real(self, rank_name) if r["label"] != "贪污进账"]

        with mock.patch.object(models.PlayerRoundOutcome, "ledger_lines", drop_salary):
            bash, _ = bugbash.scenario([{}, {}], {1: [("CORRUPT", 18, None)]})
        self.assertIn("ledger:mismatch", kinds(bash))


class TestRandomGames(unittest.TestCase):
    def test_games_run_without_crashing_the_referee(self):
        res = bugbash.run_chunk((7, list(range(12))))
        bad = [f for f in res["findings"] if f["kind"] in ("exception:referee", "exception:game")]
        self.assertEqual(bad, [])
        self.assertEqual(res["stats"]["games"], 12)
        self.assertGreater(res["stats"]["compared"], 0)

    def test_repro_spec_is_deterministic(self):
        a, b = bugbash.game_spec(3, 41), bugbash.game_spec(3, 41)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
