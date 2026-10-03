"""replay.py 测试：库里存下来的一局能原样重演，AI 决策逐个复现。"""

from __future__ import annotations

import contextlib
import io
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ai  # noqa: E402
import analysis  # noqa: E402
import replay  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402
from game import Game  # noqa: E402
from models import Origin  # noqa: E402
from storage import GameStore  # noqa: E402

CFG = DEFAULT_CONFIG
AI_SEED = 11


def play_like_the_server(store: GameStore, seed: int, human_redraws: bool = True) -> Game:
    """照 server.py 的流程打一局：1 个真人（脚本出牌）+ 2 个 AI，每一步都存库。

    AI 的 rng 用固定 seed，复盘时传同一个 seed 就该逐位复现。
    真人每轮付得起就换一次牌，顺带覆盖"换牌扣钱"那条路；
    老张是富二代，覆盖"免费换牌"（库里存的是换完那手，不花钱、流水账里也没有）。
    """
    game = Game(game_id=f"g{seed}", cfg=CFG, rng=random.Random(seed))
    game.add_player("真人")
    game.add_player("老张", is_ai=True)
    game.add_player("老李", is_ai=True)
    game.players[2].origin = Origin.RICH
    game.players[3].origin = Origin.OFFICIAL  # 官二代 AI 会用到「透风」，复盘时也得一样
    game.start_game()
    store.save(game)
    pool = ai.AgentPool(cfg=CFG, rng=random.Random(AI_SEED))
    human_rng = random.Random(seed + 100)
    while not game.is_over:
        for pid in game.ai_player_ids():
            game.select_actions(pid, ai.turn(game, pid, pool))
            game.lock_action(pid)
            store.save(game)
        human = 1
        if human_redraws and game.private_state(human).get("redraw_affordable"):
            game.redraw(human)
        hand = game.hands[human]
        idx = human_rng.sample(range(len(hand)), CFG.picks_per_round)
        picks = []
        for i in idx:
            card = hand[i].card
            target = human_rng.choice([2, 3]) if card.needs_target else None
            picks.append({"index": i, "target": target})
        game.select_actions(human, picks)
        game.lock_action(human)
        store.save(game)
        game.reveal_event()
        game.resolve()
        store.save(game)
        if not game.is_over:
            game.advance_round()
            store.save(game)
    return game


class TestReplay(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "r.db"
        self.store = GameStore(self.path)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_every_ai_decision_is_reproduced(self):
        """同一个 AI seed 重演，每个决策都该一模一样，钱/政绩/官职零偏差。"""
        game = play_like_the_server(self.store, seed=5)
        res = replay.replay(
            self.store.history(game.game_id), CFG, ai_rng=random.Random(AI_SEED)
        )
        hit, total = res.match_rate
        self.assertEqual(total, 2 * game.round_number)
        self.assertEqual(hit, total, [
            (d.round, d.name, d.predicted, d.actual) for d in res.decisions if not d.matched
        ])
        self.assertEqual(res.drifts, [])
        self.assertEqual(res.final["winners"], [game.players[w].name for w in game.winners])

    def test_redraw_costs_are_replayed(self):
        """换过牌的局：库里存的是换完那手，钱要按流水账补扣，不然从第一轮就漂。"""
        game = play_like_the_server(self.store, seed=8, human_redraws=True)
        ledger = self.store.history(game.game_id)["ledger"][1]
        self.assertTrue(
            any(row["label"] == "重新抽牌" for e in ledger for row in e["rows"]),
            "这局真人一次都没换牌，测不到这条路",
        )
        res = replay.replay(self.store.history(game.game_id), CFG, ai_rng=random.Random(AI_SEED))
        self.assertEqual(res.drifts, [])

    def test_override_changes_the_outcome(self):
        """反事实：把某人靠牌升职那一轮的出牌换成什么都不打，那次升职就没了。"""
        game = play_like_the_server(self.store, seed=5)
        history = self.store.history(game.game_id)
        promoted = next(
            (r["round"], p["player_id"])
            for r in history["archive"]
            for p in r["players"]
            if p["promotion"] in ("MERIT", "MONEY", "BOTH")  # 工龄晋升不靠牌，换了也拦不住
        )
        res = replay.replay(history, CFG, overrides={promoted: []})
        self.assertTrue(res.overridden)
        self.assertEqual(res.final["round"], promoted[0])
        name = history["players"][promoted[1] - 1]["name"]
        # 靠牌的那次升职没了（同一轮工龄刚好到了的话，可能改成工龄晋升）
        self.assertNotIn(res.final["players"][name]["promotion"], ("MERIT", "MONEY", "BOTH"))

    def test_only_requested_rounds_are_recorded(self):
        game = play_like_the_server(self.store, seed=5)
        res = replay.replay(self.store.history(game.game_id), CFG, rounds={2, 3})
        self.assertEqual({d.round for d in res.decisions}, {2, 3})

    def test_parse_override(self):
        key, picks = replay.parse_override("10:2=REPORT@1,promote_any")
        self.assertEqual(key, (10, 2))
        self.assertEqual(picks, [
            {"card": "REPORT", "target_id": 1}, {"card": "PROMOTE_ANY", "target_id": None}
        ])
        with self.assertRaises(replay.ReplayError):
            replay.parse_override("10:2=NOT_A_CARD")

    def test_unknown_game(self):
        self.assertIsNone(self.store.history("nope"))
        self.assertIsNone(self.store.latest_game_id())

    def test_cli(self):
        """--section replay：空库报错退出 1；有对局就打印出来，--json 能解析。"""
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(analysis.main(["--section", "replay", "--db", str(self.path)]), 1)
        play_like_the_server(self.store, seed=5)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = analysis.main(["--section", "replay", "--db", str(self.path), "--rounds", "1-2"])
        self.assertEqual(code, 0)
        self.assertIn("复盘 g5", out.getvalue())
        self.assertIn("AI 决策复现", out.getvalue())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            analysis.main(["--section", "replay", "--db", str(self.path), "--json"])
        import json
        self.assertEqual(json.loads(out.getvalue())["game_id"], "g5")


if __name__ == "__main__":
    unittest.main()
