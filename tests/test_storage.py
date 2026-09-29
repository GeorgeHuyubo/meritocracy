"""storage.py 测试：服务器重启后能恢复进行中的对局。"""

from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_CONFIG  # noqa: E402
from game import Game  # noqa: E402
from models import Card, DealtCard, Phase  # noqa: E402
from storage import GameStore  # noqa: E402

CFG = DEFAULT_CONFIG


class TestStorage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "t.db"
        self.store = GameStore(self.path)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def _game(self) -> Game:
        game = Game(game_id="g1", cfg=CFG, rng=random.Random(3))
        for name in ("甲", "乙", "丙"):
            game.add_player(name)
        return game

    def test_lobby_round_trip(self):
        game = self._game()
        self.store.save(game)
        loaded = self.store.load("g1", cfg=CFG)
        self.assertIsNotNone(loaded)
        self.assertIs(loaded.phase, Phase.LOBBY)
        self.assertEqual([p.name for p in loaded.ordered_players()], ["甲", "乙", "丙"])
        self.assertEqual(loaded.host_id, game.host_id)

    def test_mid_round_round_trip(self):
        game = self._game()
        game.start_game()
        game.hands[1] = [DealtCard(Card.WORK, 9)] * CFG.hand_size
        game.select_actions(1, [{"action": "WORK", "target": None}] * CFG.picks_per_round)
        game.lock_action(1)
        game.players[2].money = 17
        game.players[2].merit = 9
        game.players[2].rank = 1
        game.players[2].tenure = 2
        self.store.save(game)

        loaded = self.store.load("g1", cfg=CFG)
        self.assertIs(loaded.phase, Phase.ACTION_SELECTION)
        self.assertEqual(loaded.round_number, 1)
        self.assertEqual(loaded.hands[1], [DealtCard(Card.WORK, 9)] * CFG.hand_size)
        self.assertEqual(loaded.selections[1].cards(), [Card.WORK] * CFG.picks_per_round)
        self.assertTrue(loaded.selections[1].locked)
        self.assertFalse(loaded.selections[2].locked)
        p2 = loaded.players[2]
        self.assertEqual((p2.money, p2.merit, p2.rank, p2.tenure), (17, 9, 1, 2))
        self.assertEqual(loaded.player_by_token(game.players[3].token).id, 3)

    def test_can_keep_playing_after_restore(self):
        game = self._game()
        game.start_game()
        self.store.save(game)

        loaded = self.store.load("g1", cfg=CFG, rng=random.Random(11))
        loaded.force_lock_all()
        loaded.reveal_event()
        outcome = loaded.resolve()
        self.assertEqual(outcome.round_number, 1)
        self.assertIs(loaded.phase, Phase.ROUND_RESULT)

        self.store.save(loaded)
        again = self.store.load("g1", cfg=CFG)
        self.assertIs(again.phase, Phase.ROUND_RESULT)
        self.assertEqual(again.current_event.id, outcome.event.id)
        self.assertEqual(again.public_state()["last_result"]["round"], 1)

    def test_resolution_phase_is_rewound(self):
        # 结算过程中崩溃：恢复后退回到 REVEAL_EVENT，可以重新结算
        game = self._game()
        game.start_game()
        game.force_lock_all()
        game.reveal_event()
        game.phase = Phase.RESOLUTION
        self.store.save(game)
        loaded = self.store.load("g1", cfg=CFG)
        self.assertIs(loaded.phase, Phase.REVEAL_EVENT)
        loaded.resolve()  # 不抛异常

    def test_load_latest_and_reset(self):
        game = self._game()
        self.store.save(game)
        self.assertIsNotNone(self.store.load_latest(cfg=CFG))
        self.store.reset("g1")
        self.assertIsNone(self.store.load("g1", cfg=CFG))

    def test_round_results_are_persisted(self):
        game = self._game()
        game.start_game()
        for _ in range(2):
            game.force_lock_all()
            game.reveal_event()
            game.resolve()
            self.store.save(game)
            if not game.is_over:
                game.advance_round()
                self.store.save(game)
        rows = self.store.conn.execute(
            "SELECT round_number, event_name FROM rounds WHERE game_id = 'g1' ORDER BY round_number"
        ).fetchall()
        self.assertEqual([r["round_number"] for r in rows], [1, 2])
        self.assertTrue(all(r["event_name"] for r in rows))

    def test_required_tables_exist(self):
        names = {
            r[0]
            for r in self.store.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        self.assertTrue({"games", "players", "hands", "actions", "rounds"}.issubset(names))


if __name__ == "__main__":
    unittest.main()
