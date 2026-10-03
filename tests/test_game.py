"""game.py 单元测试：状态机、终局判定、私密信息隔离。

覆盖需求清单第 26 节的 17、18、21 条。
"""

from __future__ import annotations

import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ai  # noqa: E402
import rules  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402
from game import Game, GameError  # noqa: E402
from helpers import ScriptedRng  # noqa: E402
from models import Card, DealtCard, Origin, Phase  # noqa: E402

CFG = DEFAULT_CONFIG
PICKS = CFG.picks_per_round


def hand_of(*cards, value=10):
    """造一手牌。生产牌给个固定点数，方便断言。"""
    return [
        DealtCard(card=c, value=value if c.is_production else 0) for c in cards
    ]


def pick(*cards_and_targets):
    """[(卡, 目标), ...] -> select_actions 要的格式。"""
    return [{"action": c.value, "target": t} for c, t in cards_and_targets]


def make_game(n=3, seed=7) -> Game:
    game = Game(game_id="t", cfg=CFG, rng=random.Random(seed))
    for i in range(n):
        game.add_player(f"玩家{i + 1}")
    return game


def run_round(game: Game, choices: dict[int, list] | None = None) -> None:
    """所有人出牌 -> 锁定 -> 揭示事件 -> 结算。没给选择的人视为不出牌。"""
    choices = choices or {}
    for pid in sorted(game.players):
        if pid in choices:
            game.select_actions(pid, choices[pid])
            game.lock_action(pid)
    game.force_lock_all()
    game.reveal_event()
    game.resolve()


class TestStateMachine(unittest.TestCase):
    def test_lobby_requires_min_players(self):
        game = Game(cfg=CFG, rng=random.Random(1))
        game.add_player("独狼")
        with self.assertRaises(GameError):
            game.start_game()

    def test_max_players(self):
        game = make_game(CFG.max_players)
        with self.assertRaises(GameError):
            game.add_player("多出来的人")

    def test_no_join_after_start(self):
        game = make_game(2)
        game.start_game()
        with self.assertRaises(GameError):
            game.add_player("迟到的人")

    def test_phase_sequence(self):
        game = make_game(2)
        self.assertIs(game.phase, Phase.LOBBY)
        game.start_game()
        self.assertIs(game.phase, Phase.ACTION_SELECTION)
        self.assertEqual(game.round_number, 1)
        for pid in game.players:
            self.assertEqual(len(game.hands[pid]), CFG.hand_size)

        game.force_lock_all()
        game.reveal_event()
        self.assertIs(game.phase, Phase.REVEAL_EVENT)
        game.resolve()
        self.assertIs(game.phase, Phase.ROUND_RESULT)
        game.advance_round()
        self.assertIs(game.phase, Phase.ACTION_SELECTION)
        self.assertEqual(game.round_number, 2)

    def test_event_is_hidden_until_everyone_locks(self):
        game = make_game(2)
        game.start_game()
        self.assertIsNone(game.current_event)
        self.assertIsNone(game.public_state()["current_event"])
        with self.assertRaises(GameError):
            game.reveal_event()  # 还没人锁定
        game.force_lock_all()
        game.reveal_event()
        self.assertIsNotNone(game.public_state()["current_event"])

    def test_cannot_play_card_outside_hand(self):
        game = make_game(2)
        game.start_game()
        game.hands[1] = hand_of(*([Card.WORK] * CFG.hand_size))
        with self.assertRaises(GameError):
            game.select_action(1, Card.ATTACK, 2)

    def test_cannot_target_self(self):
        game = make_game(2)
        game.start_game()
        game.hands[1] = hand_of(*([Card.ATTACK] * CFG.hand_size))
        with self.assertRaises(GameError):
            game.select_action(1, Card.ATTACK, 1)

    def test_target_required(self):
        game = make_game(2)
        game.start_game()
        game.hands[1] = hand_of(*([Card.REPORT] * CFG.hand_size))
        with self.assertRaises(GameError):
            game.select_action(1, Card.REPORT, None)

    def test_cannot_change_after_lock(self):
        game = make_game(2)
        game.start_game()
        game.hands[1] = hand_of(*([Card.WORK, Card.CORRUPT] * 3))
        game.select_actions(1, pick((Card.WORK, None), (Card.WORK, None)))
        game.lock_action(1)
        with self.assertRaises(GameError):
            game.select_actions(1, pick((Card.CORRUPT, None), (Card.CORRUPT, None)))

    def test_you_may_lock_with_fewer_cards_or_none(self):
        """少打甚至不打都行——主动弃权是合法选择。"""
        game = make_game(2)
        game.start_game()
        game.hands[1] = hand_of(*([Card.WORK] * CFG.hand_size))
        game.select_actions(1, pick((Card.WORK, None)))  # 只选一张
        game.lock_action(1)
        self.assertEqual(len(game.selections[1].picks), 1)

        game.select_actions(2, [])  # 一张不选
        game.lock_action(2)
        self.assertEqual(game.selections[2].picks, [])
        self.assertTrue(game.all_locked())

    def test_cannot_lock_more_than_the_limit(self):
        game = make_game(2)
        game.start_game()
        game.hands[1] = hand_of(*([Card.WORK] * CFG.hand_size))
        with self.assertRaises(GameError):
            game.select_actions(1, pick(*([(Card.WORK, None)] * (CFG.picks_per_round + 1))))

    def test_cannot_play_more_copies_than_you_hold(self):
        game = make_game(2)
        game.start_game()
        game.hands[1] = hand_of(Card.WORK, *([Card.REPORT] * (CFG.hand_size - 1)))
        with self.assertRaises(GameError):
            game.select_actions(1, pick((Card.WORK, None), (Card.WORK, None)))

    def test_can_play_two_different_cards(self):
        game = make_game(3)
        game.start_game()
        game.hands[1] = hand_of(Card.WORK, Card.ATTACK, *([Card.CORRUPT] * (CFG.hand_size - 2)))
        game.select_actions(1, pick((Card.WORK, None), (Card.ATTACK, 2)))
        game.lock_action(1)
        self.assertEqual(
            [a.card for a in game.selections[1].picks], [Card.WORK, Card.ATTACK]
        )


class TestPresidentEndsGameImmediately(unittest.TestCase):
    """第 17 条：达到国家主席立即结束游戏。"""

    def test_merit_promotion_to_president_ends_game(self):
        game = make_game(3)
        game.start_game()
        p = game.players[1]
        p.rank = 3
        # 省级 -> 主席是双条件台阶：政绩和金钱都得够
        p.merit = CFG.promotion_merit_costs[3]
        p.money = CFG.promotion_money_costs[3]
        game.hands[1] = hand_of(*([Card.PROMOTE_ANY] * CFG.hand_size))

        run_round(game, {1: pick((Card.PROMOTE_ANY, None), (Card.PROMOTE_ANY, None))})

        self.assertEqual(game.players[1].rank, CFG.president_rank)
        self.assertIs(game.phase, Phase.GAME_OVER)
        self.assertEqual(game.winners, [1])
        self.assertIn("国家主席", game.game_over_reason)
        self.assertTrue(game.is_over)

    def test_game_over_before_max_rounds(self):
        game = make_game(2)
        game.start_game()
        game.players[1].rank = 3
        game.players[1].money = CFG.promotion_money_costs[3]
        game.players[1].merit = CFG.promotion_merit_costs[3]
        game.hands[1] = hand_of(*([Card.PROMOTE_ANY] * CFG.hand_size))
        run_round(game, {1: pick((Card.PROMOTE_ANY, None), (Card.PROMOTE_ANY, None))})
        self.assertIs(game.phase, Phase.GAME_OVER)
        self.assertEqual(game.round_number, 1)
        with self.assertRaises(GameError):
            game.advance_round()

    def test_simultaneous_presidents_are_split_by_money(self):
        """同一轮多人登顶时按家底决胜（规则书没定义，PRESIDENT_TIEBREAK）。"""
        game = make_game(3)
        game.start_game()
        choices = {}
        for pid in (1, 2, 3):
            game.players[pid].rank = 3
            game.players[pid].merit = CFG.promotion_merit_costs[3]
            game.players[pid].money = CFG.promotion_money_costs[3]
            game.hands[pid] = hand_of(*([Card.PROMOTE_ANY] * CFG.hand_size))
            choices[pid] = pick((Card.PROMOTE_ANY, None), (Card.PROMOTE_ANY, None))
        game.players[2].money += 20  # 家底最厚
        game.players[1].money += 8
        run_round(game, choices)
        self.assertEqual(game.winners, [2])
        names = {pid: game.players[pid].name for pid in (1, 2, 3)}
        self.assertEqual(
            game.game_over_reason,
            f"国家主席最大候选人为：{names[1]}、{names[2]}、{names[3]}，"
            f"最终还是因为{names[2]} 家底更厚，成功当选国家主席。",
        )
        # 主席只有一个：落选的退回省级，钱和政绩不动
        top = CFG.president_rank
        self.assertEqual(game.players[2].rank, top)
        for pid in (1, 3):
            self.assertEqual(game.players[pid].rank, top - 1)
        self.assertEqual(game.players[1].money, game.players[3].money + 8)
        self.assertEqual(game.last_outcome.presidents, [2])
        self.assertEqual(game.last_outcome.outcomes[1].rank_after, top - 1)

    def test_simultaneous_presidents_fall_back_to_merit(self):
        game = make_game(2)
        game.start_game()
        choices = {}
        for pid in (1, 2):
            game.players[pid].rank = 3
            game.players[pid].merit = CFG.promotion_merit_costs[3]
            game.players[pid].money = CFG.promotion_money_costs[3]
            game.hands[pid] = hand_of(*([Card.PROMOTE_ANY] * CFG.hand_size))
            choices[pid] = pick((Card.PROMOTE_ANY, None), (Card.PROMOTE_ANY, None))
        game.players[1].merit = CFG.promotion_merit_costs[3] + 15  # 政绩更高，余额也更高
        run_round(game, choices)
        self.assertEqual(game.winners, [1])

    def test_still_co_winners_when_truly_identical(self):
        game = make_game(2)
        game.start_game()
        choices = {}
        for pid in (1, 2):
            game.players[pid].rank = 3
            game.players[pid].merit = CFG.promotion_merit_costs[3]
            game.players[pid].money = CFG.promotion_money_costs[3]
            game.hands[pid] = hand_of(*([Card.PROMOTE_ANY] * CFG.hand_size))
            choices[pid] = pick((Card.PROMOTE_ANY, None), (Card.PROMOTE_ANY, None))
        run_round(game, choices)
        self.assertEqual(sorted(game.winners), [1, 2])
        self.assertTrue(all(game.players[pid].rank == CFG.president_rank for pid in (1, 2)))

    def test_partial_tie_demotes_only_the_poorer(self):
        """三人登顶、两人家底一样厚：两人共同当选，第三个退回省级。"""
        game = make_game(3)
        game.start_game()
        choices = {}
        for pid in (1, 2, 3):
            game.players[pid].rank = 3
            game.players[pid].merit = CFG.promotion_merit_costs[3]
            game.players[pid].money = CFG.promotion_money_costs[3]
            game.hands[pid] = hand_of(*([Card.PROMOTE_ANY] * CFG.hand_size))
            choices[pid] = pick((Card.PROMOTE_ANY, None), (Card.PROMOTE_ANY, None))
        game.players[1].money += 5
        game.players[3].money += 5
        run_round(game, choices)
        self.assertEqual(sorted(game.winners), [1, 3])
        self.assertEqual(game.players[2].rank, CFG.president_rank - 1)
        self.assertIn("共同当选国家主席", game.game_over_reason)

    def test_tiebreak_can_be_switched_off(self):
        import dataclasses
        cfg = dataclasses.replace(CFG, president_tiebreak=False)
        game = Game(game_id="t", cfg=cfg, rng=random.Random(7))
        for i in range(2):
            game.add_player(f"玩家{i + 1}")
        game.start_game()
        choices = {}
        for pid in (1, 2):
            game.players[pid].rank = 3
            game.players[pid].merit = cfg.promotion_merit_costs[3]
            game.players[pid].money = cfg.promotion_money_costs[3]
            game.hands[pid] = hand_of(*([Card.PROMOTE_ANY] * cfg.hand_size))
            choices[pid] = pick((Card.PROMOTE_ANY, None), (Card.PROMOTE_ANY, None))
        run_round(game, choices)
        self.assertEqual(sorted(game.winners), [1, 2])
        # 真平局：两个都是主席，没人被撤
        self.assertTrue(all(game.players[pid].rank == cfg.president_rank for pid in (1, 2)))


class TestFinalSettlement(unittest.TestCase):
    """第 18 条：10 轮结束后的最终结算。"""

    def test_richest_wins_when_nobody_reached_the_top(self):
        game = make_game(3)
        game.start_game()
        game.round_number = CFG.max_rounds
        # 注意：现在每轮有按官职发的合法工资，所以要把工资算进去
        game.players[1].money = 9
        game.players[2].money = 5
        game.players[2].rank = 3  # 官再大也压不过钱……但他工资也更高
        game.players[3].money = 5
        run_round(game)
        self.assertIs(game.phase, Phase.GAME_OVER)
        self.assertIn("无人登顶", game.game_over_reason)
        richest = max(game.players.values(), key=lambda p: p.money)
        self.assertEqual(game.winners, [richest.id])

    def test_rank_breaks_the_money_tie(self):
        game = make_game(3)
        game.start_game()
        game.round_number = CFG.max_rounds
        for pid in (1, 2, 3):
            game.players[pid].money = 5
        game.players[2].rank = 2
        run_round(game)
        self.assertEqual(game.winners, [2])

    def test_merit_breaks_the_last_tie(self):
        game = make_game(2)
        game.start_game()
        game.round_number = CFG.max_rounds
        game.players[1].money = 5
        game.players[2].money = 5
        game.players[2].merit = 3
        run_round(game)
        self.assertEqual(game.winners, [2])

    def test_full_draw(self):
        game = make_game(2)
        game.start_game()
        game.round_number = CFG.max_rounds
        run_round(game)
        self.assertEqual(sorted(game.winners), [1, 2])
        self.assertIn("平局", game.game_over_reason)

    def test_game_never_exceeds_max_rounds(self):
        game = Game(cfg=CFG, rng=random.Random(99))
        for i in range(4):
            game.add_player(f"玩家{i + 1}")
        game.start_game()
        while not game.is_over:
            for pid in sorted(game.players):
                idxs = list(range(len(game.hands[pid])))
                picks = []
                for _ in range(CFG.picks_per_round):
                    i = game.rng.choice(idxs)
                    idxs.remove(i)
                    card = game.hands[pid][i].card
                    target = None
                    if card.needs_target:
                        target = game.rng.choice([o for o in game.players if o != pid])
                    picks.append({"index": i, "target": target})
                game.select_actions(pid, picks)
                game.lock_action(pid)
            game.reveal_event()
            game.resolve()
            if not game.is_over:
                game.advance_round()
        self.assertLessEqual(game.round_number, CFG.max_rounds)
        self.assertTrue(game.winners)


class TestPrivacy(unittest.TestCase):
    """第 21 条：不向其他客户端泄露 private state。"""

    PUBLIC_PLAYER_KEYS = {
        "id", "name", "rank", "rank_name", "merit", "tenure", "connected", "is_ai",
        # 严重警告是公开的：官场上谁挨过处分，大家都知道
        "warnings",
        # 出身也是公开的。主要是为了 AI：官二代的晋升门槛和别人不一样，
        # 不知道就会算错"他还差多远"，终局刹车会失灵。
        "origin",
    }

    def test_public_player_entries_have_no_secret_fields(self):
        game = make_game(3)
        game.start_game()
        for entry in game.public_state()["players"]:
            self.assertEqual(set(entry), self.PUBLIC_PLAYER_KEYS)

    def test_public_state_never_contains_money_hand_or_token(self):
        game = make_game(3)
        game.start_game()
        game.players[1].money = 123456
        game.hands[1] = hand_of(*([Card.ATTACK] * CFG.hand_size))
        game.select_actions(1, pick((Card.ATTACK, 2), (Card.ATTACK, 3)))
        game.lock_action(1)

        public = game.public_state()
        full = json.dumps(public, ensure_ascii=False)
        # promotion_costs 是静态规则说明（写着"晋升需要多少钱"），不是任何人的钱
        # promotion_costs / hand_size / picks_per_round 都是静态规则说明，不是谁的秘密
        static_keys = {"promotion_costs", "hand_size", "picks_per_round"}
        dynamic = json.dumps(
            {k: v for k, v in public.items() if k not in static_keys}, ensure_ascii=False
        )
        self.assertNotIn("123456", full)
        self.assertNotIn("money", dynamic)
        self.assertNotIn("hand", dynamic)
        self.assertNotIn("ATTACK", full)  # 行动与目标都不公开
        self.assertNotIn("picks", dynamic)
        for p in game.players.values():
            self.assertNotIn(p.token, full)

    def test_locked_list_shows_who_locked_but_not_what(self):
        game = make_game(2)
        game.start_game()
        game.hands[1] = hand_of(*([Card.CORRUPT] * CFG.hand_size))
        game.select_actions(1, pick((Card.CORRUPT, None), (Card.CORRUPT, None)))
        game.lock_action(1)
        public = game.public_state()
        self.assertEqual(public["locked_players"], [1])
        self.assertNotIn("CORRUPT", json.dumps(public, ensure_ascii=False))

    def test_private_state_is_only_about_the_owner(self):
        game = make_game(2)
        game.start_game()
        game.players[1].money = 77
        game.players[2].money = 88
        priv = game.private_state(1)
        self.assertEqual(priv["money"], 77)
        blob = json.dumps(priv, ensure_ascii=False)
        self.assertNotIn("88", blob)
        self.assertNotIn(game.players[2].token, blob)
        self.assertEqual(priv["hand"], [d.view() for d in game.hands[1]])

    def test_report_and_attack_identities_stay_secret(self):
        game = make_game(3)
        game.start_game()
        for pid in game.players:
            game.hands[pid] = hand_of(*(list(Card) * 2))
        game.select_actions(1, pick((Card.CORRUPT, None), (Card.CORRUPT, None)))
        game.select_actions(2, pick((Card.REPORT, 1), (Card.WORK, None)))
        game.select_actions(3, pick((Card.ATTACK, 1), (Card.WORK, None)))
        for pid in game.players:
            game.lock_action(pid)
        game.reveal_event()
        game.resolve()

        public = json.dumps(game.public_state(), ensure_ascii=False)
        outcome = game.last_outcome
        self.assertTrue(outcome.outcomes[1].reported or outcome.outcomes[1].attacked)
        # 公开文本里只会出现"被举报/被攻击"的那个人，不会点名举报者和攻击者
        for msg in outcome.public_messages:
            self.assertNotIn("玩家2 举报", msg)
            self.assertNotIn("玩家3 攻击", msg)
        # 受害者自己的私密结算里也不含举报者身份
        victim = game.private_state(1)["private_result"]
        self.assertNotIn("玩家2", json.dumps(victim, ensure_ascii=False))
        self.assertIn("locked_players", public)

    def test_public_player_facts_contain_only_public_things(self):
        """结算结果里的结构化公开事实不能混进任何私密字段。"""
        allowed = {
            "player_id", "attacked", "attack_merit_loss", "tenure_reset_by_attack",
            "merit_promotion_blocked", "promotion", "demotion",
            "rank_before", "rank_after", "merit_after", "tenure_after",
            # 这两个是"查办从哪来的"：公报里本来就写着（"被匿名举报" vs
            # "在反腐风暴中被查办"），是布尔值和事件名，不指向任何具体玩家。
            "reported_by_player", "report_from_event",
            # 严重警告也是公开的，公报里就写着记了几次、还差几次降级
            "warnings_issued", "warnings_after",
            # 政治攻击是**明攻击**：公报里点名写着谁抢了谁的功劳，
            # 所以攻击者身份是公开信息。举报者身份永远不在此列。
            "attacked_by",
        }
        forbidden_keys = (
            "money", "corrupt", "confiscat", "card", "target", "hand", "note", "reward",
        )
        game = make_game(3)
        game.start_game()
        for pid in game.players:
            game.hands[pid] = hand_of(*(list(Card) * 2))
        game.select_actions(1, pick((Card.CORRUPT, None), (Card.CORRUPT, None)))
        game.select_actions(2, pick((Card.REPORT, 1), (Card.WORK, None)))
        game.select_actions(3, pick((Card.ATTACK, 1), (Card.WORK, None)))
        for pid in game.players:
            game.lock_action(pid)
        game.reveal_event()
        outcome = game.resolve()

        facts = game.public_state()["last_result"]["player_facts"]
        self.assertEqual(len(facts), 3)
        for entry in facts:
            self.assertEqual(set(entry), allowed)
            for key in entry:
                for word in forbidden_keys:
                    self.assertNotIn(word, key.lower())

        # 任何一个金额都不能以数值形式出现在公开事实里
        secrets_now = set()
        for pid, o in outcome.outcomes.items():
            secrets_now.update(
                v
                for v in (
                    game.players[pid].money,
                    o.corrupt_amount,
                    o.money_confiscated,
                    o.money_from_reports,
                )
                if v > 2  # 0/1/2 这种小数字和 rank/tenure 撞车，没有信息量
            )
        numbers = {
            v for entry in facts for k, v in entry.items()
            if isinstance(v, int) and k not in ("player_id", "merit_after")
        }
        self.assertFalse(secrets_now & numbers, f"公开事实里出现了金额: {secrets_now & numbers}")

    def test_wealth_top_ids_match_the_broadcast(self):
        """wealth_top_ids 只是广播里已经点过的名，不能多点一个人。"""
        game = make_game(3)
        game.start_game()
        for pid in game.players:
            game.hands[pid] = hand_of(*([Card.CORRUPT] * CFG.hand_size))
        for pid in (1, 2):
            game.select_action(pid, Card.CORRUPT)
        game.select_action(3, Card.WORK) if Card.WORK in game.hands[3] else None
        game.force_lock_all()
        game.reveal_event()
        outcome = game.resolve()
        public = game.public_state()["last_result"]
        named = [
            p["name"]
            for p in game.public_state()["players"]
            if any(p["name"] in msg for msg in public["wealth_broadcast"])
        ]
        self.assertEqual(
            sorted(named),
            sorted(game.players[pid].name for pid in public["wealth_top_ids"]),
        )
        # 点名的必须真的是本轮到手最多的（工资 + 净落袋的脏钱）
        income = {
            pid: game.salary_paid.get(pid, 0) + o.net_corrupt_gain
            for pid, o in outcome.outcomes.items()
        }
        best = max(income.values())
        self.assertEqual(
            sorted(public["wealth_top_ids"]),
            sorted(pid for pid, v in income.items() if v == best),
        )

    def test_wealth_broadcast_hides_amounts(self):
        game = make_game(2)
        game.start_game()
        for pid in game.players:
            game.hands[pid] = hand_of(*([Card.CORRUPT] * CFG.hand_size))
            game.select_actions(pid, pick((Card.CORRUPT, None), (Card.CORRUPT, None)))
            game.lock_action(pid)
        game.reveal_event()
        outcome = game.resolve()
        amounts = {o.corrupt_amount for o in outcome.outcomes.values() if o.corrupt_amount}
        for msg in outcome.wealth_broadcast:
            for amount in amounts:
                self.assertNotIn(str(amount), msg)


class TestLiveLedger(unittest.TestCase):
    """本轮还没结算，但钱已经动了的那几笔，也要立刻出现在流水账上。

    工资在回合开头到账、换牌当场扣钱，而流水账是结算时才生成的——
    中间这段时间玩家会看到余额变了、账上却找不到这一笔。
    """

    def _entries(self, game, pid=1):
        return game.private_state(pid)["ledger"]

    def test_salary_shows_up_before_the_round_resolves(self):
        game = make_game(2)
        game.start_game()
        entries = self._entries(game)
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0]["pending"])
        self.assertEqual(entries[0]["round"], 1)
        labels = [r["label"] for r in entries[0]["rows"]]
        self.assertEqual(labels, ["合法工资"])
        self.assertEqual(entries[0]["money_after"], game.players[1].money)

    def test_redraw_shows_up_immediately_too(self):
        game = make_game(2)
        game.start_game()
        game.players[1].money += 20
        game.redraw(1)
        rows = self._entries(game)[0]["rows"]
        self.assertEqual([r["label"] for r in rows], ["合法工资", "重新抽牌"])
        self.assertEqual(rows[1]["money"], -CFG.redraw_cost(0))

    def test_the_pending_entry_is_not_duplicated_after_resolving(self):
        """结算后这一轮只能出现一次，不能既有归档又有"进行中"。"""
        game = make_game(2)
        game.start_game()
        run_round(game, {})
        rounds = [e["round"] for e in self._entries(game)]
        self.assertEqual(rounds, [1])
        self.assertFalse(self._entries(game)[0].get("pending"))

    def test_next_round_adds_a_fresh_pending_entry(self):
        game = make_game(2)
        game.start_game()
        run_round(game, {})
        game.advance_round()
        entries = self._entries(game)
        self.assertEqual([e["round"] for e in entries], [1, 2])
        self.assertFalse(entries[0].get("pending"))
        self.assertTrue(entries[1]["pending"])

    def test_no_pending_entry_when_nothing_happened_yet(self):
        """主席工资是 0，开局又没换牌——那就不该凭空多一行。"""
        game = make_game(2)
        game.start_game()
        game.players[1].rank = CFG.president_rank
        game.salary_paid.pop(1, None)
        self.assertEqual(self._entries(game), [])


class TestPostmortem(unittest.TestCase):
    """终局复盘：只发给本人，含金钱数字，绝不进公开状态。"""

    def _finished_game(self):
        game = make_game(3)
        game.start_game()
        game.round_number = CFG.max_rounds
        game.players[1].money = 9
        run_round(game)
        self.assertTrue(game.is_over)
        return game

    def test_only_available_after_the_game_ends(self):
        game = make_game(3)
        game.start_game()
        self.assertIsNone(game.private_state(1)["postmortem"])

    def test_standing_agrees_with_who_actually_won(self):
        """复盘名次必须和结算口径一致，不能出现"你赢了但你垫底"。"""
        game = self._finished_game()
        self.assertTrue(game.winners)
        first = game.final_standing()[0]
        self.assertIn(first, game.winners)
        self.assertFalse(game.private_state(first)["postmortem"]["is_last"])
        self.assertEqual(game.private_state(first)["postmortem"]["place"], 1)

    def test_gives_each_player_their_own_placing(self):
        game = self._finished_game()
        places = {pid: game.private_state(pid)["postmortem"]["place"] for pid in game.players}
        self.assertEqual(sorted(places.values()), [1, 2, 3])
        last = game.final_standing()[-1]
        self.assertTrue(game.private_state(last)["postmortem"]["is_last"])
        self.assertFalse(game.private_state(game.final_standing()[0])["postmortem"]["is_last"])

    def test_never_leaks_into_public_state(self):
        """复盘是每人一份的私密数据，不能进公开广播。

        注意终局的 reveal 里确实有金钱——那是游戏结束后才摊开的全揭示，
        由 TestFullReveal 守着"结束前一个字都不漏"。
        """
        game = self._finished_game()
        public = game.public_state()
        # promotion_costs 是静态规则说明（写着晋升要多少钱），不是谁的钱
        skip = {"reveal", "promotion_costs"}
        blob = json.dumps(
            {k: v for k, v in public.items() if k not in skip}, ensure_ascii=False
        )
        self.assertNotIn("postmortem", blob)
        self.assertNotIn("money", blob)
        self.assertNotIn("ledger", blob)


class TestFullReveal(unittest.TestCase):
    """终局全揭示：游戏结束前一个字都不能漏，结束后才摊开。"""

    def test_hidden_while_the_game_is_running(self):
        game = make_game(3)
        game.start_game()
        game.players[1].money = 4242
        self.assertIsNone(game.full_reveal())
        public = game.public_state()
        self.assertIsNone(public["reveal"])
        self.assertNotIn("4242", json.dumps(public, ensure_ascii=False))

    def test_hidden_even_at_round_result(self):
        game = make_game(3)
        game.start_game()
        game.players[1].money = 4242
        run_round(game)
        self.assertIs(game.phase, Phase.ROUND_RESULT)
        self.assertIsNone(game.public_state()["reveal"])
        self.assertNotIn("4242", json.dumps(game.public_state(), ensure_ascii=False))

    def test_everything_comes_out_when_it_ends(self):
        game = make_game(3)
        game.start_game()
        for pid in game.players:
            game.hands[pid] = hand_of(*(list(Card) * 2))
        run_round(game, {
            1: pick((Card.CORRUPT, None), (Card.WORK, None)),
            2: pick((Card.REPORT, 1), (Card.WORK, None)),
        })
        game.advance_round()
        game.round_number = CFG.max_rounds
        run_round(game)
        self.assertTrue(game.is_over)

        rev = game.public_state()["reveal"]
        self.assertIsNotNone(rev)
        # 名次齐全，而且第一名就是赢家
        self.assertEqual([r["place"] for r in rev["standing"]], [1, 2, 3])
        self.assertTrue(rev["standing"][0]["won"])
        # 每个人的钱都摊开了
        for row in rev["standing"]:
            self.assertIn("money", row)
        # 每一轮每个人出了什么牌、打了谁，都能回看
        self.assertEqual(len(rev["rounds"]), 2)
        first = {p["player_id"]: p for p in rev["rounds"][0]["players"]}
        self.assertEqual(first[1]["cards"], ["CORRUPT", "WORK"])
        self.assertEqual(first[2]["cards"], ["REPORT", "WORK"])
        self.assertEqual(first[2]["targets"][0], 1)  # 举报的是谁也看得见


class TestRedraw(unittest.TestCase):
    """换牌要花钱，价钱按当前官职算，官越大越贵。"""

    def test_it_costs_money(self):
        game = make_game(2)
        game.start_game()
        cost = CFG.redraw_cost(0)
        game.players[1].money = cost + 3
        self.assertEqual(game.redraw(1), cost)
        self.assertEqual(game.players[1].money, 3)
        self.assertEqual(len(game.hands[1]), CFG.hand_size)

    def test_official_hears_the_event_before_choosing(self):
        """官二代「透风」：选牌阶段就看得到本轮事件；别人连这个键都没有；揭晓的就是透露的那个。"""
        game = make_game(3)
        game.players[1].origin = Origin.OFFICIAL
        game.start_game()
        tip = game.private_state(1).get("tipoff_event")
        self.assertIsNotNone(tip)
        self.assertNotIn("tipoff_event", game.private_state(2))
        self.assertNotIn("tipoff_event", game.private_state(3))
        self.assertIsNone(game.public_state()["current_event"])
        self.assertNotIn(tip["name"], json.dumps(game.public_state(), ensure_ascii=False))
        for pid in game.players:
            game.select_actions(pid, [])
            game.lock_action(pid)
        revealed = game.reveal_event()
        self.assertEqual(revealed.id, tip["id"])
        self.assertNotIn("tipoff_event", game.private_state(1))  # 揭晓之后就不需要透风了

    def test_the_tipoff_survives_a_restart(self):
        """服务器重启后官二代听到的风声不能变。"""
        game = make_game(2)
        game.players[1].origin = Origin.OFFICIAL
        game.start_game()
        tip = game.private_state(1)["tipoff_event"]["id"]
        restored = Game.from_snapshot(game.to_snapshot(), cfg=CFG)
        self.assertEqual(restored.private_state(1)["tipoff_event"]["id"], tip)

    def test_grinder_card_preview_flags_overtime(self):
        """卷王手牌上要写出「加班」，牌面数字和普通人一样。"""
        game = make_game(2)
        game.players[1].origin = Origin.GRINDER
        game.start_game()
        mine, plain = (game.private_state(p)["card_preview"]["WORK"] for p in (1, 2))
        self.assertEqual(mine["overtime"], CFG.origin_grinder_overtime_multiplier)
        self.assertFalse(plain["overtime"])
        self.assertEqual((mine["merit_lo"], mine["merit_hi"]), (plain["merit_lo"], plain["merit_hi"]))

    def test_grinder_always_holds_two_works_even_after_a_redraw(self):
        game = make_game(2)
        game.players[1].origin = Origin.GRINDER
        game.start_game()
        works = lambda: sum(d.card is Card.WORK for d in game.hands[1])
        self.assertGreaterEqual(works(), 2)
        game.players[1].money = 50
        game.redraw(1)
        self.assertGreaterEqual(works(), 2)

    def test_red_has_a_one_shot_family_card(self):
        """一纸调令：只有红二代有；不在手牌里也能选；打出去就算用掉（没升成也一样）。"""
        game = make_game(2)
        game.players[1].origin = Origin.RED
        game.start_game()
        self.assertTrue(game.private_state(1)["family_card"]["usable"])
        self.assertIsNone(game.private_state(2)["family_card"])
        with self.assertRaises(GameError):
            game.select_actions(2, [{"action": "PROMOTE_FAMILY"}])
        game.players[1].merit = game.players[1].money = 0  # 资源不够：会白用
        game.select_actions(1, [{"action": "PROMOTE_FAMILY"}])
        for pid in game.players:
            game.lock_action(pid)
        game.reveal_event()
        game.resolve()
        self.assertIn(1, game.family_used)
        self.assertIsNone(game.private_state(1)["family_card"])  # 用掉就没了
        if not game.is_over:
            game.advance_round()
            with self.assertRaises(GameError):
                game.select_actions(1, [{"action": "PROMOTE_FAMILY"}])

    def test_family_card_is_not_offered_on_the_last_step(self):
        game = make_game(2)
        game.players[1].origin = Origin.RED
        game.start_game()
        game.players[1].rank = CFG.president_rank - 1
        info = game.private_state(1)["family_card"]
        self.assertFalse(info["usable"])
        self.assertIn("主席", info["why"])
        with self.assertRaises(GameError):
            game.select_actions(1, [{"action": "PROMOTE_FAMILY"}])

    def test_family_card_use_survives_a_restart(self):
        game = make_game(2)
        game.players[1].origin = Origin.RED
        game.start_game()
        game.family_used.add(1)
        restored = Game.from_snapshot(game.to_snapshot(), cfg=CFG)
        self.assertIn(1, restored.family_used)

    def test_rich_redraws_once_free_then_pays_the_base_price(self):
        """富二代每轮第一次免费，第二次才像别人一样从底价开始；下一轮重新免费。"""
        game = make_game(2)
        game.players[1].origin = Origin.RICH
        game.start_game()
        game.players[1].money = 0
        priv = game.private_state(1)
        self.assertTrue(priv["redraw_available"])
        self.assertEqual(priv["redraw_cost"], 0)
        self.assertEqual(priv["redraw_next_cost"], CFG.redraw_cost(0))
        self.assertTrue(priv["redraw_affordable"], "一分钱没有也能免费换")
        self.assertEqual(game.redraw(1), 0)
        self.assertEqual(game.players[1].money, 0)
        self.assertFalse(game.private_state(1)["redraw_affordable"])  # 第二次要钱了
        game.players[1].money = 10
        self.assertEqual(game.redraw(1), CFG.redraw_cost(0))
        # 免费那次不进流水账（0 元），也不影响普通人
        self.assertEqual(game.private_state(2)["redraw_cost"], CFG.redraw_cost(0))

    def test_free_redraw_resets_every_round(self):
        game = make_game(2)
        game.players[1].origin = Origin.RICH
        game.start_game()
        game.redraw(1)
        for pid in game.players:
            game.select_actions(pid, [])
            game.lock_action(pid)
        game.reveal_event()
        game.resolve()
        if not game.is_over:
            game.advance_round()
            self.assertEqual(game.private_state(1)["redraw_cost"], 0)

    def test_cannot_afford_means_cannot_redraw(self):
        game = make_game(2)
        game.start_game()
        game.players[1].money = CFG.redraw_cost(0) - 1
        with self.assertRaises(GameError):
            game.redraw(1)

    def test_each_redraw_in_the_same_round_costs_double(self):
        """1 -> 2 -> 4 -> 8：不这样的话有钱人能在一轮里反复重抽到满意为止。"""
        game = make_game(2)
        game.start_game()
        base = CFG.redraw_cost(0)
        game.players[1].money = base * 7  # 正好够 1 + 2 + 4
        paid = [game.redraw(1) for _ in range(3)]
        self.assertEqual(paid, [base, base * 2, base * 4])
        self.assertEqual(game.players[1].money, 0)
        with self.assertRaises(GameError):
            game.redraw(1)

    def test_the_price_resets_next_round(self):
        game = make_game(2)
        game.start_game()
        game.players[1].money = 999
        game.redraw(1)
        game.redraw(1)
        run_round(game, {})
        game.advance_round()
        self.assertEqual(game.redraw_count, {})
        self.assertEqual(game.redraw(1), CFG.redraw_cost(0))

    def test_early_ranks_are_easier_to_redraw_than_late_ones(self):
        """前期好换、后期难换——对工资和对升职门槛两个口径都要成立。"""
        vs_salary = [
            CFG.redraw_cost(r) / CFG.salary(r) for r in range(CFG.president_rank)
        ]
        vs_threshold = [
            CFG.redraw_cost(r) / CFG.money_cost(r) for r in range(CFG.president_rank)
        ]
        self.assertEqual(vs_salary, sorted(vs_salary))
        self.assertEqual(vs_threshold, sorted(vs_threshold))
        self.assertLess(vs_threshold[0], vs_threshold[-1])

    def test_base_rank_can_afford_one_redraw_on_salary_alone(self):
        """手气差是最没意思的输法，基层至少要买得起一次重抽。"""
        self.assertLessEqual(CFG.redraw_cost(0), CFG.salary(0))

    def test_higher_rank_costs_more(self):
        costs = [CFG.redraw_cost(r) for r in range(CFG.president_rank)]
        self.assertEqual(costs, sorted(costs))
        self.assertLess(costs[0], costs[-1])

    def test_the_spend_shows_up_in_the_ledger(self):
        """不记账的话，玩家只会看到钱莫名其妙变少了。"""
        game = make_game(2)
        game.start_game()
        cost = CFG.redraw_cost(0)
        game.players[1].money = cost + 50
        game.redraw(1)
        game.hands[1] = hand_of(*([Card.WORK] * CFG.hand_size))
        run_round(game, {1: pick((Card.WORK, None))})
        rows = game.ledger[1][-1]["rows"]
        spent = [r for r in rows if r["label"] == "重新抽牌"]
        self.assertEqual(len(spent), 1)
        self.assertEqual(spent[0]["money"], -cost)

    def test_redraw_clears_your_picks(self):
        game = make_game(2)
        game.start_game()
        game.players[1].money = 99
        game.hands[1] = hand_of(*([Card.WORK] * CFG.hand_size))
        game.select_actions(1, pick((Card.WORK, None), (Card.WORK, None)))
        game.redraw(1)
        self.assertEqual(game.selections[1].picks, [])

    def test_cannot_redraw_after_locking(self):
        game = make_game(2)
        game.start_game()
        game.players[1].money = 99
        game.select_actions(1, [])
        game.lock_action(1)
        with self.assertRaises(GameError):
            game.redraw(1)

    def test_salary_lands_before_you_pick_and_can_pay_for_a_redraw(self):
        """工资在发牌的同时到账，所以这笔钱当轮就能拿去换牌。"""
        game = make_game(2)
        self.assertEqual(game.players[1].money, 0)
        game.start_game()
        self.assertEqual(game.players[1].money, CFG.salary(0))
        self.assertTrue(game.private_state(1)["redraw_affordable"])
        game.redraw(1)  # 纯靠工资换掉一手牌
        self.assertEqual(game.players[1].money, 0)

    def test_the_tab_resets_each_round(self):
        game = make_game(2)
        game.start_game()
        game.players[1].money = 99
        game.redraw(1)
        self.assertGreater(game.redraw_spent.get(1, 0), 0)
        run_round(game, {})
        game.advance_round()
        self.assertEqual(game.redraw_spent, {})


class TestLedger(unittest.TestCase):
    """流水账：每一笔钱和政绩是怎么来的。"""

    def test_lines_add_up_to_the_balance(self):
        game = make_game(2)
        game.start_game()
        game.hands[1] = hand_of(*(list(Card) * 2))
        run_round(game, {1: pick((Card.WORK, None), (Card.CORRUPT, None))})
        ledger = game.private_state(1)["ledger"]
        self.assertEqual(len(ledger), 1)
        entry = ledger[0]
        self.assertEqual(entry["round"], 1)
        self.assertEqual(sum(r["money"] for r in entry["rows"]), entry["money_after"])
        self.assertEqual(sum(r["merit"] for r in entry["rows"]), entry["merit_after"])
        labels = [r["label"] for r in entry["rows"]]
        self.assertIn("合法工资", labels)

    def test_ledger_is_private(self):
        game = make_game(2)
        game.start_game()
        run_round(game)
        self.assertNotIn("ledger", json.dumps(game.public_state(), ensure_ascii=False))

    def test_counts_what_happened_to_you(self):
        game = make_game(3)
        game.start_game()
        for pid in game.players:
            game.hands[pid] = hand_of(*(list(Card) * 2))
        # 玩家1 贪污两笔，玩家2、3 举报他
        run_round(game, {
            1: pick((Card.CORRUPT, None), (Card.CORRUPT, None)),
            2: pick((Card.REPORT, 1), (Card.WORK, None)),
            3: pick((Card.REPORT, 1), (Card.WORK, None)),
        })
        game.advance_round()          # 上一轮结算完要先进入下一轮
        game.round_number = CFG.max_rounds
        run_round(game)               # 打满轮数，游戏结束
        pm = game.private_state(1)["postmortem"]
        self.assertGreater(pm["stats"]["money_confiscated"], 0)
        # 两个人同时举报只算一次查实 -> 一次严重警告，还不到降级线
        self.assertEqual(pm["stats"]["demoted"], 0)
        self.assertEqual(game.players[1].warnings, 1)
        labels = [c["label"] for c in pm["top_causes"]]
        self.assertIn("赃款被举报没收", labels)


class TestSnapshotRoundTrip(unittest.TestCase):
    def test_snapshot_restores_state(self):
        game = make_game(3)
        game.start_game()
        game.hands[1] = hand_of(*([Card.WORK] * CFG.hand_size))
        game.select_actions(1, pick((Card.WORK, None), (Card.WORK, None)))
        game.lock_action(1)

        restored = Game.from_snapshot(game.to_snapshot(), cfg=CFG)
        self.assertEqual(restored.phase, game.phase)
        self.assertEqual(restored.round_number, game.round_number)
        self.assertEqual(sorted(restored.players), sorted(game.players))
        self.assertEqual(restored.hands[1], game.hands[1])
        self.assertEqual(restored.selections[1].cards(), [Card.WORK, Card.WORK])
        self.assertTrue(restored.selections[1].locked)
        self.assertIsNotNone(restored.player_by_token(game.players[1].token))

    def test_token_lookup_rejects_bad_token(self):
        game = make_game(2)
        self.assertIsNone(game.player_by_token("not-a-real-token"))
        self.assertIsNone(game.player_by_token(""))


class TestDealing(unittest.TestCase):
    def test_a_full_hand_every_round(self):
        game = make_game(4)
        game.start_game()
        for _ in range(3):
            for pid in game.players:
                self.assertEqual(len(game.hands[pid]), CFG.hand_size)
                for dealt in game.hands[pid]:
                    self.assertIsInstance(dealt.card, Card)
                    # 点数在发牌时就摇好了：生产牌必须有值，其余恒为 0
                    if dealt.card.is_production:
                        self.assertGreater(dealt.value, 0)
                    else:
                        self.assertEqual(dealt.value, 0)
            run_round(game)
            if game.is_over:
                break
            game.advance_round()

    def test_deal_uses_the_configured_distribution(self):
        hand = rules.deal_hand(ScriptedRng([0, 1, 2, 3, 0]), CFG)
        self.assertEqual(len(hand), CFG.hand_size)

    def test_the_dealt_value_is_what_gets_scored(self):
        """先摇后选：结算时用的就是发牌时那个数字，不会再摇一次。"""
        game = make_game(2)
        game.start_game()
        game.hands[1] = [DealtCard(Card.WORK, 12)] * CFG.hand_size
        game.select_actions(1, [{"index": 0, "target": None}])
        self.assertEqual(game.selections[1].picks[0].value, 12)


if __name__ == "__main__":
    unittest.main()


class TestRestart(unittest.TestCase):
    """房主随时可以结束本局重开，原班人马留在房间里。"""

    def _mid_game(self):
        game = Game(game_id="r", cfg=CFG, rng=random.Random(5))
        for i in range(3):
            game.add_player(f"P{i + 1}")
        game.add_player("老张", is_ai=True)
        game.start_game()
        pool = ai.AgentPool(cfg=CFG, rng=random.Random(5))
        for _ in range(2):
            for pid in sorted(game.players):
                game.select_actions(pid, ai.choose(game, pid, pool))
                game.lock_action(pid)
            game.reveal_event()
            game.resolve()
            if not game.is_over:
                game.advance_round()
        return game

    def test_host_can_abort_mid_game(self):
        game = self._mid_game()
        self.assertNotEqual(game.phase, Phase.LOBBY)
        game.restart(requester_id=game.host_id)
        self.assertIs(game.phase, Phase.LOBBY)
        self.assertEqual(game.round_number, 0)

    def test_everyone_stays_in_the_room_with_the_same_identity(self):
        game = self._mid_game()
        before = {
            p.id: (p.name, p.token, p.is_ai) for p in game.players.values()
        }
        game.restart(requester_id=game.host_id)
        after = {p.id: (p.name, p.token, p.is_ai) for p in game.players.values()}
        # token 一变，客户端存的身份就作废，所有人都得重新输名字
        self.assertEqual(before, after)

    def test_warnings_do_not_carry_into_the_next_game(self):
        """降职警告是累计的，重开必须清零——不然上一局的处分跟着你进新局。"""
        game = self._mid_game()
        for p in game.players.values():
            p.warnings = 1
        game.restart(requester_id=game.host_id)
        for p in game.players.values():
            self.assertEqual(p.warnings, 0, p.name)

    def test_scores_and_history_are_wiped(self):
        game = self._mid_game()
        game.restart(requester_id=game.host_id)
        for p in game.players.values():
            self.assertEqual((p.money, p.merit, p.rank, p.tenure), (0, 0, 0, 0))
            self.assertEqual(p.warnings, 0)
        self.assertEqual(game.history, [])
        self.assertEqual(game.archive, [])
        self.assertEqual(game.ledger, {})
        self.assertEqual(game.stats, {})
        self.assertEqual(game.winners, [])
        self.assertIsNone(game.last_outcome)
        self.assertIsNone(game.full_reveal())

    def test_only_the_host_may_abort(self):
        game = self._mid_game()
        other = max(game.players)
        with self.assertRaises(GameError):
            game.restart(requester_id=other)

    def test_can_play_again_right_after(self):
        game = self._mid_game()
        game.restart(requester_id=game.host_id)
        game.start_game(requester_id=game.host_id)
        self.assertIs(game.phase, Phase.ACTION_SELECTION)
        self.assertEqual(game.round_number, 1)
        self.assertEqual(len(game.hands), len(game.players))

    def test_restart_after_game_over_also_works(self):
        game = Game(game_id="r2", cfg=CFG, rng=random.Random(9))
        for i in range(2):
            game.add_player(f"P{i + 1}")
        game.start_game()
        pool = ai.AgentPool(cfg=CFG, rng=random.Random(9))
        while not game.is_over:
            for pid in sorted(game.players):
                game.select_actions(pid, ai.choose(game, pid, pool))
                game.lock_action(pid)
            game.reveal_event()
            game.resolve()
            if not game.is_over:
                game.advance_round()
        game.restart(requester_id=game.host_id)
        self.assertIs(game.phase, Phase.LOBBY)
        self.assertFalse(game.is_over)


class TestReportSourceIsDistinguished(unittest.TestCase):
    """真人举报和反腐风暴后果一样，但公报必须说清楚是哪一种。

    说不清楚的话，被风暴扫到的人会以为桌上有人在针对自己，
    然后去报复一个根本不存在的敌人。
    """

    STORM = next(
        e["id"] for e in CFG.event_definitions if e["effects"].get("storm_report")
    )

    def _run(self, event_id, reporter=False):
        game = make_game(2)
        game.start_game()
        for pid in game.players:
            game.hands[pid] = hand_of(*(list(Card) * 2))
        game.select_actions(1, pick((Card.CORRUPT, None), (Card.CORRUPT, None)))
        game.select_actions(2, pick((Card.REPORT, 1)) if reporter else [])
        for pid in game.players:
            game.lock_action(pid)
        game.reveal_event()
        game.current_event = rules.event_by_id(event_id, CFG)
        outcome = game.resolve()
        said = " ".join(outcome.public_messages)
        return outcome, said

    def test_a_real_report_says_so(self):
        outcome, said = self._run("CALM", reporter=True)
        self.assertIn("被匿名举报", said)
        self.assertNotIn("反腐风暴", said)
        facts = {f["player_id"]: f for f in outcome.public_view()["player_facts"]}
        self.assertTrue(facts[1]["reported_by_player"])
        self.assertEqual(facts[1]["report_from_event"], "")

    def test_the_storm_is_not_dressed_up_as_a_report(self):
        outcome, said = self._run(self.STORM, reporter=False)
        self.assertIn("反腐风暴", said)
        self.assertNotIn("被匿名举报", said)
        facts = {f["player_id"]: f for f in outcome.public_view()["player_facts"]}
        self.assertFalse(facts[1]["reported_by_player"])
        self.assertEqual(facts[1]["report_from_event"], "反腐风暴")

    def test_both_at_once_mentions_both(self):
        outcome, said = self._run(self.STORM, reporter=True)
        self.assertIn("被匿名举报", said)
        self.assertIn("反腐风暴", said)
        facts = {f["player_id"]: f for f in outcome.public_view()["player_facts"]}
        self.assertTrue(facts[1]["reported_by_player"])
        self.assertEqual(facts[1]["report_from_event"], "反腐风暴")

    def test_the_victim_learns_which_it_was(self):
        _, _ = self._run(self.STORM, reporter=False)
        game = make_game(2)
        game.start_game()
        for pid in game.players:
            game.hands[pid] = hand_of(*(list(Card) * 2))
        game.select_actions(1, pick((Card.CORRUPT, None), (Card.CORRUPT, None)))
        for pid in game.players:
            game.lock_action(pid)
        game.reveal_event()
        game.current_event = rules.event_by_id(self.STORM, CFG)
        outcome = game.resolve()
        notes = " ".join(outcome.outcomes[1].private_notes)
        self.assertIn("没人举报你", notes)  # 别让他去报复不存在的敌人

    def test_the_source_never_names_the_reporter(self):
        outcome, said = self._run("CALM", reporter=True)
        name = "玩家2"
        self.assertNotIn(name, said)
        facts = outcome.public_view()["player_facts"]
        self.assertNotIn(name, json.dumps(facts, ensure_ascii=False))
