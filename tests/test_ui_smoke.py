"""把每个界面都用真实 payload 渲染一遍，确保前端不抛异常。

为什么需要这个：前端渲染里任何一处 ReferenceError 都会让 render() 提前退出，
界面就停在上一屏不动——而 Python 这边的测试一个都抓不到。
真踩过一次：终局画面调了一个已经被删掉的函数，玩家永远卡在"结算中"。

需要 node。没装 node 就跳过。
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

import ai  # noqa: E402
import rules  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402
from game import Game  # noqa: E402
from models import Origin  # noqa: E402

CFG = DEFAULT_CONFIG
NODE = shutil.which("node")


def expected_card_effects(rank: int = 0, value: int = 10) -> dict:
    """牌面上该显示的数字，由**引擎**算出来，交给前端比对。

    踩过：以权谋私的政绩在前端写死了 /2，配置改成 1/4 之后牌面一直显示错的
    （该 +2 却写 +5），而所有 Python 测试都照样绿。
    """
    import rules

    return {
        "rank": rank,
        "value": value,
        "WORK": rules.work_merit(value, rank, None, CFG),
        "CORRUPT": rules.corrupt_money(value, rank, None, CFG),
        "GRAFT_money": rules.corrupt_money(value, rank, None, CFG),
        "GRAFT_merit": rules.graft_merit(value, rank, None, CFG),
    }


def api_config_payload() -> dict:
    """直接问真正的 /api/config 要数据，免得测试里的规则表和线上对不上。"""
    import asyncio

    import server

    resp = asyncio.run(server.api_config())
    return json.loads(resp.body)


def build_payloads() -> list[dict]:
    """打一整局，把每个阶段的 public/private payload 录下来。"""
    out: list[dict] = []
    game = Game(game_id="uismoke", cfg=CFG, rng=random.Random(4))
    for i in range(4):
        game.add_player(f"P{i + 1}")

    def snap(label: str, pid: int | None = 1) -> None:
        out.append(
            {
                "label": label,
                "public": game.public_state(),
                "private": game.private_state(pid) if pid else None,
                "my_id": pid,
            }
        )

    snap("大厅")
    snap("未加入（旁观）", None)

    # 挑出身：三张候选摊开、已经选定、以及"别人还没选"三种样子都要能画
    game.start_game(draft_origins=True)
    snap("挑出身")
    game.choose_origin(1, game.origin_choices[1][0])
    snap("挑出身（自己已选，等别人）")
    game.force_origins()
    snap("行动选择")

    # 选了两张牌 -> 结算顺序面板要能渲染（含晋升卡那几条提示分支）
    hand = game.hands[1]
    promo = next((i for i, d in enumerate(hand) if d.card.is_promotion), None)
    other = next(i for i in range(len(hand)) if i != promo)
    for label, pair in (
        ("行动选择（两张牌，排好顺序）", [0, 1]),
        ("行动选择（晋升卡在前）", [promo, other] if promo is not None else [0, 1]),
        ("行动选择（晋升卡在后）", [other, promo] if promo is not None else [1, 0]),
    ):
        out.append({
            "label": label,
            "public": game.public_state(),
            "private": game.private_state(1),
            "my_id": 1,
            "picks": [
                {"index": i, "card": hand[i].card.value,
                 "target": 2 if hand[i].card.needs_target else None}
                for i in pair
            ],
        })

    pool = ai.AgentPool(cfg=CFG, rng=random.Random(4))
    first = True
    while not game.is_over:
        for pid in sorted(game.players):
            game.select_actions(pid, ai.choose(game, pid, pool))
            game.lock_action(pid)
        game.reveal_event()
        if first:
            snap("事件揭示")
        game.resolve()
        if first:
            snap("本轮结算")
            first = False
        if not game.is_over:
            game.advance_round()
    snap("游戏结束（全揭示）")

    # 每个出身各来一屏：面板上那几句断言会被技能推翻（红二代降不下来、
    # 会计只被抄一半），渲染不能照着通用规则写。
    for oid in CFG.origin_ids():
        og = Game(game_id=f"ui{oid}", cfg=CFG, rng=random.Random(5))
        for i in range(3):
            og.add_player(f"R{i + 1}")
        og.players[1].origin = Origin(oid)
        og.start_game()
        og.players[1].warnings = CFG.warnings_before_demotion - 1
        out.append({
            "label": f"行动选择（{CFG.origin(oid)['name']}）",
            "public": og.public_state(),
            "private": og.private_state(1),
            "my_id": 1,
            "panel_must_say": (
                ["降不下来"] if oid == "RED"
                else ["一半"] if oid == "ACCOUNTANT"
                else []
            ),
            "panel_must_not_say": ["再记"] if oid == "RED" else [],
        })

    # 官二代的晋升门槛和别人不一样，规则表必须按**他自己的**那份画。
    # 用 /api/config 里那份通用的，他会看到"还差 15"而结算只要 10。
    vip = Game(game_id="uivip", cfg=CFG, rng=random.Random(9))
    for i in range(3):
        vip.add_player(f"Q{i + 1}")
    vip.players[1].origin = Origin.OFFICIAL
    vip.start_game()
    out.append({
        "label": "行动选择（官二代，门槛打折）",
        "public": vip.public_state(),
        "private": vip.private_state(1),
        "my_id": 1,
        # 引擎算出来的那份，JS 要逐个对上
        "expect_merit_costs": [
            rules.merit_cost_at(r, "OFFICIAL", CFG) for r in range(CFG.president_rank)
        ],
    })
    return out


@unittest.skipUnless(NODE, "需要 node 才能跑前端渲染")
class TestUIRenders(unittest.TestCase):
    def test_every_screen_renders_without_throwing(self):
        payloads = build_payloads()
        labels = [p["label"] for p in payloads]
        self.assertIn("游戏结束（全揭示）", labels, "得覆盖到终局画面")

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8") as fh:
            json.dump({"config": api_config_payload(), "screens": payloads,
                       "card_effects": expected_card_effects(),
                       # 出身文案只在 config.py 定义一处，前端不许自己写一份
                       "origins": [dict(o) for o in CFG.origin_definitions],
                       # 规则速查里"抢官大的人更值"那句的两个数
                       "work_at_ranks": [
                           rules.work_merit(
                               round(
                                   sum(v * n for v, n in CFG.work_card_distribution)
                                   / sum(n for _, n in CFG.work_card_distribution)
                               ),
                               r, None, CFG,
                           )
                           for r in (0, CFG.president_rank - 1)
                       ]},
                      fh, ensure_ascii=False)
            path = fh.name
        try:
            proc = subprocess.run(
                [NODE, str(APP_DIR / "tests" / "ui_smoke.js"), path],
                capture_output=True, text=True, timeout=60,
            )
        finally:
            Path(path).unlink(missing_ok=True)
        self.assertEqual(
            proc.returncode, 0,
            f"有界面渲染失败：\n{proc.stdout}\n{proc.stderr}",
        )

    def test_app_js_is_syntactically_valid(self):
        proc = subprocess.run(
            [NODE, "--check", str(APP_DIR / "static" / "app.js")],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
