"""llm_play.py：不调模型（mock 后端），确认整局流程、提示词、反馈和报告都跑得通。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llm_play  # noqa: E402


class TestLlmPlayPipeline(unittest.TestCase):
    def test_mock_game_and_report(self):
        out = Path(tempfile.mkdtemp())
        res = llm_play.play_one((0, 7, "sonnet", "mock", out))
        self.assertEqual(len(res["players"]), 6)
        self.assertTrue(any(p["winner"] for p in res["players"]))
        self.assertGreater(res["stats"]["prompt_chars"], 1000)  # 每一席的提示词都拼过
        self.assertTrue(all("strongest" in p["feedback"] for p in res["players"]))
        text = llm_play.report(out, summarize=False, model="sonnet", backend_name="mock")
        self.assertIn("各身份战绩", text)
        self.assertTrue((out / "report_quick.md").exists())  # mock / 不归纳：写快速报告，不碰 report.md

    def test_session_backend_speaks_stream_json(self):
        """常驻会话模式：用一个假的 claude（读 stream-json、回 result 事件）把通信走一遍。"""
        import os
        import stat

        fake_dir = Path(tempfile.mkdtemp())
        fake = fake_dir / "claude"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import sys, json\n"
            "for line in sys.stdin:\n"
            "    msg = json.loads(line)['message']['content']\n"
            "    res = ('{\"strongest\": \"卷王\", \"weakest\": \"贫农\", \"fun\": 7}' if '对局结束' in msg\n"
            "           else '{\"picks\": [], \"reason\": \"观望\"}')\n"
            "    print(json.dumps({'type': 'result', 'result': res}), flush=True)\n",
            encoding="utf-8")
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        old = os.environ["PATH"]
        os.environ["PATH"] = f"{fake_dir}:{old}"
        try:
            res = llm_play.play_one((0, 5, "sonnet", "session", Path(tempfile.mkdtemp()), []))
        finally:
            os.environ["PATH"] = old
        self.assertGreater(res["stats"]["llm_ok"], 0)
        self.assertEqual(res["stats"].get("llm_fallback", 0), 0)
        self.assertEqual(res["players"][0]["feedback"]["strongest"], "卷王")

    def test_llm_decides_its_own_redraws(self):
        """大模型说要换牌就真的换（花钱），然后看着新手牌出牌；后备 AI 不会替它换。"""
        import os
        import stat

        fake_dir = Path(tempfile.mkdtemp())
        fake = fake_dir / "claude"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import sys, json\n"
            "for line in sys.stdin:\n"
            "    msg = json.loads(line)['message']['content']\n"
            "    if '对局结束' in msg:\n"
            "        res = '{\"strongest\": \"卷王\", \"weakest\": \"贫农\", \"fun\": 7}'\n"
            "    elif '【新一轮】' in msg and '你现在钱不够' not in msg and '换牌：' in msg:\n"
            "        res = '{\"redraw\": true}'\n"
            "    else:\n"
            "        res = '{\"picks\": [], \"reason\": \"观望\"}'\n"
            "    print(json.dumps({'type': 'result', 'result': res}), flush=True)\n",
            encoding="utf-8")
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        old = os.environ["PATH"]
        os.environ["PATH"] = f"{fake_dir}:{old}"
        try:
            res = llm_play.play_one((0, 9, "sonnet", "session", Path(tempfile.mkdtemp()), [], "peasant_old"))
        finally:
            os.environ["PATH"] = old
        self.assertGreater(res["stats"].get("redraws", 0), 0)
        self.assertEqual(res["variant"], "peasant_old")
        self.assertTrue(any(d["redraws"] for d in res["decisions"]))

    def _fake_claude(self, body: str) -> str:
        """写一个假的 claude，返回放它的目录（调用方自己把它加进 PATH）。"""
        import stat

        d = Path(tempfile.mkdtemp())
        f = d / "claude"
        f.write_text("#!/usr/bin/env python3\nimport sys, json, os\n" + body, encoding="utf-8")
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
        return str(d)

    def _with_path(self, d: str, fn):
        import os

        old = os.environ["PATH"]
        os.environ["PATH"] = f"{d}:{old}"
        os.environ["LLM_BACKOFF_BASE"] = "0.01"
        try:
            return fn()
        finally:
            os.environ["PATH"] = old
            os.environ.pop("LLM_BACKOFF_BASE", None)

    def test_resume_skips_finished_games_and_feedback_every(self):
        """--resume 跳过已经跑完的局；--feedback-every 2 只有偶数局收反馈。"""
        out = Path(tempfile.mkdtemp())
        llm_play.main(["--games", "2", "--backend", "mock", "--no-summary", "--resume", str(out),
                       "--feedback-every", "2", "--min-free-gb", "0"])
        first = sorted(p.name for p in out.glob("game_*.json"))
        mtime = (out / "game_000.json").stat().st_mtime
        llm_play.main(["--games", "4", "--backend", "mock", "--no-summary", "--resume", str(out),
                       "--feedback-every", "2", "--min-free-gb", "0"])
        self.assertEqual(first, ["game_000.json", "game_001.json"])
        self.assertEqual(len(list(out.glob("game_*.json"))), 4)
        self.assertEqual((out / "game_000.json").stat().st_mtime, mtime)  # 没重跑
        import json as _j
        fb = {g: _j.loads((out / f"game_{g:03d}.json").read_text())["players"][0]["feedback"] for g in range(4)}
        self.assertIn("strongest", fb[0])
        self.assertIsNone(fb[1])
        self.assertTrue((out / "progress.log").exists())

    def test_session_restarts_after_the_process_dies(self):
        """常驻会话的进程中途退出：重开一个接着打，不退回程序 AI。"""
        state = Path(tempfile.mkdtemp()) / "launches"
        d = self._fake_claude(
            f"p = {str(state)!r}\n"
            "n = int(open(p).read()) if os.path.exists(p) else 0\n"
            "open(p, 'w').write(str(n + 1))\n"
            "for i, line in enumerate(sys.stdin):\n"
            "    if n == 0 and i == 1:\n"
            "        sys.exit(1)  # 第一个进程答完一次就死\n"
            "    print(json.dumps({'type': 'result', 'result': '{\"picks\": [], \"reason\": \"x\"}'}), flush=True)\n")
        res = self._with_path(d, lambda: llm_play.play_one(
            (0, 3, "sonnet", "session", Path(tempfile.mkdtemp()), [], "base", False)))
        self.assertGreaterEqual(res["stats"].get("session_restarts", 0), 1)
        self.assertEqual(res["stats"].get("llm_fallback", 0), 0)

    def test_rate_limit_is_retried_not_counted_as_fallback(self):
        """网关限流：退避后重试成功，不算退回程序 AI。"""
        state = Path(tempfile.mkdtemp()) / "count"
        d = self._fake_claude(
            f"p = {str(state)!r}\n"
            "for line in sys.stdin:\n"
            "    n = int(open(p).read()) if os.path.exists(p) else 0\n"
            "    open(p, 'w').write(str(n + 1))\n"
            "    if n == 0:\n"
            "        print(json.dumps({'type': 'result', 'is_error': True, 'result': 'API Error: 429 rate limit'}), flush=True)\n"
            "    else:\n"
            "        print(json.dumps({'type': 'result', 'result': '{\"picks\": [], \"reason\": \"x\"}'}), flush=True)\n")
        res = self._with_path(d, lambda: llm_play.play_one(
            (0, 4, "sonnet", "session", Path(tempfile.mkdtemp()), [], "base", False)))
        self.assertGreaterEqual(res["stats"].get("gateway_retries", 0), 1)
        self.assertEqual(res["stats"].get("llm_fallback", 0), 0)

    def test_notebook_carries_lessons_across_games(self):
        """--notebook：每局打完心得并进笔记本、存快照；下一局开局读到的是更新后的版本；报告里有学习曲线。"""
        out = Path(tempfile.mkdtemp())
        try:
            llm_play.main(["--games", "3", "--parallel", "1", "--backend", "mock", "--no-summary",
                           "--resume", str(out), "--notebook", "--feedback-every", "2", "--min-free-gb", "0"])
        finally:
            llm_play.NOTEBOOK = None
        self.assertTrue((out / "notebook.md").exists())
        self.assertEqual(len(list((out / "notebook_history").glob("after_game_*.md"))), 3)
        import json as _j
        versions = [_j.loads((out / f"game_{g:03d}.json").read_text())["notebook_version"] for g in range(3)]
        self.assertEqual(versions, [0, 1, 2])  # 一局一局跑：每局都读到上一局更新后的笔记本
        self.assertIn("学习曲线", (out / "report_quick.md").read_text(encoding="utf-8"))

    def test_extract_json_tolerates_chatter(self):
        got = llm_play.extract_json('好的，我的决定是：\n{"picks": [{"index": 2, "target": 4}], "reason": "拦他"}\n以上')
        self.assertEqual(got["picks"][0]["target"], 4)


if __name__ == "__main__":
    unittest.main()


class TestMixedSeats(unittest.TestCase):
    """混坐：1 个大模型 + 5 个 Python 陪练，大模型坐的身份按局号轮换，附带同种子的 Python 对照局。"""

    def test_mock_mixed_matches_control_and_report(self):
        out = Path(tempfile.mkdtemp())
        opts = {"llm_seats": 1, "crowd": "learned", "control": 2, "base_seed": 3}
        origins = list(llm_play.DEFAULT_CONFIG.origin_ids())
        for g in range(6):
            res = llm_play.play_one((g, 3000 + g, "sonnet", "mock", out, [], "base", True, 0.0, opts))
            llm = [p for p in res["players"] if p["controller"] == "llm"]
            self.assertEqual(len(llm), 1)
            self.assertEqual(llm[0]["origin"], origins[g % 6])
            self.assertEqual(res["focal"]["origin"], origins[g % 6])
            self.assertEqual(len(res["telemetry"]), res["rounds"])
            # mock 的"大模型"就是同种子的学习型 AI：和第 0 份对照局必须逐局一模一样（同一套代码路径）
            c0 = json.loads((out / f"control_{g:03d}_0.json").read_text(encoding="utf-8"))
            self.assertTrue((out / f"control_{g:03d}_1.json").exists())
            self.assertEqual(c0["winners"], res["winners"])
            strip = lambda tel: [[{k: v for k, v in p.items() if k != "controller"} for p in r["players"]]
                                 for r in tel]
            self.assertEqual(strip(c0["telemetry"]), strip(res["telemetry"]))
        text = llm_play.report(out, summarize=False, model="sonnet", backend_name="mock")
        self.assertIn("1 个大模型 + 5 个 Python 陪练", text)
        self.assertIn("| +0.0 |", text)  # 和对照逐局相同，差值恰好是 0
