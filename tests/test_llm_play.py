"""llm_play.py：不调模型（mock 后端），确认整局流程、提示词、反馈和报告都跑得通。"""

from __future__ import annotations

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
        self.assertTrue((out / "report.md").exists())

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

    def test_extract_json_tolerates_chatter(self):
        got = llm_play.extract_json('好的，我的决定是：\n{"picks": [{"index": 2, "target": 4}], "reason": "拦他"}\n以上')
        self.assertEqual(got["picks"][0]["target"], 4)


if __name__ == "__main__":
    unittest.main()
