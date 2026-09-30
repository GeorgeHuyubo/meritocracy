"""端到端冒烟测试：真的起一个服务器，用 3 个 WebSocket 客户端打一局。

重点验证"线上传输的东西"本身就不含别人的私密数据——这是单测 TestPrivacy
在内存里验证过的同一件事，但这里走的是真实 payload。

需要 fastapi / uvicorn / websockets；缺依赖时整个用例会被跳过：
    python3 -m pip install -r requirements.txt
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from config import DEFAULT_CONFIG as CFG  # noqa: E402

try:
    import websockets
    import fastapi  # noqa: F401
    import uvicorn  # noqa: F401

    DEPS_OK = True
except ModuleNotFoundError:
    DEPS_OK = False


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Client:
    def __init__(self, url: str) -> None:
        self.url = url
        self.ws = None
        self.public = None
        self.private = None
        self.player_id = None
        self.token = ""
        self.room = ""

    async def __aenter__(self):
        self.ws = await websockets.connect(self.url)
        return self

    async def __aexit__(self, *exc):
        await self.ws.close()

    async def send(self, **payload):
        await self.ws.send(json.dumps(payload))

    async def pump(self, until_state=True, timeout=5.0):
        """读到一条 state 消息为止，顺便记录 identity。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=deadline - time.time())
            msg = json.loads(raw)
            if msg["type"] == "identity":
                self.player_id = msg["player_id"]
                self.token = msg["token"]
                self.room = msg.get("room", self.room)
            elif msg["type"] == "state":
                self.public = msg["public"]
                self.private = msg["private"]
                if until_state:
                    return msg
            elif msg["type"] == "error":
                raise AssertionError(f"server error: {msg['message']}")
        raise AssertionError("timed out waiting for state")

    async def drain(self, seconds=0.4):
        """把积压的广播读干净，返回最后一条 state。"""
        end = time.time() + seconds
        last = None
        while time.time() < end:
            try:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=max(0.05, end - time.time()))
            except asyncio.TimeoutError:
                break
            msg = json.loads(raw)
            if msg["type"] == "state":
                self.public = msg["public"]
                self.private = msg["private"]
                last = msg
            elif msg["type"] == "identity":
                self.player_id = msg["player_id"]
                self.token = msg["token"]
                self.room = msg.get("room", self.room)
        return last


@unittest.skipUnless(DEPS_OK, "需要 fastapi / uvicorn / websockets")
class TestServerEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        env = dict(os.environ)
        env["MERITOCRACY_DB"] = str(Path(cls.tmpdir.name) / "e2e.db")
        env["MERITOCRACY_GAME_ID"] = "e2e"
        # free_port() 是"先 bind 再 close"，和真正启动之间有竞争窗口，
        # 所以失败了就换个端口重来，别让整套测试偶发翻车。
        last_out = ""
        for attempt in range(3):
            cls.port = free_port()
            cls.proc = subprocess.Popen(
                [sys.executable, str(APP_DIR / "server.py"), "--port", str(cls.port)],
                cwd=str(APP_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            for _ in range(100):
                if cls.proc.poll() is not None:
                    break  # 进程已经挂了，直接换端口重试
                try:
                    socket.create_connection(("127.0.0.1", cls.port), timeout=0.2).close()
                    return
                except OSError:
                    time.sleep(0.1)
            cls.proc.kill()
            if cls.proc.stdout:
                last_out = cls.proc.stdout.read().decode(errors="replace")
        raise AssertionError(f"服务器连试 3 次都没起来:\n{last_out}")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.proc.kill()
        cls.tmpdir.cleanup()

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/ws"

    def test_full_round_over_websocket(self):
        asyncio.run(self._scenario())

    async def _scenario(self):
        async with Client(self.url) as a, Client(self.url) as b, Client(self.url) as c:
            for client, name in ((a, "甲"), (b, "乙"), (c, "丙")):
                await client.send(type="hello", token="")
                await client.pump()
                await client.send(type="join", name=name)
                await client.pump()
            await asyncio.gather(a.drain(), b.drain(), c.drain())

            self.assertEqual(len(a.public["players"]), 3)
            self.assertTrue(a.private["is_host"])
            self.assertFalse(b.private["is_host"])

            # ---- 开局：先挑出身 ----
            await a.send(type="start")
            await asyncio.gather(a.drain(), b.drain(), c.drain())
            self.assertEqual(a.public["phase"], "ORIGIN_SELECT")
            for client in (a, b, c):
                offered = client.private["origin_choices"]
                self.assertEqual(len(offered), CFG.origin_choices_offered)
                # 候选要带全文案，前端不用再查一遍表
                for o in offered:
                    self.assertTrue(o["name"] and o["skill"] and o["description"])
                # 自己的候选是自己的事，但出身本身公开——这时候还没人定下来
                self.assertIsNone(
                    next(
                        x for x in client.public["players"]
                        if x["id"] == client.player_id
                    )["origin"]
                )
            for client in (a, b, c):
                await client.send(
                    type="choose_origin", origin=client.private["origin_choices"][0]["id"]
                )
            await asyncio.gather(a.drain(), b.drain(), c.drain())

            # 选完之后出身对所有人可见
            for entry in b.public["players"]:
                self.assertIn(entry["origin"], {o["id"] for o in b.public["origins"]})

            self.assertEqual(a.public["phase"], "ACTION_SELECTION")
            self.assertIsNone(a.public["current_event"], "锁定之前绝不能看到事件")
            self.assertEqual(len(a.private["hand"]), a.public["hand_size"])
            self.assertGreaterEqual(a.public["picks_per_round"], 1)
            # 点数在发牌时就摇好了，玩家选牌之前就能看见
            for d in a.private["hand"]:
                self.assertIn("card", d)
                self.assertIn("value", d)

            # ---- 私密性：广播 payload 里不能出现别人的手牌/金钱 ----
            blob = json.dumps(a.public, ensure_ascii=False)
            for other in (b, c):
                self.assertNotIn(other.token, blob)
            self.assertNotIn("picks\":", blob.replace('"picks_per_round"', ''))
            self.assertNotIn("token", blob)
            for entry in b.public["players"]:
                self.assertNotIn("money", entry)

            # ---- 三个人各出 picks_per_round 张牌 ----
            n = a.public["picks_per_round"]
            for client in (a, b, c):
                others = [
                    p["id"] for p in client.public["players"] if p["id"] != client.player_id
                ]
                # 手牌现在是 [{"card": ..., "value": 发牌时摇好的点数}]，按下标出牌
                picks = []
                for i in range(n):
                    card = client.private["hand"][i]["card"]
                    target = others[0] if card in ("REPORT", "ATTACK") else None
                    picks.append({"index": i, "target": target})
                await client.send(type="lock", picks=picks)
            await asyncio.gather(a.drain(), b.drain(), c.drain())

            # 锁定名单是公开信息，但"锁了什么"不是。
            # 注意不能直接在整个 payload 里搜 "WORK"：事件卡的说明文字里就写着
            # "本轮 WORK 基础政绩 +4"，那是规则文本，不是谁的出牌。
            self.assertEqual(sorted(a.public["locked_players"]), [1, 2, 3])
            without_flavor = {
                k: v for k, v in a.public.items()
                if k not in ("current_event", "last_result", "public_messages")
            }
            blob2 = json.dumps(without_flavor, ensure_ascii=False)
            for card in ("WORK", "CORRUPT", "GRAFT", "REPORT", "ATTACK", "PROMOTE"):
                self.assertNotIn(card, blob2)
            self.assertNotIn("picks\":", blob2)

            # ---- 事件揭示 + 结算（服务器有 2.5 秒的展示间隔）----
            deadline = time.time() + 15
            while a.public["phase"] not in ("ROUND_RESULT", "GAME_OVER") and time.time() < deadline:
                await asyncio.gather(a.drain(0.5), b.drain(0.5), c.drain(0.5))
            self.assertIn(a.public["phase"], ("ROUND_RESULT", "GAME_OVER"))
            self.assertIsNotNone(a.public["current_event"])
            self.assertIsNotNone(a.public["last_result"])
            self.assertEqual(a.public["last_result"]["round"], 1)
            self.assertIsNotNone(a.private["private_result"])

            # ---- 刷新页面：用 token 恢复身份 ----
            async with Client(self.url) as a2:
                await a2.send(type="hello", token=a.token)
                await a2.pump()
                await a2.drain()
                self.assertEqual(a2.player_id, a.player_id)
                self.assertEqual(a2.private["money"], a.private["money"])
                self.assertEqual(a2.private["hand"], a.private["hand"])

            # ---- 进入下一轮 ----
            if a.public["phase"] == "ROUND_RESULT":
                for client in (a, b, c):
                    await client.send(type="ready")
                await asyncio.gather(a.drain(), b.drain(), c.drain())
                self.assertEqual(a.public["phase"], "ACTION_SELECTION")
                self.assertEqual(a.public["round"], 2)
                self.assertIsNone(a.public["current_event"])

    def test_non_player_cannot_act(self):
        asyncio.run(self._spectator())

    async def _spectator(self):
        async with Client(self.url) as spec:
            await spec.send(type="hello", token="")
            await spec.pump()
            self.assertIsNone(spec.private)
            await spec.send(type="lock", picks=[{"index": 0, "target": None}])
            raw = await asyncio.wait_for(spec.ws.recv(), timeout=5)
            self.assertEqual(json.loads(raw)["type"], "error")

    def test_host_can_abort_and_restart_mid_game(self):
        asyncio.run(self._abort())

    async def _abort(self):
        # 这些测试共用一个服务器，默认房间已经被别的用例开局了，所以另开一间
        async with Client(self.url) as a, Client(self.url) as b:
            await a.send(type="create", name="主")
            await a.pump()
            room = a.room
            await b.send(type="join_room", room=room, name="客")
            await b.pump()
            await asyncio.gather(a.drain(), b.drain())
            host_token, guest_token = a.token, b.token

            await a.send(type="start")
            await asyncio.gather(a.drain(), b.drain())
            await self._pick_origins(a, b)
            self.assertEqual(a.public["phase"], "ACTION_SELECTION")

            # 客人不是房主，结束不了
            await b.send(type="reset")
            raw = await asyncio.wait_for(b.ws.recv(), timeout=5)
            self.assertEqual(json.loads(raw)["type"], "error")
            await asyncio.gather(a.drain(), b.drain())
            self.assertEqual(a.public["phase"], "ACTION_SELECTION")

            # 房主中途结束这一局
            await a.send(type="reset")
            await asyncio.gather(a.drain(), b.drain())
            self.assertEqual(a.public["phase"], "LOBBY")
            self.assertEqual(a.public["round"], 0)

            # 原班人马还在，身份没变（不用重新输名字）
            self.assertEqual([p["name"] for p in a.public["players"]], ["主", "客"])
            self.assertEqual((a.token, b.token), (host_token, guest_token))
            self.assertIsNotNone(a.private)
            self.assertIsNotNone(b.private)
            self.assertTrue(a.private["is_host"])

            # 重开之后出身要清干净——上一局的技能带进下一局是 bug，
            # warnings 就这么漏过一次
            for entry in a.public["players"]:
                self.assertIsNone(entry["origin"])

            # 可以直接再开一局
            await a.send(type="start")
            await asyncio.gather(a.drain(), b.drain())
            await self._pick_origins(a, b)
            self.assertEqual(a.public["phase"], "ACTION_SELECTION")
            self.assertEqual(a.public["round"], 1)

    async def _pick_origins(self, *clients):
        """走完"挑出身"那一步：每人取第一个候选。"""
        for client in clients:
            await client.send(
                type="choose_origin", origin=client.private["origin_choices"][0]["id"]
            )
        await asyncio.gather(*(c.drain() for c in clients))


if __name__ == "__main__":
    unittest.main()
