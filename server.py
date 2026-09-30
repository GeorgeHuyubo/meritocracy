"""Meritocracy Web 服务器。

    python3 /meritocracy/server.py            # 默认 0.0.0.0:8000
    python3 /meritocracy/server.py --port 9000

这一层只做四件事：房间管理、WebSocket 收发、阶段推进的计时、持久化，
外加替 AI 玩家出牌。所有规则判断都在 game.py / rules.py 里，
客户端提交的任何东西都不被信任。

AI 玩家走的是和真人**完全一样**的接口：`ai.choose()` 只吃
`game.public_state()` 和它自己的 `game.private_state(pid)`，
所以它看到的信息和浏览器里那个玩家一模一样，作弊在结构上就不可能。

私密数据隔离：每个连接只会收到
    public  = game.public_state()              （所有人一致）
    private = game.private_state(自己的 id)     （只有本人）
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import secrets
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
except ModuleNotFoundError as exc:  # pragma: no cover - 环境缺依赖时的友好提示
    raise SystemExit(
        f"缺少依赖 {exc.name}。请先安装：\n"
        f"    python3 -m pip install -r {APP_DIR / 'requirements.txt'}\n"
    ) from exc

import ai
from config import DEFAULT_CONFIG
from game import Game, GameError
from models import Phase
from storage import GameStore

STATIC_DIR = APP_DIR / "static"
DB_PATH = os.environ.get("MERITOCRACY_DB", str(APP_DIR / "meritocracy.db"))
# 房间号一律大写（客户端输入也会转大写），默认房间名跟着规范化，否则对不上
DEFAULT_ROOM = os.environ.get("MERITOCRACY_GAME_ID", "MAIN").strip().upper()

AI_NAMES = ["老张", "老李", "小王", "老陈", "小刘", "老赵", "小孙", "老周"]
ROOM_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # 去掉了形近字


def new_room_code() -> str:
    return "".join(secrets.choice(ROOM_ALPHABET) for _ in range(4))


class Room:
    """一个房间 = 一局游戏 + 它的连接 + 它的 AI 大脑。"""

    def __init__(self, code: str, cfg=DEFAULT_CONFIG, game: Game | None = None) -> None:
        self.code = code
        self.cfg = cfg
        self.game = game or Game(game_id=code, cfg=cfg)
        self.connections: dict[WebSocket, int | None] = {}
        self.lock = asyncio.Lock()
        self.reveal_task: asyncio.Task | None = None
        self.pool = ai.AgentPool(cfg=cfg, rng=random.SystemRandom())

    def reset_brains(self) -> None:
        """新开一局时 AI 的记忆要清空，不能跨局。"""
        self.pool = ai.AgentPool(cfg=self.cfg, rng=random.SystemRandom())


class Hub:
    def __init__(self) -> None:
        self.store = GameStore(DB_PATH)
        self.cfg = DEFAULT_CONFIG
        self.rooms: dict[str, Room] = {}
        self.ws_room: dict[WebSocket, str] = {}
        # 默认房间：不输房间号直接加入的人都进这里
        self.rooms[DEFAULT_ROOM] = self._restore_or_new(DEFAULT_ROOM)

    def _restore_or_new(self, code: str) -> Room:
        restored = self.store.load(code, cfg=self.cfg)
        if restored is not None and not restored.is_over:
            print(f"[恢复] 房间 {code}：第 {restored.round_number} 轮，阶段 {restored.phase.value}")
            return Room(code, self.cfg, restored)
        return Room(code, self.cfg)

    # -- 房间 ---------------------------------------------------------

    def room_of(self, ws: WebSocket) -> Room:
        code = self.ws_room.get(ws, DEFAULT_ROOM)
        room = self.rooms.get(code)
        if room is None:
            raise GameError("房间不存在或已解散。")
        return room

    def create_room(self) -> Room:
        for _ in range(50):
            code = new_room_code()
            if code not in self.rooms:
                room = Room(code, self.cfg)
                self.rooms[code] = room
                return room
        raise GameError("房间号用尽了，稍后再试。")

    # -- 广播 ---------------------------------------------------------

    def _payload_for(self, room: Room, ws: WebSocket, public: dict[str, Any]) -> dict[str, Any]:
        player_id = room.connections.get(ws)
        private = None
        if player_id is not None and player_id in room.game.players:
            private = room.game.private_state(player_id)
        return {"type": "state", "room": room.code, "public": public, "private": private}

    async def broadcast(self, room: Room) -> None:
        public = room.game.public_state()
        dead: list[WebSocket] = []
        for ws in list(room.connections):
            try:
                await ws.send_json(self._payload_for(room, ws, public))
            except Exception:
                dead.append(ws)
        for ws in dead:
            room.connections.pop(ws, None)
            self.ws_room.pop(ws, None)

    async def send_error(self, ws: WebSocket, message: str) -> None:
        try:
            await ws.send_json({"type": "error", "message": message})
        except Exception:
            pass

    def save(self, room: Room) -> None:
        try:
            self.store.save(room.game)
        except Exception as exc:  # 持久化失败不能影响对局
            print(f"[警告] 房间 {room.code} 保存失败：{exc}")

    # -- AI 代打 -------------------------------------------------------

    def drive_ai(self, room: Room) -> bool:
        """替所有 AI 玩家做完这一步。返回是否真的动过。

        用的是和浏览器完全一样的 public/private payload，看不到别人的钱和手牌。
        """
        game = room.game
        moved = False
        if game.phase is Phase.ORIGIN_SELECT:
            for pid in game.ai_player_ids():
                if game.players[pid].origin is not None:
                    continue
                try:
                    game.choose_origin(pid, ai.choose_origin(game, pid, room.pool))
                    moved = True
                except GameError as exc:
                    print(f"[AI] 房间 {room.code} 玩家 {pid} 挑出身失败：{exc}")
                    game.force_origins()
                    moved = True
        elif game.phase is Phase.ACTION_SELECTION:
            for pid in game.ai_player_ids():
                sel = game.selections.get(pid)
                if sel is None or sel.locked:
                    continue
                try:
                    # 有人马上要赢、而手上没有干扰牌时，AI 会花钱重抽去拦他。
                    # 换一次还是没摸到干扰牌就接着换：他登顶游戏就结束了，
                    # 这时候省下来的钱一分都花不出去。价格逐次翻倍，
                    # 付不起时 wants_redraw 自己会转 False，不会换个没完。
                    for _ in range(ai.MAX_PANIC_REDRAWS):
                        if not ai.wants_redraw(game, pid, room.pool):
                            break
                        game.redraw(pid)
                    game.select_actions(pid, ai.choose(game, pid, room.pool))
                    game.lock_action(pid)
                    moved = True
                except GameError as exc:  # AI 出了非法牌不能拖垮整局
                    print(f"[AI] 房间 {room.code} 玩家 {pid} 出牌失败：{exc}")
                    game.force_lock_all()
                    moved = True
        elif game.phase is Phase.ROUND_RESULT:
            for pid in game.ai_player_ids():
                if pid not in game.ready:
                    game.mark_ready(pid)
                    moved = True
        return moved

    # -- 阶段推进 -----------------------------------------------------

    async def maybe_advance(self, room: Room) -> None:
        """所有人锁定 -> 揭示事件 -> 结算；所有人确认 -> 下一轮。"""
        game = room.game
        before = game.phase
        if self.drive_ai(room):
            self.save(room)
            await self.broadcast(room)

        # AI 挑完出身可能直接把阶段推进到出牌了，但那一趟它们还没出牌，
        # 得再跑一次 drive_ai，否则全场卡在"等 AI 锁定"
        if before is Phase.ORIGIN_SELECT and game.phase is not before:
            await self.maybe_advance(room)
            return

        if game.phase is Phase.ACTION_SELECTION and game.all_locked():
            if room.reveal_task is None or room.reveal_task.done():
                room.reveal_task = asyncio.create_task(self._reveal_and_resolve(room))
        elif game.phase is Phase.ROUND_RESULT and game.everyone_ready():
            game.advance_round()
            self.save(room)
            await self.broadcast(room)
            await self.maybe_advance(room)  # 新一轮里 AI 立刻出牌

    async def _reveal_and_resolve(self, room: Room) -> None:
        async with room.lock:
            if room.game.phase is not Phase.ACTION_SELECTION or not room.game.all_locked():
                return
            room.game.reveal_event()
            self.save(room)
        await self.broadcast(room)

        await asyncio.sleep(self.cfg.reveal_event_seconds)

        async with room.lock:
            if room.game.phase is not Phase.REVEAL_EVENT:
                return
            room.game.resolve()
            self.save(room)
        await self.broadcast(room)
        await self.maybe_advance(room)  # 让 AI 直接点"继续"

    # -- 客户端消息 ---------------------------------------------------

    async def handle(self, ws: WebSocket, msg: dict[str, Any]) -> None:
        kind = msg.get("type")

        if kind == "create":
            room = self.create_room()
            self.ws_room[ws] = room.code
            room.connections[ws] = None
            async with room.lock:
                player = room.game.add_player(str(msg.get("name") or ""))
                room.connections[ws] = player.id
                self.save(room)
            await ws.send_json({"type": "identity", "room": room.code,
                                "player_id": player.id, "token": player.token})
            await self.broadcast(room)
            return

        if kind in ("join", "join_room"):
            code = str(msg.get("room") or DEFAULT_ROOM).strip().upper()
            room = self.rooms.get(code)
            if room is None:
                raise GameError(f"房间 {code} 不存在。")
            if room.connections.get(ws) is not None:
                raise GameError("你已经加入过了。")
            self.ws_room[ws] = code
            room.connections[ws] = None
            async with room.lock:
                player = room.game.add_player(str(msg.get("name") or ""))
                room.connections[ws] = player.id
                self.save(room)
            await ws.send_json({"type": "identity", "room": code,
                                "player_id": player.id, "token": player.token})
            await self.broadcast(room)
            return

        if kind == "hello":
            code = str(msg.get("room") or DEFAULT_ROOM).strip().upper()
            room = self.rooms.get(code) or self.rooms[DEFAULT_ROOM]
            self.ws_room[ws] = room.code
            player = room.game.player_by_token(str(msg.get("token") or ""))
            if player is not None:
                room.connections[ws] = player.id
                player.connected = True
                await ws.send_json({"type": "identity", "room": room.code,
                                    "player_id": player.id, "token": player.token})
            else:
                room.connections[ws] = None
            await self.broadcast(room)
            await self.maybe_advance(room)
            return

        room = self.room_of(ws)
        game = room.game
        player_id = room.connections.get(ws)
        if player_id is None:
            raise GameError("你还没有加入游戏。")

        if kind == "add_ai":
            async with room.lock:
                if player_id != game.host_id:
                    raise GameError("只有房主可以添加 AI。")
                used = {p.name for p in game.players.values()}
                name = next((n for n in AI_NAMES if n not in used), None)
                if name is None:
                    name = f"AI{len(game.players) + 1}"
                game.add_player(name, is_ai=True)
                self.save(room)
            await self.broadcast(room)

        elif kind == "remove_ai":
            async with room.lock:
                game.remove_player(int(msg.get("player_id")), requester_id=player_id)
                self.save(room)
            await self.broadcast(room)

        elif kind == "start":
            async with room.lock:
                # 线上才走"挑出身"那一步；模拟器和测试直接开局
                game.start_game(requester_id=player_id, draft_origins=True)
                self.save(room)
            await self.broadcast(room)
            await self.maybe_advance(room)

        elif kind == "choose_origin":
            async with room.lock:
                game.choose_origin(player_id, msg.get("origin"))
                self.save(room)
            await self.broadcast(room)
            await self.maybe_advance(room)

        elif kind == "force_origins":
            async with room.lock:
                game.force_origins(requester_id=player_id)
                self.save(room)
            await self.broadcast(room)
            await self.maybe_advance(room)

        elif kind == "select":
            async with room.lock:
                game.select_actions(player_id, msg.get("picks"))
                self.save(room)
            await self.broadcast(room)

        elif kind == "redraw":
            async with room.lock:
                game.redraw(player_id)
                self.save(room)
            await self.broadcast(room)

        elif kind == "lock":
            async with room.lock:
                if msg.get("picks") is not None:
                    game.select_actions(player_id, msg["picks"])
                game.lock_action(player_id)
                self.save(room)
            await self.broadcast(room)
            await self.maybe_advance(room)

        elif kind == "force":
            async with room.lock:
                game.force_lock_all(requester_id=player_id)
                self.save(room)
            await self.broadcast(room)
            await self.maybe_advance(room)

        elif kind == "ready":
            async with room.lock:
                game.mark_ready(player_id)
                self.save(room)
            await self.broadcast(room)
            await self.maybe_advance(room)

        elif kind == "reset":
            async with room.lock:
                # 任何阶段都能结束（中途弃局也算），原班人马留在房间里
                if room.reveal_task is not None and not room.reveal_task.done():
                    room.reveal_task.cancel()
                    room.reveal_task = None
                game.restart(requester_id=player_id)
                room.reset_brains()
                self.store.reset(room.code)
                self.save(room)
            await self.broadcast(room)

        else:
            raise GameError(f"未知指令：{kind}")

    async def disconnect(self, ws: WebSocket) -> None:
        code = self.ws_room.pop(ws, None)
        room = self.rooms.get(code) if code else None
        if room is None:
            return
        player_id = room.connections.pop(ws, None)
        if player_id is not None and player_id in room.game.players:
            still_online = any(pid == player_id for pid in room.connections.values())
            if not still_online:
                room.game.players[player_id].connected = False
        await self.broadcast(room)
        await self.maybe_advance(room)


hub: Hub | None = None


@asynccontextmanager
async def lifespan(_app: "FastAPI"):
    global hub
    hub = Hub()
    yield
    if hub is not None:
        hub.store.close()


app = FastAPI(title="Meritocracy", lifespan=lifespan)


def _asset_version() -> str:
    """用静态文件的修改时间做版本戳，改了代码就自动失效浏览器缓存。"""
    stamps = []
    for name in ("app.js", "style.css"):
        f = STATIC_DIR / name
        stamps.append(str(int(f.stat().st_mtime)) if f.exists() else "0")
    return "-".join(stamps)


@app.get("/")
async def index() -> HTMLResponse:
    """首页每次都现读现发，并给 js/css 挂上版本戳。

    不这么做的话，改完前端代码你会一直看到浏览器缓存里的旧界面——
    这个坑踩过一次。
    """
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    v = _asset_version()
    html = html.replace("/static/app.js", f"/static/app.js?v={v}")
    html = html.replace("/static/style.css", f"/static/style.css?v={v}")
    return HTMLResponse(
        html, headers={"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"}
    )


@app.get("/api/config")
async def api_config() -> JSONResponse:
    """只暴露规则说明用的公开配置，不含任何对局状态。"""
    cfg = DEFAULT_CONFIG
    return JSONResponse(
        {
            "max_rounds": cfg.max_rounds,
            "rank_names": cfg.rank_names,
            "tenure_required": cfg.tenure_required,
            "major_corruption_threshold": cfg.major_corruption_threshold,
            "promotion_money_costs": cfg.promotion_money_costs,
            "promotion_merit_costs": cfg.promotion_merit_costs,
            "report_reward_ratio": str(cfg.report_reward_ratio),
            "redraw_costs": list(cfg.redraw_costs),
            "redraw_cost_growth": cfg.redraw_cost_growth,
            "warnings_before_demotion": cfg.warnings_before_demotion,
            "major_corruption_warnings": cfg.major_corruption_warnings,
            "promotion_requires_both": [
                cfg.needs_both(r) for r in range(cfg.president_rank)
            ],
            "rank_multipliers": [str(m) for m in cfg.rank_multipliers],
            "rank_salary": list(cfg.rank_salary),
            "hand_size": cfg.hand_size,
            "picks_per_round": cfg.picks_per_round,
            "president_rank": cfg.president_rank,
            "merit_overflow_divisor": cfg.merit_overflow_divisor,
            "money_overflow_divisor": cfg.money_overflow_divisor,
            "attack_merit_penalty": cfg.attack_merit_penalty,
            "attack_steal_fraction": str(cfg.attack_steal_fraction),
            "attack_mode": cfg.attack_mode,
            "attack_hush_money_ratio": str(cfg.attack_hush_money_ratio),
            "attack_corruption_merit_ratio": str(cfg.attack_corruption_merit_ratio),
            "graft_merit_ratio": str(cfg.graft_merit_ratio),
            "storm_fraction": str(cfg.event_storm_fraction),
            "final_ranking_keys": list(cfg.final_ranking_keys),
            "events": [
                {"name": e["name"], "description": e["description"]}
                for e in cfg.event_definitions
                if e["weight"] > 0
            ],
            "origins": [dict(o) for o in cfg.origin_definitions]
            if cfg.origins_enabled else [],
            "origin_choices_offered": cfg.origin_choices_offered,
        }
    )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    assert hub is not None
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            try:
                await hub.handle(ws, msg)
            except GameError as exc:
                await hub.send_error(ws, str(exc))
            except Exception as exc:  # 规则之外的意外，不能让连接直接断掉
                await hub.send_error(ws, f"服务器内部错误：{exc}")
    except WebSocketDisconnect:
        pass
    finally:
        await hub.disconnect(ws)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Meritocracy 服务器")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    args = parser.parse_args()

    banner = (
        "=" * 52,
        "  Meritocracy 服务器已启动",
        f"  本机访问：  http://127.0.0.1:{args.port}",
        f"  局域网访问：http://{_lan_ip()}:{args.port}",
        "=" * 52,
    )
    print("\n".join(banner), flush=True)  # 重定向到文件时也能立刻看到地址
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
