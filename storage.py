"""SQLite 持久化。

保存对局、玩家、当前轮次、玩家状态、当前手牌、已选行动、每轮事件与结算结果。
服务器重启后可以把进行中的对局恢复出来。

写入策略很朴素：每次状态变化整局重写（一局最多 6 人 / 10 轮，行数极少），
换来的是"快照一定自洽"，不会出现半新半旧的状态。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from config import Config, DEFAULT_CONFIG
from game import Game

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id           TEXT PRIMARY KEY,
    phase             TEXT NOT NULL,
    round_number      INTEGER NOT NULL,
    max_rounds        INTEGER NOT NULL,
    next_player_id    INTEGER NOT NULL,
    current_event_id  TEXT,
    winners           TEXT NOT NULL DEFAULT '[]',
    game_over_reason  TEXT NOT NULL DEFAULT '',
    public_log        TEXT NOT NULL DEFAULT '[]',
    progress          TEXT NOT NULL DEFAULT '{}',   -- 流水账 / 对局档案 / 累计统计
    updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS players (
    game_id   TEXT NOT NULL,
    player_id INTEGER NOT NULL,
    name      TEXT NOT NULL,
    token     TEXT NOT NULL,
    money     INTEGER NOT NULL,
    merit     INTEGER NOT NULL,
    rank      INTEGER NOT NULL,
    tenure    INTEGER NOT NULL,
    is_ai     INTEGER NOT NULL DEFAULT 0,
    warnings  INTEGER NOT NULL DEFAULT 0,
    origin    TEXT,
    PRIMARY KEY (game_id, player_id)
);

CREATE TABLE IF NOT EXISTS hands (
    game_id      TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    player_id    INTEGER NOT NULL,
    cards        TEXT NOT NULL,
    PRIMARY KEY (game_id, round_number, player_id)
);

CREATE TABLE IF NOT EXISTS actions (
    game_id      TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    player_id    INTEGER NOT NULL,
    picks        TEXT NOT NULL DEFAULT '[]',   -- 一轮可以出多张牌，存 JSON
    locked       INTEGER NOT NULL,
    ready        INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (game_id, round_number, player_id)
);

CREATE TABLE IF NOT EXISTS rounds (
    game_id      TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    event_id     TEXT NOT NULL,
    event_name   TEXT NOT NULL,
    result       TEXT NOT NULL,
    PRIMARY KEY (game_id, round_number)
);
"""


class GameStore:
    def __init__(self, path: str | Path = "meritocracy.db") -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """给已经存在的库补上后加的列（CREATE TABLE IF NOT EXISTS 不会改旧表）。"""
        wanted = {
            "games": [("progress", "TEXT NOT NULL DEFAULT '{}'")],
            # redraw_available 已废弃，旧库里留着不管
            "players": [
                ("warnings", "INTEGER NOT NULL DEFAULT 0"),
                ("origin", "TEXT"),
            ],
        }
        for table, columns in wanted.items():
            have = {
                r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")
            }
            for name, decl in columns:
                if name in have:
                    continue
                try:
                    self.conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {name} {decl}"
                    )
                except sqlite3.OperationalError as exc:
                    # "先查再加"之间别的进程可能已经加上了。两个服务器共用
                    # 同一个库时这是常态（本机就同时跑着 8000 和 8088），
                    # 撞上了就说明列已经在了，不是错误。
                    if "duplicate column" not in str(exc).lower():
                        raise

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------

    def save(self, game: Game) -> None:
        snap = game.to_snapshot()
        gid = snap["game_id"]
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO games (game_id, phase, round_number, max_rounds, next_player_id,
                                   current_event_id, winners, game_over_reason, public_log,
                                   progress, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(game_id) DO UPDATE SET
                    phase=excluded.phase,
                    round_number=excluded.round_number,
                    max_rounds=excluded.max_rounds,
                    next_player_id=excluded.next_player_id,
                    current_event_id=excluded.current_event_id,
                    winners=excluded.winners,
                    game_over_reason=excluded.game_over_reason,
                    public_log=excluded.public_log,
                    progress=excluded.progress,
                    updated_at=datetime('now')
                """,
                (
                    gid,
                    snap["phase"],
                    snap["round_number"],
                    game.cfg.max_rounds,
                    snap["next_player_id"],
                    snap["current_event_id"],
                    json.dumps(snap["winners"]),
                    snap["game_over_reason"],
                    json.dumps(snap["public_log"], ensure_ascii=False),
                    json.dumps(
                        {
                            "ledger": snap["ledger"],
                            "archive": snap["archive"],
                            "stats": snap["stats"],
                            "salary_paid": snap["salary_paid"],
                            "redraw_spent": snap["redraw_spent"],
                            "redraw_count": snap["redraw_count"],
                        },
                        ensure_ascii=False,
                    ),
                ),
            )

            self.conn.execute("DELETE FROM players WHERE game_id = ?", (gid,))
            self.conn.executemany(
                "INSERT INTO players (game_id, player_id, name, token, money, merit, rank,"
                " tenure, is_ai, warnings, origin)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (gid, p["id"], p["name"], p["token"], p["money"], p["merit"], p["rank"],
                     p["tenure"], 1 if p.get("is_ai") else 0, p.get("warnings", 0),
                     p.get("origin"))
                    for p in snap["players"]
                ],
            )

            rnd = snap["round_number"]
            self.conn.execute(
                "DELETE FROM hands WHERE game_id = ? AND round_number = ?", (gid, rnd)
            )
            self.conn.executemany(
                "INSERT INTO hands (game_id, round_number, player_id, cards) VALUES (?, ?, ?, ?)",
                [
                    (gid, rnd, int(pid), json.dumps(cards))
                    for pid, cards in snap["hands"].items()
                ],
            )

            self.conn.execute(
                "DELETE FROM actions WHERE game_id = ? AND round_number = ?", (gid, rnd)
            )
            ready = set(snap["ready"])
            self.conn.executemany(
                "INSERT INTO actions (game_id, round_number, player_id, picks, locked, ready)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        gid,
                        rnd,
                        int(pid),
                        json.dumps(sel["picks"]),
                        1 if sel["locked"] else 0,
                        1 if int(pid) in ready else 0,
                    )
                    for pid, sel in snap["selections"].items()
                ],
            )

            for result in snap["results"]:
                self.conn.execute(
                    "INSERT INTO rounds (game_id, round_number, event_id, event_name, result)"
                    " VALUES (?, ?, ?, ?, ?)"
                    " ON CONFLICT(game_id, round_number) DO UPDATE SET"
                    "   event_id=excluded.event_id, event_name=excluded.event_name,"
                    "   result=excluded.result",
                    (
                        gid,
                        result["round"],
                        result["event"]["id"],
                        result["event"]["name"],
                        json.dumps(result, ensure_ascii=False),
                    ),
                )

    # ------------------------------------------------------------------

    def load(self, game_id: str, cfg: Config = DEFAULT_CONFIG, rng: Any = None) -> Game | None:
        row = self.conn.execute(
            "SELECT * FROM games WHERE game_id = ?", (game_id,)
        ).fetchone()
        if row is None:
            return None
        return self._build(row, cfg, rng)

    def load_latest(self, cfg: Config = DEFAULT_CONFIG, rng: Any = None) -> Game | None:
        row = self.conn.execute(
            "SELECT * FROM games ORDER BY updated_at DESC, rowid DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return self._build(row, cfg, rng)

    def _build(self, row: sqlite3.Row, cfg: Config, rng: Any) -> Game:
        gid = row["game_id"]
        rnd = row["round_number"]
        players = [
            {
                "id": r["player_id"],
                "name": r["name"],
                "token": r["token"],
                "money": r["money"],
                "merit": r["merit"],
                "rank": r["rank"],
                "tenure": r["tenure"],
                "is_ai": bool(r["is_ai"]),
                "warnings": r["warnings"],
                "origin": r["origin"],
            }
            for r in self.conn.execute(
                "SELECT * FROM players WHERE game_id = ? ORDER BY player_id", (gid,)
            )
        ]
        hands = {
            str(r["player_id"]): json.loads(r["cards"])
            for r in self.conn.execute(
                "SELECT * FROM hands WHERE game_id = ? AND round_number = ?", (gid, rnd)
            )
        }
        selections: dict[str, Any] = {}
        ready: list[int] = []
        for r in self.conn.execute(
            "SELECT * FROM actions WHERE game_id = ? AND round_number = ?", (gid, rnd)
        ):
            selections[str(r["player_id"])] = {
                "picks": json.loads(r["picks"]),
                "locked": bool(r["locked"]),
            }
            if r["ready"]:
                ready.append(r["player_id"])
        results = [
            json.loads(r["result"])
            for r in self.conn.execute(
                "SELECT * FROM rounds WHERE game_id = ? ORDER BY round_number", (gid,)
            )
        ]
        try:
            progress = json.loads(row["progress"] or "{}")
        except (KeyError, IndexError, json.JSONDecodeError):
            progress = {}
        snapshot = {
            "game_id": gid,
            "phase": row["phase"],
            "round_number": rnd,
            "next_player_id": row["next_player_id"],
            "current_event_id": row["current_event_id"],
            "winners": json.loads(row["winners"]),
            "game_over_reason": row["game_over_reason"],
            "public_log": json.loads(row["public_log"]),
            "players": players,
            "hands": hands,
            "selections": selections,
            "ready": ready,
            "results": results,
            **progress,
        }
        return Game.from_snapshot(snapshot, cfg=cfg, rng=rng)

    def latest_game_id(self) -> str | None:
        row = self.conn.execute(
            "SELECT game_id FROM games ORDER BY updated_at DESC, rowid DESC LIMIT 1"
        ).fetchone()
        return row["game_id"] if row else None

    def history(self, game_id: str) -> dict[str, Any] | None:
        """整局的原始记录，给复盘工具（replay.py）重演用。

        和 load() 不同：load 只恢复**当前**这一轮，这里把每一轮的手牌、出牌、
        事件都拿出来。手牌存的是换完牌之后那一手（每次状态变化整局重写），
        换牌花了多少钱在 progress.ledger 里。
        """
        row = self.conn.execute(
            "SELECT * FROM games WHERE game_id = ?", (game_id,)
        ).fetchone()
        if row is None:
            return None
        players = [
            {
                "id": r["player_id"],
                "name": r["name"],
                "is_ai": bool(r["is_ai"]),
                "origin": r["origin"],
            }
            for r in self.conn.execute(
                "SELECT * FROM players WHERE game_id = ? ORDER BY player_id", (game_id,)
            )
        ]
        hands: dict[int, dict[int, list]] = {}
        for r in self.conn.execute("SELECT * FROM hands WHERE game_id = ?", (game_id,)):
            hands.setdefault(r["round_number"], {})[r["player_id"]] = json.loads(r["cards"])
        actions: dict[int, dict[int, list]] = {}
        for r in self.conn.execute("SELECT * FROM actions WHERE game_id = ?", (game_id,)):
            actions.setdefault(r["round_number"], {})[r["player_id"]] = json.loads(r["picks"])
        events = {
            r["round_number"]: r["event_id"]
            for r in self.conn.execute("SELECT * FROM rounds WHERE game_id = ?", (game_id,))
        }
        try:
            progress = json.loads(row["progress"] or "{}")
        except json.JSONDecodeError:
            progress = {}
        return {
            "game_id": game_id,
            "phase": row["phase"],
            "winners": json.loads(row["winners"]),
            "players": players,
            "hands": hands,
            "actions": actions,
            "events": events,  # 只有结算过的轮次才有
            "archive": progress.get("archive") or [],
            "ledger": {int(k): v for k, v in (progress.get("ledger") or {}).items()},
        }

    def reset(self, game_id: str) -> None:
        with self.conn:
            for table in ("games", "players", "hands", "actions", "rounds"):
                self.conn.execute(f"DELETE FROM {table} WHERE game_id = ?", (game_id,))
