"""自我对局训练学习型 AI（ai.LearnedAgent）：每局赢家 reward 1（并列按 1/人数），其他人 0。

    python3 learn.py --minutes 100                 # 从头训（第 0 代 ≈ 手写 AI），写 policies/latest.json
    python3 learn.py --minutes 100 --resume        # 接着上次的快照训

算法：线性 softmax 策略 + REINFORCE。
  * 每个学习型座位记下每次决策的 ∇log π = φ(选中的组合) − E_π[φ]
  * 局末按 (reward − 1/6) 回传：赢了就往这局做过的选择那边推，输了就往反方向推
  * Adam 更新，8 个进程并行打一批局、回传梯度和

对手联盟（只和自己打会原地转圈，或者学出只会打自己的怪招）：
  * 一半的局六个座位全是当前策略
  * 三成：3 个当前策略 + 3 个手写 AI（这部分的胜率就是"比手写 AI 强多少"的进度条）
  * 两成：3 个当前策略 + 3 个历史快照
  * 再有 15% 的局把一个非学习座位换成脚本打法（只贪 / 只干活 / 只举报……），让它见过各种怪人
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ai  # noqa: E402
import analysis  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402

N = 6
POLICY_DIR = Path(__file__).resolve().parent / "policies"
INIT_THETA = {"hand": 50.0}  # 1 / 温度 0.02：第 0 代 ≈ 手写 AI 的"最高分 + 一点噪声"
BOTS = ["corrupt", "worker", "climber", "reporter", "attacker", "builder"]
TRACKED = ["dirty", "n_report", "n_attack", "promote", "promote_money", "family"]


def _learner(cfg, rng, theta, record):
    pool = ai.AgentPool(cfg=cfg, rng=rng, policy=theta, record=record)
    return pool, (lambda game, me, hand, others, _rng: ai.turn(game, me.id, pool))


def _smart(cfg, rng):
    pool = ai.AgentPool(cfg=cfg, rng=rng)
    return lambda game, me, hand, others, _rng: ai.turn(game, me.id, pool)


def play_batch(args: tuple) -> dict[str, Any]:
    theta, league, n_games, seed = args
    cfg = DEFAULT_CONFIG
    rng = random.Random(seed)
    grad: Counter = Counter()
    chosen: Counter = Counter()
    stats: Counter = Counter()
    for _ in range(n_games):
        r = rng.random()
        kind = "self" if r < 0.5 else ("smart" if r < 0.8 or not league else "league")
        seats: list[Any] = []
        learner_pools: dict[int, Any] = {}
        n_learn = N if kind == "self" else 3
        order = list(range(N))
        rng.shuffle(order)
        learn_seats = set(order[:n_learn])
        bot_seat = None
        if kind != "self" and rng.random() < 0.15:
            bot_seat = order[n_learn]
        for s in range(N):
            if s in learn_seats:
                pool, fn = _learner(cfg, rng, theta, True)
                learner_pools[s] = pool
                seats.append(fn)
            elif s == bot_seat:
                seats.append(analysis.STRATEGIES[rng.choice(BOTS)])
            elif kind == "league":
                seats.append(_learner(cfg, rng, rng.choice(league), False)[1])
            else:
                seats.append(_smart(cfg, rng))
        origins = list(cfg.origin_ids())
        rng.shuffle(origins)
        rec = analysis.play(N, rng, cfg, seats, origins=origins[:N])
        pids = sorted(rec.cards)
        for s, pool in learner_pools.items():
            pid = pids[s]
            reward = 1.0 / len(rec.winners) if pid in rec.winners else 0.0
            adv = reward - 1.0 / N
            agent = pool.agents.get(pid)
            if agent is None:
                continue
            for g in agent.trace:
                for k, v in g.items():
                    grad[k] += adv * v
            stats["trajectories"] += 1
            stats[f"reward_{kind}"] += reward
            stats[f"seats_{kind}"] += 1
            stats["decisions"] += len(agent.trace)
            chosen.update(agent.chosen_stats)
        stats["games"] += 1
        stats["president"] += bool(rec.ended_by_president)
    return {"grad": dict(grad), "stats": dict(stats), "chosen": dict(chosen)}


class Adam:
    def __init__(self, lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m: dict[str, float] = {}
        self.v: dict[str, float] = {}
        self.t = 0

    def step(self, theta: dict[str, float], grad: dict[str, float]) -> None:
        """梯度上升（reward 越大越好）。"""
        self.t += 1
        for k, g in grad.items():
            self.m[k] = self.b1 * self.m.get(k, 0.0) + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v.get(k, 0.0) + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            theta[k] = theta.get(k, 0.0) + self.lr * mh / (math.sqrt(vh) + self.eps)

    def state(self) -> dict[str, Any]:
        return {"m": self.m, "v": self.v, "t": self.t, "lr": self.lr}

    @classmethod
    def from_state(cls, d: dict[str, Any]) -> "Adam":
        a = cls(d["lr"])
        a.m, a.v, a.t = dict(d["m"]), dict(d["v"]), int(d["t"])
        return a


def save(path: Path, theta, adam, league, meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "theta": theta, "adam": adam.state(), "league": league, "meta": meta,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--games-per-worker", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--snapshot-every", type=int, default=10, help="每多少批存一份历史快照进联盟")
    ap.add_argument("--out", default=str(POLICY_DIR / "latest.json"))
    args = ap.parse_args(argv)

    out = Path(args.out)
    if args.resume and out.exists():
        d = json.loads(out.read_text(encoding="utf-8"))
        theta, adam, league, meta = d["theta"], Adam.from_state(d["adam"]), d["league"], d["meta"]
    else:
        theta, adam, league = dict(INIT_THETA), Adam(args.lr), []
        meta = {"batches": 0, "games": 0, "history": []}

    deadline = time.time() + args.minutes * 60
    seed0 = int(time.time()) % 100000
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        while time.time() < deadline:
            b = meta["batches"]
            jobs = [(dict(theta), league, args.games_per_worker, seed0 * 1000 + b * 100 + w)
                    for w in range(args.workers)]
            grad: Counter = Counter()
            stats: Counter = Counter()
            chosen: Counter = Counter()
            for res in ex.map(play_batch, jobs):
                grad.update(res["grad"])
                stats.update(res["stats"])
                chosen.update(res["chosen"])
            n = max(1, stats["trajectories"])
            adam.step(theta, {k: v / n for k, v in grad.items()})
            meta["batches"] += 1
            meta["games"] += stats["games"]
            vs_smart = 100 * stats["reward_smart"] / max(1, stats["seats_smart"])
            row = {
                "batch": meta["batches"], "games": meta["games"],
                "vs_smart": round(vs_smart, 2), "seats_smart": stats["seats_smart"],
                "president": round(100 * stats["president"] / max(1, stats["games"]), 1),
                # 学习型每次出牌里：贪 / 举报 / 攻击 / 买官的比例——看它的打法怎么变
                "mix": {k: round(chosen[k] / max(1, stats["decisions"]), 3) for k in TRACKED},
            }
            meta["history"].append(row)
            if meta["batches"] % args.snapshot_every == 0:
                league = (league + [dict(theta)])[-6:]
            save(out, theta, adam, league, meta)
            top = sorted(((abs(v), k, v) for k, v in theta.items() if k != "hand"), reverse=True)[:5]
            print(f"批 {row['batch']:>4}  累计 {row['games']:>7} 局  "
                  f"学习型 vs 手写同桌每席胜率 {row['vs_smart']:5.1f}%（{row['seats_smart']} 席）  "
                  f"贪{row['mix']['dirty']:.2f} 举报{row['mix']['n_report']:.2f} "
                  f"攻击{row['mix']['n_attack']:.2f}  hand={theta.get('hand', 0):.1f}  "
                  + "  ".join(f"{k}={v:+.2f}" for _, k, v in top), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
