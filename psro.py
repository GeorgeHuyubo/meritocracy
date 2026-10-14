"""PSRO（策略空间响应迭代）：逼近这个 6 人博弈的纳什均衡，作为平衡评估的参照 AI。

像剪刀石头布的均衡是"三种各 1/3"：要找一组打法的混合比例，让谁单方面换打法都占不到便宜。
单独一个 AI 只是一种打法，往往能被针对（富二代专项训练对阵固定陪练 12.7% -> 19.8%），
在那种局面里测出来的平衡会随 AI 变来变去。

每一轮：
  1. 训练一个针对当前混合的最佳应对（learn.py --br-vs，通用权重：坐哪个身份都用同一套）
  2. 用新的一批局测它比 1/6 多赢多少 = 当前混合的**可被剥削度**（按身份拆开），逐轮下降 = 在收敛
  3. 加进打法池，用复制者动态重新求混合比例（每一步按当前混合重新估计各打法的胜率，最后取平均）

    python3 psro.py run --iters 4 --br-minutes 90 --solve-rounds 8 --games 1500 --workers 10
    python3 psro.py run --resume                 # 断了接着跑（按 meta.json 里记的阶段跳过）
    python3 psro.py status                       # 看每轮的可被剥削度和混合比例
    python3 psro.py freeze                       # 把最终混合写成 policies/best.json（旧的备份）
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence

import ai
import analysis
import balance_stats as bs
from config import DEFAULT_CONFIG

ROOT = Path(__file__).resolve().parent
POP = ROOT / "policies" / "pop"
N = 6


# --------------------------------------------------------------------------
# 复制者动态（纯函数，单元测试用剪刀石头布验证）
# --------------------------------------------------------------------------

def replicator(payoff_fn: Callable[[list[float]], list[float]], sigma0: Sequence[float],
               rounds: int, eta: float) -> tuple[list[float], list[dict[str, Any]]]:
    """σ ← σ · exp(η (u − σ·u))，u 每一步按当前 σ 重新估计。返回各步 σ 的平均和每一步的记录。

    循环克制（剪刀石头布）时 σ 本身会绕圈，平均值才收敛到均衡；严格占优时平均值也会偏向占优那个。
    """
    sigma = [max(1e-9, x) for x in sigma0]
    z = sum(sigma)
    sigma = [x / z for x in sigma]
    avg = [0.0] * len(sigma)
    trace = []
    for t in range(rounds):
        u = payoff_fn(sigma)
        v = sum(s * x for s, x in zip(sigma, u))
        trace.append({"round": t, "sigma": [round(x, 4) for x in sigma], "u": [round(x, 4) for x in u],
                      "value": round(v, 4)})
        nxt = [s * math.exp(eta * (x - v)) for s, x in zip(sigma, u)]
        z = sum(nxt)
        sigma = [x / z for x in nxt]
        avg = [a + s for a, s in zip(avg, sigma)]
    return [a / rounds for a in avg], trace


# --------------------------------------------------------------------------
# 模拟：某个打法坐 1 席、其余 5 席从混合里抽
# --------------------------------------------------------------------------

def _seat_from(cfg, rng, theta):
    pool = ai.AgentPool(cfg=cfg, rng=rng, policy=theta)
    return lambda game, me, hand, others, _rng: ai.turn(game, me.id, pool)


def _sample(rng, mixture):
    total = sum(w for _, w in mixture)
    r, acc = rng.random() * total, 0.0
    for th, w in mixture:
        acc += w
        if r < acc:
            return th
    return mixture[-1][0]


def eval_block(args: tuple) -> dict[str, Any]:
    focal_theta, mixture, n_games, seed = args
    cfg = DEFAULT_CONFIG
    rng = random.Random(seed)
    share = 0.0
    by_origin: Counter = Counter()
    seats_by_origin: Counter = Counter()
    sq = 0.0
    for _ in range(n_games):
        seat = rng.randrange(N)
        origins = list(cfg.origin_ids())
        rng.shuffle(origins)
        assign = [_seat_from(cfg, rng, _sample(rng, mixture)) for _ in range(N)]
        assign[seat] = _seat_from(cfg, rng, focal_theta)
        rec = analysis.play(N, rng, cfg, assign, origins=origins[:N])
        pid = sorted(rec.cards)[seat]
        w = 1.0 / len(rec.winners) if pid in rec.winners else 0.0
        share += w
        sq += w * w
        oid = rec.origin_of.get(pid)
        by_origin[oid] += w
        seats_by_origin[oid] += 1
    return {"n": n_games, "share": share, "sq": sq, "by_origin": dict(by_origin),
            "seats_by_origin": dict(seats_by_origin)}


def evaluate_one(theta, mixture, games: int, workers: int, seed: int, ex=None) -> dict[str, Any]:
    """这个打法对混合的胜率（± 标准误）和按身份的胜率。"""
    chunks = max(1, min(workers * 2, games // 25 or 1))
    per = [games // chunks + (1 if i < games % chunks else 0) for i in range(chunks)]
    jobs = [(theta, mixture, n, seed * 1000 + i) for i, n in enumerate(per) if n]
    own = ex is None
    ex = ex or ProcessPoolExecutor(max_workers=workers)
    try:
        res = list(ex.map(eval_block, jobs))
    finally:
        if own:
            ex.shutdown()
    n = sum(r["n"] for r in res)
    s = sum(r["share"] for r in res)
    sq = sum(r["sq"] for r in res)
    m = s / n
    var = max(0.0, sq / n - m * m)
    by, seats = Counter(), Counter()
    for r in res:
        by.update(r["by_origin"])
        seats.update(r["seats_by_origin"])
    return {"u": m, "se": math.sqrt(var / n), "n": n,
            "by_origin": {o: round(by[o] / seats[o], 4) for o in seats}}


# --------------------------------------------------------------------------
# 打法池
# --------------------------------------------------------------------------

def load_meta(pop: Path) -> dict[str, Any]:
    return json.loads((pop / "meta.json").read_text(encoding="utf-8"))


def save_meta(pop: Path, meta: dict[str, Any]) -> None:
    tmp = pop / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(pop / "meta.json")


def thetas(pop: Path, meta: dict[str, Any]) -> list[dict[str, float]]:
    return [ai.load_policy(str(pop / p["path"])) for p in meta["policies"]]


def mixture_of(pop: Path, meta: dict[str, Any]) -> list[tuple[dict[str, float], float]]:
    return [(th, w) for th, w in zip(thetas(pop, meta), meta["weights"]) if w > 0]


def write_mixture_file(path: Path, pop: Path, meta: dict[str, Any], note: str = "") -> None:
    members = [{"name": p["name"], "weight": round(w, 6), "theta": th}
               for p, th, w in zip(meta["policies"], thetas(pop, meta), meta["weights"]) if w > 0]
    path.write_text(json.dumps({"mixture": members, "meta": {"note": note, "psro_iter": meta["iter"]}},
                               ensure_ascii=False, indent=1), encoding="utf-8")


def init_pop(pop: Path, start: Path) -> dict[str, Any]:
    pop.mkdir(parents=True, exist_ok=True)
    shutil.copy(start, pop / "p0.json")
    meta = {"policies": [{"name": "p0", "path": "p0.json", "from": str(start)}], "weights": [1.0],
            "iter": 0, "stage": "br", "history": []}
    save_meta(pop, meta)
    return meta


def log(pop: Path, msg: str) -> None:
    line = time.strftime("%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(pop / "progress.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(args) -> None:
    pop = Path(args.pop)
    if args.resume and (pop / "meta.json").exists():
        meta = load_meta(pop)
    else:
        if (pop / "meta.json").exists() and not args.fresh:
            raise SystemExit(f"{pop} 已经有打法池了：接着跑用 --resume，重新开始用 --fresh")
        if args.fresh and pop.exists():
            shutil.rmtree(pop)
        meta = init_pop(pop, Path(args.start))
    log(pop, f"开始：第 {meta['iter']} 轮、阶段 {meta['stage']}，打法池 {len(meta['policies'])} 个")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        while meta["iter"] < args.iters:
            k = meta["iter"]
            name = f"p{len(meta['policies'])}" if meta["stage"] == "br" else meta.get("pending", {}).get("name")
            if meta["stage"] == "br":
                mix_path = pop / f"mixture_iter{k}.json"
                write_mixture_file(mix_path, pop, meta, f"第 {k} 轮训练最佳应对用的混合")
                best_i = max(range(len(meta["weights"])), key=lambda i: meta["weights"][i])
                init = pop / meta["policies"][best_i]["path"]
                out = pop / f"{name}.json"
                log(pop, f"第 {k} 轮：训练最佳应对 {name}（从 {init.name} 出发，{args.br_minutes} 分钟）")
                cmd = [sys.executable, str(ROOT / "learn.py"), "--br-vs", str(mix_path), "--init", str(init),
                       "--minutes", str(args.br_minutes), "--workers", str(args.workers), "--out", str(out)]
                with open(pop / f"learn_{name}.log", "w", encoding="utf-8") as fh:
                    subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
                meta["pending"] = {"name": name, "path": f"{name}.json"}
                meta["stage"] = "exploit"
                save_meta(pop, meta)
            if meta["stage"] == "exploit":
                br = ai.load_policy(str(pop / meta["pending"]["path"]))
                mixture = mixture_of(pop, meta)
                res = evaluate_one(br, mixture, args.exploit_games, args.workers, seed=90000 + k, ex=ex)
                gain = res["u"] - 1.0 / N
                meta["history"].append({"iter": k, "br": meta["pending"]["name"],
                                        "exploitability": round(gain, 4), "se": round(res["se"], 4),
                                        "br_by_origin": res["by_origin"],
                                        "weights_before": [round(w, 4) for w in meta["weights"]]})
                log(pop, f"第 {k} 轮：最佳应对 {meta['pending']['name']} 对混合 {100 * res['u']:.1f}%，"
                         f"可被剥削度 {100 * gain:+.1f} ±{196 * res['se']:.1f} 点；按身份 "
                         + "、".join(f"{o} {100 * v:.1f}" for o, v in sorted(res["by_origin"].items())))
                meta["stage"] = "solve"
                save_meta(pop, meta)
            if meta["stage"] == "solve":
                if meta["pending"]["name"] not in [p["name"] for p in meta["policies"]]:
                    meta["policies"].append(dict(meta["pending"]))
                    # 新打法先给一份起始比例，其余按原比例缩
                    start = 1.0 / len(meta["policies"])
                    meta["weights"] = [w * (1 - start) for w in meta["weights"]] + [start]
                ths = thetas(pop, meta)

                def payoff(sigma: list[float], _t=[0]) -> list[float]:
                    mix = [(th, s) for th, s in zip(ths, sigma) if s > 1e-4]
                    u = [evaluate_one(th, mix, args.games, args.workers, seed=50000 + k * 1000 + _t[0] * 50 + i,
                                      ex=ex)["u"] for i, th in enumerate(ths)]
                    _t[0] += 1
                    return u

                avg, trace = replicator(payoff, meta["weights"], args.solve_rounds, args.eta)
                meta["weights"] = [round(x, 6) for x in avg]
                meta["history"][-1]["solve"] = trace
                meta["history"][-1]["weights_after"] = meta["weights"]
                log(pop, f"第 {k} 轮：新的混合比例 " + "、".join(
                    f"{p['name']} {100 * w:.0f}%" for p, w in zip(meta["policies"], meta["weights"])))
                meta["iter"] = k + 1
                meta["stage"] = "br"
                meta.pop("pending", None)
                save_meta(pop, meta)
    log(pop, "完成。")
    print(status_text(pop))


def status_text(pop: Path) -> str:
    meta = load_meta(pop)
    lines = [f"打法池 {len(meta['policies'])} 个，已完成 {meta['iter']} 轮（当前阶段 {meta['stage']}）", "",
             "| 轮 | 最佳应对 | 可被剥削度（点） | 求解后的混合比例 |", "|---|---|---|---|"]
    names = [p["name"] for p in meta["policies"]]
    for h in meta["history"]:
        w = h.get("weights_after")
        mix = "、".join(f"{n} {100 * x:.0f}%" for n, x in zip(names, w)) if w else "—"
        lines.append(f"| {h['iter']} | {h['br']} | {100 * h['exploitability']:+.1f} ±{196 * h['se']:.1f} | {mix} |")
    return "\n".join(lines)


def freeze(args) -> None:
    pop = Path(args.pop)
    meta = load_meta(pop)
    best = ROOT / "policies" / "best.json"
    backup = ROOT / "policies" / f"best_pre_psro_{time.strftime('%Y%m%d%H%M%S')}.json"
    shutil.copy(best, backup)
    write_mixture_file(best, pop, meta, f"PSRO 均衡混合（{meta['iter']} 轮，打法池 {len(meta['policies'])} 个），"
                                        f"平衡评估的参照 AI，调规则期间冻结")
    print(f"写到 {best}（旧的备份在 {backup.name}）")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pop", default=str(POP))
    r.add_argument("--start", default=str(ROOT / "policies" / "best.json"))
    r.add_argument("--iters", type=int, default=4)
    r.add_argument("--br-minutes", type=float, default=90)
    r.add_argument("--solve-rounds", type=int, default=8)
    r.add_argument("--games", type=int, default=1500, help="求解时每一步每个打法估几局")
    r.add_argument("--exploit-games", type=int, default=3000, help="测可被剥削度用几局")
    r.add_argument("--eta", type=float, default=25.0, help="复制者动态的步长（胜率差 0.02 -> 比例乘 e^0.5）")
    r.add_argument("--workers", type=int, default=10)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--fresh", action="store_true")
    s = sub.add_parser("status")
    s.add_argument("--pop", default=str(POP))
    f = sub.add_parser("freeze")
    f.add_argument("--pop", default=str(POP))
    args = ap.parse_args(argv)
    if args.cmd == "run":
        run(args)
    elif args.cmd == "status":
        print(status_text(Path(args.pop)))
    else:
        freeze(args)


if __name__ == "__main__":
    main()
