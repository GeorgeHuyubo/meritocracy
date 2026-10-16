"""按身份的 PSRO：每个身份各有自己的打法池和混合比例，逼近"每个身份都按自己技能打、谁也剥削不了谁"的均衡。

psro.py 求的是"六个身份共用一个混合"的均衡：池子里没有哪一整套打法能多赢，但不保证每个身份都打到最好。
专项打法（learn.py --focus-origin）是针对某一群固定对手的最佳应对，本身会被剥削——六个座位都换成
各自的专项打法，结果完全偏离（贫农 34.6%、会计 3.1%）。这里把两者结合起来：

每一轮：
  1. 给六个身份各训练一个最佳应对，对手 = 其他身份当前各自的混合（learn.py --focus-origin --opponents 按身份的混合）
  2. 测每个身份的可被剥削度 = 新打法 − 这个身份当前的混合，坐同一个身份、同一批对手
  3. 六个身份同时用复制者动态重新求各自的混合比例（每一步按当前的按身份混合重新估胜率，取平均）

    python3 psro_origin.py run --iters 3 --br-minutes 40 --workers 10
    python3 psro_origin.py run --resume
    python3 psro_origin.py status
    python3 psro_origin.py freeze      # 写成 policies/best.json（{"profile": {出身: 混合}}，旧的备份）

起点：policies/best.json（psro.py 的均衡混合）里比例 ≥5% 的打法，六个身份共用；
第 0 轮的最佳应对直接用 policies/focus_eq_<出身>.json（针对那个混合训练的专项打法），不用重练。
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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import ai
import analysis
import psro
from config import DEFAULT_CONFIG

ROOT = Path(__file__).resolve().parent
POP = ROOT / "policies" / "pop_origin"
N = 6


def origins() -> list[str]:
    return list(DEFAULT_CONFIG.origin_ids())


# --------------------------------------------------------------------------
# 复制者动态（每个身份一份比例，同时更新）
# --------------------------------------------------------------------------

def replicator_joint(payoff_fn: Callable[[dict[str, list[float]]], dict[str, list[float]]],
                     sigma0: dict[str, list[float]], rounds: int, eta: float
                     ) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
    sig = {o: [max(1e-9, x) / sum(max(1e-9, y) for y in s) for x in s] for o, s in sigma0.items()}
    avg = {o: [0.0] * len(s) for o, s in sig.items()}
    trace = []
    for t in range(rounds):
        u = payoff_fn(sig)
        rec = {"round": t, "sigma": {}, "u": {}}
        nxt = {}
        for o, s in sig.items():
            v = sum(a * b for a, b in zip(s, u[o]))
            raw = [a * math.exp(eta * (x - v)) for a, x in zip(s, u[o])]
            z = sum(raw)
            nxt[o] = [x / z for x in raw]
            rec["sigma"][o] = [round(x, 4) for x in s]
            rec["u"][o] = [round(x, 4) for x in u[o]]
        sig = nxt
        for o, s in sig.items():
            avg[o] = [a + b for a, b in zip(avg[o], s)]
        trace.append(rec)
    return {o: [x / rounds for x in a] for o, a in avg.items()}, trace


# --------------------------------------------------------------------------
# 模拟：某个身份坐 1 席（指定打法，或者按这个身份的混合抽），其余座位按各自身份的混合抽
# --------------------------------------------------------------------------

def focal_block(args: tuple) -> dict[str, float]:
    origin, focal_theta, profile, n_games, seed = args
    cfg = DEFAULT_CONFIG
    rng = random.Random(seed)
    share = sq = 0.0
    for _ in range(n_games):
        others = [o for o in cfg.origin_ids() if o != origin]
        rng.shuffle(others)
        seat = rng.randrange(N)
        seats_o = others[:seat] + [origin] + others[seat:]
        assign = [psro._seat_from(cfg, rng, psro._sample(rng, profile[o])) for o in seats_o]
        if focal_theta is not None:
            assign[seat] = psro._seat_from(cfg, rng, focal_theta)
        rec = analysis.play(N, rng, cfg, assign, origins=seats_o)
        pid = sorted(rec.cards)[seat]
        w = 1.0 / len(rec.winners) if pid in rec.winners else 0.0
        share += w
        sq += w * w
    return {"n": n_games, "share": share, "sq": sq}


def evaluate_focal(origin: str, theta, profile, games: int, workers: int, seed: int, ex) -> dict[str, float]:
    chunks = max(1, min(workers * 2, games // 25 or 1))
    per = [games // chunks + (1 if i < games % chunks else 0) for i in range(chunks)]
    res = list(ex.map(focal_block, [(origin, theta, profile, n, seed * 1000 + i) for i, n in enumerate(per) if n]))
    n = sum(r["n"] for r in res)
    m = sum(r["share"] for r in res) / n
    var = max(0.0, sum(r["sq"] for r in res) / n - m * m)
    return {"u": m, "se": math.sqrt(var / n), "n": n}


# --------------------------------------------------------------------------
# 状态
# --------------------------------------------------------------------------

_THETA_CACHE: dict[str, dict[str, float]] = {}


def theta_at(pop: Path, path: str) -> dict[str, float]:
    key = str(pop / path)
    if key not in _THETA_CACHE:
        _THETA_CACHE[key] = ai.load_policy(key)
    return _THETA_CACHE[key]


def profile_of(pop: Path, meta: dict[str, Any], weights: dict[str, list[float]] | None = None):
    weights = weights or meta["weights"]
    return {o: [(theta_at(pop, p["path"]), w) for p, w in zip(meta["pools"][o], weights[o]) if w > 1e-4]
            for o in origins()}


def write_profile_file(path: Path, pop: Path, meta: dict[str, Any], note: str = "") -> None:
    prof = {o: [{"name": p["name"], "weight": round(w, 6), "theta": theta_at(pop, p["path"])}
                for p, w in zip(meta["pools"][o], meta["weights"][o]) if w > 1e-4] for o in origins()}
    path.write_text(json.dumps({"profile": prof, "meta": {"note": note, "iter": meta["iter"]}},
                               ensure_ascii=False), encoding="utf-8")


def save_meta(pop: Path, meta: dict[str, Any]) -> None:
    tmp = pop / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(pop / "meta.json")


def load_meta(pop: Path) -> dict[str, Any]:
    return json.loads((pop / "meta.json").read_text(encoding="utf-8"))


def init_pop(pop: Path, start: Path, seed_brs: str, min_weight: float) -> dict[str, Any]:
    pop.mkdir(parents=True, exist_ok=True)
    d = json.loads(start.read_text(encoding="utf-8"))
    members = d.get("mixture") or [{"name": "p0", "weight": 1.0, "theta": d["theta"]}]
    keep = [m for m in members if m["weight"] >= min_weight] or members
    z = sum(m["weight"] for m in keep)
    shared = []
    for m in keep:
        path = f"shared_{m['name']}.json"
        (pop / path).write_text(json.dumps({"theta": m["theta"]}, ensure_ascii=False), encoding="utf-8")
        shared.append({"name": m["name"], "path": path})
    meta = {"pools": {o: [dict(s) for s in shared] for o in origins()},
            "weights": {o: [m["weight"] / z for m in keep] for o in origins()},
            "iter": 0, "stage": "br", "history": []}
    if seed_brs:
        pending = {}
        for o in origins():
            src = Path(seed_brs.format(origin=o))
            if not src.exists():
                pending = {}
                break
            shutil.copy(src, pop / f"{o}_br0.json")
            pending[o] = {"name": f"{o}_br0", "path": f"{o}_br0.json"}
        if pending:
            meta["pending"] = pending
            meta["stage"] = "exploit"
    save_meta(pop, meta)
    return meta


def log(pop: Path, msg: str) -> None:
    line = time.strftime("%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(pop / "progress.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# --------------------------------------------------------------------------
# 主循环
# --------------------------------------------------------------------------

def run(args) -> None:
    pop = Path(args.pop)
    if args.resume and (pop / "meta.json").exists():
        meta = load_meta(pop)
    else:
        if (pop / "meta.json").exists() and not args.fresh:
            raise SystemExit(f"{pop} 已经有打法池了：接着跑用 --resume，重新开始用 --fresh")
        if args.fresh and pop.exists():
            shutil.rmtree(pop)
        meta = init_pop(pop, Path(args.start), args.seed_brs, args.min_weight)
    log(pop, f"开始：第 {meta['iter']} 轮、阶段 {meta['stage']}")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        while meta["iter"] < args.iters:
            k = meta["iter"]
            if meta["stage"] == "br":
                prof_path = pop / f"profile_iter{k}.json"
                write_profile_file(prof_path, pop, meta, f"第 {k} 轮训练最佳应对用的按身份混合")
                meta.setdefault("pending", {})
                for o in origins():
                    if o in meta["pending"]:
                        continue
                    pool, w = meta["pools"][o], meta["weights"][o]
                    init = pop / pool[max(range(len(w)), key=lambda i: w[i])]["path"]
                    name = f"{o}_br{k}"
                    log(pop, f"第 {k} 轮：训练 {name}（{args.br_minutes} 分钟）")
                    cmd = [sys.executable, str(ROOT / "learn.py"), "--focus-origin", o, "--opponents", str(prof_path),
                           "--init", str(init), "--minutes", str(args.br_minutes), "--workers", str(args.workers),
                           "--out", str(pop / f"{name}.json")]
                    with open(pop / f"learn_{name}.log", "w", encoding="utf-8") as fh:
                        subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
                    meta["pending"][o] = {"name": name, "path": f"{name}.json"}
                    save_meta(pop, meta)
                meta["stage"] = "exploit"
                save_meta(pop, meta)
            if meta["stage"] == "exploit":
                prof = profile_of(pop, meta)
                gains = {}
                for i, o in enumerate(origins()):
                    br = theta_at(pop, meta["pending"][o]["path"])
                    r_br = evaluate_focal(o, br, prof, args.exploit_games, args.workers, 91000 + k * 10 + i, ex)
                    r_cur = evaluate_focal(o, None, prof, args.exploit_games, args.workers, 91000 + k * 10 + i, ex)
                    gains[o] = {"br": round(r_br["u"], 4), "cur": round(r_cur["u"], 4),
                                "gain": round(r_br["u"] - r_cur["u"], 4),
                                "se": round(math.sqrt(r_br["se"] ** 2 + r_cur["se"] ** 2), 4)}
                    log(pop, f"第 {k} 轮：{o} 当前混合 {100 * r_cur['u']:.1f}% -> 最佳应对 {100 * r_br['u']:.1f}%"
                             f"（可被剥削 {100 * gains[o]['gain']:+.1f} ±{196 * gains[o]['se']:.1f}）")
                meta["history"].append({"iter": k, "gains": gains,
                                        "weights_before": {o: [round(x, 4) for x in w]
                                                           for o, w in meta["weights"].items()}})
                meta["stage"] = "solve"
                save_meta(pop, meta)
            if meta["stage"] == "solve":
                for o in origins():
                    pend = meta["pending"][o]
                    if pend["name"] not in [p["name"] for p in meta["pools"][o]]:
                        meta["pools"][o].append(dict(pend))
                        start = 1.0 / len(meta["pools"][o])
                        meta["weights"][o] = [w * (1 - start) for w in meta["weights"][o]] + [start]

                def payoff(sig: dict[str, list[float]], _t=[0]) -> dict[str, list[float]]:
                    prof = profile_of(pop, meta, sig)
                    out = {}
                    for oi, o in enumerate(origins()):
                        out[o] = [evaluate_focal(o, theta_at(pop, p["path"]), prof, args.games, args.workers,
                                                 60000 + k * 1000 + _t[0] * 50 + oi * 8 + j, ex)["u"]
                                  for j, p in enumerate(meta["pools"][o])]
                    _t[0] += 1
                    return out

                avg, trace = replicator_joint(payoff, meta["weights"], args.solve_rounds, args.eta)
                meta["weights"] = {o: [round(x, 6) for x in a] for o, a in avg.items()}
                meta["history"][-1]["solve"] = trace
                meta["history"][-1]["weights_after"] = meta["weights"]
                for o in origins():
                    log(pop, f"第 {k} 轮：{o} 新的混合 " + "、".join(
                        f"{p['name']} {100 * w:.0f}%" for p, w in zip(meta["pools"][o], meta["weights"][o])))
                meta["iter"] = k + 1
                meta["stage"] = "br"
                meta.pop("pending", None)
                save_meta(pop, meta)
    log(pop, "完成。")
    print(status_text(pop))


def status_text(pop: Path) -> str:
    meta = load_meta(pop)
    lines = [f"按身份 PSRO：已完成 {meta['iter']} 轮（当前阶段 {meta['stage']}）", "",
             "可被剥削度（新的最佳应对 − 这个身份当前的混合，百分点）：", "",
             "| 轮 | " + " | ".join(origins()) + " |", "|---" * (len(origins()) + 1) + "|"]
    for h in meta["history"]:
        lines.append(f"| {h['iter']} | " + " | ".join(
            f"{100 * h['gains'][o]['gain']:+.1f}" for o in origins()) + " |")
    lines += ["", "当前各身份的混合："]
    for o in origins():
        lines.append(f"- {o}：" + "、".join(f"{p['name']} {100 * w:.0f}%"
                                           for p, w in zip(meta["pools"][o], meta["weights"][o])))
    return "\n".join(lines)


def freeze(args) -> None:
    pop = Path(args.pop)
    meta = load_meta(pop)
    best = ROOT / "policies" / "best.json"
    backup = ROOT / "policies" / f"best_pre_psro_origin_{time.strftime('%Y%m%d%H%M%S')}.json"
    shutil.copy(best, backup)
    write_profile_file(best, pop, meta, f"按身份 PSRO 均衡（{meta['iter']} 轮），平衡评估的参照 AI，调规则期间冻结")
    print(f"写到 {best}（旧的备份在 {backup.name}）")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pop", default=str(POP))
    r.add_argument("--start", default=str(ROOT / "policies" / "best.json"))
    r.add_argument("--seed-brs", default=str(ROOT / "policies" / "focus_eq_{origin}.json"),
                   help="第 0 轮直接用的最佳应对（{origin} 换成出身）；空 = 第 0 轮也重新训练")
    r.add_argument("--min-weight", type=float, default=0.05, help="起点只保留比例不低于这个的打法")
    r.add_argument("--iters", type=int, default=3)
    r.add_argument("--br-minutes", type=float, default=40)
    r.add_argument("--solve-rounds", type=int, default=8)
    r.add_argument("--games", type=int, default=1000, help="求解时每一步每个（身份, 打法）估几局")
    r.add_argument("--exploit-games", type=int, default=3000)
    r.add_argument("--eta", type=float, default=25.0)
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
