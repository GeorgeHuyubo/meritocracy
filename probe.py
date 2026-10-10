"""平衡探针：一个被测座位（搜索型 AI / 学习型 / 手写 / 指定权重）+ 5 个陪练，按身份统计。

焦点座位设计见 telemetry.py：第 g 局被测身份 = ORIGINS[g % 6]、座位轮换；同一个 g 下
各个"打法"（arm）的身份、座位、每轮手牌和事件都一样，所以可以逐局配对相减：

    搜索提升 Δ = 搜索 AI 坐这个座位的胜率 − 学习型 AI 坐同一个座位的胜率

某个身份的 Δ 特别大 = 普通 AI 把这个身份打亏了，Python AI 关于它的平衡结论不可信。

    python3 probe.py --smoke                                   # 12 局小样本，看耗时和格式
    python3 probe.py --aa --games 60                            # 预算 0：必须和学习型逐位相同
    python3 probe.py --crowd learned --games 2400 --budget 96 --workers 10 --time-limit-hours 3
    python3 probe.py --crowd smart --games 900 --budget 64 --workers 10
    python3 probe.py --resume search_runs/<目录>                # 接着跑（按 g 跳过已完成的）
    python3 probe.py --summarize search_runs/<目录>             # 只重出汇总
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import ai
import balance_stats as bs
import search_ai
import telemetry as T
from config import DEFAULT_CONFIG, Config

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "search_runs"


def arm_pool(cfg: Config, arm: str, rng: random.Random) -> ai.AgentPool:
    """learned = cfg.ai_policy（policies/best.json）；smart = 手写；policy:<路径> = 指定权重。"""
    if arm == "smart":
        return ai.make_pool(replace(cfg, ai_policy=""), rng)
    if arm.startswith("policy:"):
        return ai.make_pool(replace(cfg, ai_policy=arm.split(":", 1)[1]), rng)
    return ai.make_pool(cfg, rng)


def play_probe_game(cfg: Config, g: int, seed: int, arm: str, crowd: str,
                    scfg: search_ai.SearchConfig, residuals: Any = None,
                    want_telemetry: bool = False, crowd_tag: str = "") -> dict[str, Any]:
    focal_origin, focal_seat, seats = T.focal_assignment(g, seed, cfg)
    gs = T.game_seed(seed, g)
    game = T.new_game(cfg, seats, gs, game_id=f"probe{g}")
    fpid = focal_seat + 1
    crowd_pool = arm_pool(cfg, crowd, random.Random(f"{gs}/crowd{crowd_tag}"))
    deciders: dict[int, Any] = {p: (lambda gm, pid: ai.turn(gm, pid, crowd_pool)) for p in game.players}
    searcher = None
    if arm.startswith("search"):
        # search-rb：换不换牌照普通 AI 的规则，只搜出牌（拆分"提升来自换牌时机还是出牌"）
        # search-rbn：同上但每个决策都搜（排除"搜得少"这个混杂因素）
        sc = (replace(scfg, redraw_mode="base") if arm == "search-rb" else
              replace(scfg, redraw_mode="base", skip_top_prob=1.1) if arm == "search-rbn" else scfg)
        searcher = search_ai.Searcher(fpid, cfg, sc, base_rng_seed=gs, residuals=residuals)
        deciders[fpid] = searcher.turn
    else:
        # 和 Searcher.base_pool 同一个种子：预算 0 的搜索 AI 和 learned 逐位相同（A/A 检查）
        focal_pool = arm_pool(cfg, arm, random.Random(f"{gs}/base"))
        deciders[fpid] = lambda gm, pid: ai.turn(gm, pid, focal_pool)
    controller = {p: crowd for p in game.players}
    controller[fpid] = arm
    t0 = time.process_time()
    tel = T.drive_game(game, deciders, gs, T.focal_order(len(game.players), fpid), controller)
    cpu = time.process_time() - t0
    summ = T.game_summary(game, controller)
    cards: Counter = Counter()
    for r in tel:
        for p in r["players"]:
            if p["pid"] == fpid:
                cards.update(p["cards"])
    out = {
        "g": g, "seed": seed, "arm": arm, "crowd": crowd,
        "focal_origin": focal_origin, "focal_seat": focal_seat, "focal_pid": fpid,
        "origins": seats, "winners": summ["winners"],
        "focal_share": (1.0 / len(summ["winners"])) if fpid in summ["winners"] else 0.0,
        "place": summ["final_standing"].index(fpid) + 1,
        "rounds": summ["rounds"], "end": summ["end"], "cpu_s": round(cpu, 2),
        "focal_cards": dict(cards),
        "search_stats": searcher.stats() if searcher else None,
    }
    if want_telemetry:
        out["telemetry"] = tel
        out["summary"] = summ
    return out


def variant_cfg(variant: str) -> Config:
    """sweep.py 的规则变体（陪练和被测座位都用这个变体续训出来的 AI，有的话）。"""
    if not variant or variant == "base":
        return DEFAULT_CONFIG
    import sweep

    return sweep.build(variant, own_ai=True)


def run_block(params: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = variant_cfg(params.get("variant", ""))
    scfg = search_ai.SearchConfig(**params["scfg"])
    residuals = search_ai.load_residuals(Path(params["residuals"])) if params.get("residuals") else None
    out = []
    for g in params["games"]:
        for arm in params["arms"]:
            if (g, arm) in params["done"]:
                continue
            out.append(play_probe_game(cfg, g, params["seed"], arm, params["crowd"], scfg,
                                       residuals, params.get("telemetry", False)))
    return out


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------

def load_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for f in sorted(run_dir.glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


def summarize(run_dir: Path, base_arm: str = "learned") -> dict[str, Any]:
    rows = load_rows(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8")) if (run_dir / "meta.json").exists() else {}
    by_arm: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        by_arm[r["arm"]][r["g"]] = r
    origins = list(DEFAULT_CONFIG.origin_ids())
    out: dict[str, Any] = {"meta": meta, "arms": {}, "uplift": {}}
    for arm, games in by_arm.items():
        per = {}
        for oid in origins + ["ALL"]:
            sel = [r for r in games.values() if oid == "ALL" or r["focal_origin"] == oid]
            m, se = bs.mean_se([r["focal_share"] for r in sel])
            pm, pse = bs.mean_se([r["place"] for r in sel])
            cards = Counter()
            for r in sel:
                cards.update(r.get("focal_cards") or {})
            tot = sum(cards.values()) or 1
            per[oid] = {"n": len(sel), "mean": m, "se": se, "placement": pm, "placement_se": pse,
                        "card_mix": {k: round(v / tot, 3) for k, v in sorted(cards.items())}}
        st = [r["search_stats"] for r in games.values() if r.get("search_stats")]
        if st:
            per["ALL"]["search"] = {
                k: round(sum(s[k] for s in st) / len(st), 3)
                for k in ("decisions", "searched", "deviated", "redraws_chosen", "mean_gain")}
        per["ALL"]["cpu_s"] = round(sum(r["cpu_s"] for r in games.values()) / max(1, len(games)), 2)
        out["arms"][arm] = per
    if base_arm in by_arm:
        for arm, games in by_arm.items():
            if arm == base_arm:
                continue
            up = {}
            for oid in origins + ["ALL"]:
                gs = [g for g, r in games.items() if g in by_arm[base_arm]
                      and (oid == "ALL" or r["focal_origin"] == oid)]
                a = [games[g]["focal_share"] for g in gs]
                b = [by_arm[base_arm][g]["focal_share"] for g in gs]
                m, se = bs.paired_diff(a, b) if gs else (float("nan"), float("nan"))
                pa = [games[g]["place"] for g in gs]
                pb = [by_arm[base_arm][g]["place"] for g in gs]
                pm, pse = bs.paired_diff(pa, pb) if gs else (float("nan"), float("nan"))
                up[oid] = {"n": len(gs), "delta": m, "se": se, "place_delta": pm, "place_se": pse}
            out["uplift"][f"{arm}-{base_arm}"] = up
    return out


def render(summary: dict[str, Any]) -> str:
    cfg = DEFAULT_CONFIG
    name = {o["id"]: o["name"] for o in cfg.origin_definitions}
    origins = list(cfg.origin_ids())
    meta = summary.get("meta", {})
    lines = [f"# 平衡探针（陪练 = {meta.get('crowd', '?')}，预算 {meta.get('scfg', {}).get('budget', '?')}，"
             f"种子 {meta.get('seed', '?')}）", ""]
    arms = list(summary["arms"])
    lines.append("| 身份 | " + " | ".join(f"{a} 胜率" for a in arms) + " | " +
                 " | ".join(f"{k} 提升" for k in summary["uplift"]) + " |")
    lines.append("|---" * (1 + len(arms) + len(summary["uplift"])) + "|")
    for oid in origins + ["ALL"]:
        cells = [bs.fmt_pct(summary["arms"][a][oid]["mean"], summary["arms"][a][oid]["se"])
                 + f"（{summary['arms'][a][oid]['n']}）" for a in arms]
        ups = []
        for k, up in summary["uplift"].items():
            u = up[oid]
            mark = {1: " ↑", -1: " ↓", 0: ""}[bs.significant(u["delta"], u["se"])]
            ups.append(f"{100 * u['delta']:+.1f} ±{100 * bs.Z * u['se']:.1f}{mark}"
                       if not math.isnan(u["delta"]) else "—")
        lines.append(f"| {name.get(oid, '合计')} | " + " | ".join(cells) + " | " + " | ".join(ups) + " |")
    lines.append("")
    lines.append("公平线 16.67%。提升 = 同一批对局里该打法减学习型 AI 的胜率（逐局配对），↑/↓ = 95% 区间不含 0。")
    for a in arms:
        s = summary["arms"][a]["ALL"].get("search")
        if s:
            lines.append(f"- {a}：每局决策 {s['decisions']}、搜索 {s['searched']}、改选 {s['deviated']}、"
                         f"选了换牌 {s['redraws_chosen']}，平均每局 {summary['arms'][a]['ALL']['cpu_s']} CPU 秒")
    lines.append("")
    lines.append("出牌结构（被测座位）：")
    for oid in origins:
        mixes = "；".join(f"{a}: " + "、".join(f"{k} {100 * v:.0f}%" for k, v in
                                                summary["arms"][a][oid]["card_mix"].items() if v >= 0.03)
                         for a in arms)
        lines.append(f"- {name[oid]}：{mixes}")
    return "\n".join(lines)


def emit_sources(summary: dict[str, Any], run_dir: Path) -> list[Path]:
    """给 scorecard.py 的来源文件：每个打法一份（焦点设计）。"""
    meta = summary.get("meta", {})
    rows = load_rows(run_dir)
    paths = []
    for arm in summary["arms"]:
        fam = {"learned": "py_learned", "smart": "py_smart"}.get(
            arm, "search" if arm.startswith("search") else "py_learned" if arm.startswith("policy:") else arm)
        src = bs.origin_source(
            f"{arm}_vs_{meta.get('crowd', '?')}", fam, "focal", meta.get("crowd"),
            meta.get("rules_fp"),
            [(r["focal_origin"], r["focal_share"], r["place"]) for r in rows if r["arm"] == arm],
            notes=f"probe {run_dir.name}", n_games=sum(1 for r in rows if r["arm"] == arm))
        p = run_dir / f"source_{arm.replace(':', '_').replace('/', '_')}.json"
        p.write_text(json.dumps(src, ensure_ascii=False, indent=1), encoding="utf-8")
        paths.append(p)
    return paths


def write_summary(run_dir: Path) -> str:
    s = summarize(run_dir)
    (run_dir / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    md = render(s)
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    emit_sources(s, run_dir)
    return md


# --------------------------------------------------------------------------
# 主程序
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crowd", default="learned", help="陪练：learned | smart | policy:<路径>")
    ap.add_argument("--arms", default="search,learned", help="被测座位的几种打法，逗号分隔")
    ap.add_argument("--games", type=int, default=None, help="默认 600；--resume 时默认沿用原来的局数")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--budget", type=int, default=96)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--money-model", default="residual", choices=["residual", "lognormal"])
    ap.add_argument("--residuals", default=str(search_ai.RESIDUALS_PATH))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 0))
    ap.add_argument("--variant", default="", help="sweep.py 的规则变体（如 redfee100）")
    ap.add_argument("--only-origin", default="", help="只跑被测身份是这个的局（RICH 等），局号仍按 g 分层")
    ap.add_argument("--block", type=int, default=6, help="每个任务打几局（每局里各打法都跑）")
    ap.add_argument("--time-limit-hours", type=float, default=0.0)
    ap.add_argument("--telemetry", action="store_true", help="每局存完整遥测（文件会大很多）")
    ap.add_argument("--out", default="")
    ap.add_argument("--resume", default="")
    ap.add_argument("--summarize", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--aa", action="store_true", help="A/A 检查：预算 0，搜索 AI 必须和学习型逐位相同")
    args = ap.parse_args(argv)

    if args.summarize:
        print(write_summary(Path(args.summarize)))
        return
    if not args.resume:
        args.games = args.games or 600
    if args.smoke:
        args.games, args.budget, args.k, args.workers = 12, 8, 2, min(args.workers, 4)
    if args.aa:
        args.budget = 0
    residuals = args.residuals if args.money_model == "residual" and Path(args.residuals).exists() else ""
    if args.money_model == "residual" and not residuals and args.budget > 0:
        print(f"注意：残差表 {args.residuals} 不存在，退回对数正态（先跑 search_ai.py --calibrate-money 2000）")
        args.money_model = "lognormal"
    scfg = asdict(search_ai.SearchConfig(budget=args.budget, k=args.k, margin=args.margin,
                                         money_model=args.money_model, seed=args.seed))
    arms = [a for a in args.arms.split(",") if a]

    if args.resume:
        run_dir = Path(args.resume)
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        args.crowd, arms, args.seed, scfg = meta["crowd"], meta["arms"], meta["seed"], meta["scfg"]
        args.games = args.games or meta["games"]
        args.only_origin = args.only_origin or meta.get("only_origin", "")
        args.variant = args.variant or meta.get("variant", "")
        residuals = meta.get("residuals", "")
    else:
        tag = "smoke" if args.smoke else ("aa" if args.aa else f"{args.crowd.replace(':', '_').replace('/', '_')}")
        if args.variant:
            tag += f"-{args.variant.replace('+', '_')}"
        run_dir = Path(args.out) if args.out else RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}-{tag}"
        run_dir.mkdir(parents=True, exist_ok=True)
    vcfg = variant_cfg(args.variant)
    meta = {"crowd": args.crowd, "arms": arms, "seed": args.seed, "games": args.games, "scfg": scfg,
            "residuals": residuals, "rules_fp": T.rules_fingerprint(vcfg), "variant": args.variant,
            "ai_policy": vcfg.ai_policy, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    done = {(r["g"], r["arm"]) for r in load_rows(run_dir)}
    only = args.only_origin
    meta["only_origin"] = only
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    origins = list(DEFAULT_CONFIG.origin_ids())
    todo = [g for g in range(args.games) if any((g, a) not in done for a in arms)
            and (not only or origins[g % len(origins)] == only)]
    blocks = [todo[i:i + args.block] for i in range(0, len(todo), args.block)]
    print(f"{run_dir}：{len(todo)} 局待跑（{len(blocks)} 个任务，{args.workers} 个进程），打法 {arms}，"
          f"陪练 {args.crowd}，预算 {scfg['budget']}")
    deadline = time.time() + args.time_limit_hours * 3600 if args.time_limit_hours else None
    t0 = time.time()
    finished = 0
    files = {a: open(run_dir / f"{a.replace(':', '_').replace('/', '_')}.jsonl", "a", encoding="utf-8")
             for a in arms}
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            pending = set()
            it = iter(blocks)

            def submit_next() -> bool:
                if deadline and time.time() > deadline:
                    return False
                b = next(it, None)
                if b is None:
                    return False
                pending.add(ex.submit(run_block, {
                    "games": b, "arms": arms, "seed": args.seed, "crowd": args.crowd, "scfg": scfg,
                    "residuals": residuals, "done": {d for d in done if d[0] in b}, "variant": args.variant,
                    "telemetry": args.telemetry}))
                return True

            for _ in range(args.workers * 2):
                if not submit_next():
                    break
            while pending:
                fin, _ = wait(pending, return_when=FIRST_COMPLETED)
                for f in fin:
                    pending.discard(f)
                    for r in f.result():
                        files[r["arm"]].write(json.dumps(r, ensure_ascii=False) + "\n")
                        files[r["arm"]].flush()
                    finished += 1
                    el = time.time() - t0
                    print(f"  [{finished}/{len(blocks)}] {el / 60:.1f} 分钟，预计还要 "
                          f"{el / finished * (len(blocks) - finished) / 60:.1f} 分钟", flush=True)
                    submit_next()
    finally:
        for fh in files.values():
            fh.close()
    print(write_summary(run_dir))


if __name__ == "__main__":
    main()
