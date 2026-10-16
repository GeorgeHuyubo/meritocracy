"""政绩路线 vs 金钱路线：两条路线平衡不平衡（参照 AI = policies/best.json，PSRO 均衡混合）。

    python3 routes.py                       # 现行规则
    python3 routes.py --variant md3         # sweep.py 的规则变体（比如政绩溢出 ÷3）
    python3 routes.py --variant md3 --games 3000 --workers 10 --out search_runs/routes/md3.md

两部分：
  1. 六身份混战（全部参照 AI）：按每个人主要靠哪条路升职分组（政绩为主 / 两边都有 / 金钱为主），
     各组的占比和胜率；各身份升职走政绩的比例、出牌里埋头工作 / 贪污的比例；身份胜率、主席率、局长。
     分组是结果不是原因（落后的人也可能被迫换路线），所以还有第 2 部分。
  2. 因果实验（1 个座位改打法，其余参照 AI，同一批种子配对）：
       只走政绩（禁贪污和以权谋私）vs 照常打；卷王这轮升不了职、又有两张埋头工作就双埋头 vs 卷王照常打。
"""

from __future__ import annotations

import argparse
import dataclasses
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import ai
import balance_stats as bs
import probe
import rules
import telemetry as T
from models import Card

GROUPS = ("政绩为主", "两边都有", "金钱为主")


def _cfg(variant: str, policy: str):
    cfg = probe.variant_cfg(variant)
    return dataclasses.replace(cfg, ai_policy=policy) if policy else cfg


def melee_block(args: tuple) -> list[dict[str, Any]]:
    variant, games, seed, policy = (*args, "")[:4]
    cfg = _cfg(variant, policy)
    rows = []
    for g in games:
        seats = list(cfg.origin_ids())
        random.Random(f"{seed}/{g}").shuffle(seats)
        gs = T.game_seed(seed, g)
        game = T.new_game(cfg, seats, gs)
        pool = ai.make_pool(cfg, random.Random(f"{gs}/mix"))
        tel = T.drive_game(game, {p: (lambda gm, pid: ai.turn(gm, pid, pool)) for p in game.players}, gs)
        nw = len(game.winners)
        end = "president" if game.history[-1].presidents else "timeout"
        for pid, pl in game.players.items():
            routes, cards = Counter(), Counter()
            for r in tel:
                for x in r["players"]:
                    if x["pid"] == pid:
                        if x["promotion"] != "NONE":
                            routes[x["promotion"]] += 1
                        cards.update(x["cards"])
            rows.append({"origin": pl.origin.value, "win": 1 / nw if pid in game.winners else 0.0,
                         "end": end, "rounds": game.round_number, "routes": dict(routes), "cards": dict(cards)})
    return rows


def group_of(routes: dict[str, int]) -> str | None:
    merit = routes.get("MERIT", 0)
    money = routes.get("MONEY", 0) + routes.get("BOTH", 0)
    tot = merit + money
    if tot == 0:
        return None
    share = merit / tot
    return "政绩为主" if share >= 0.6 else ("金钱为主" if share <= 0.4 else "两边都有")


def can_promote_now(game, pid: int) -> bool:
    """手里有晋升卡、资源也够这一级：这轮就能升。"""
    p = game.players[pid]
    cfg = game.cfg
    oid = p.origin.value if p.origin else None
    tc, mc = rules.merit_cost_at(p.rank, oid, cfg), rules.money_cost_at(p.rank, oid, cfg)
    merit_ok = tc is not None and p.merit >= tc
    money_ok = mc is not None and p.money >= mc
    cards = [d.card for d in game.hands[pid]]
    if cfg.needs_both(p.rank):
        return merit_ok and money_ok and any(c.is_promotion for c in cards)
    return any((c.can_use_merit and merit_ok) or (c.can_use_money and money_ok) for c in cards)


def focal_block(args: tuple) -> list[tuple[str, float]]:
    """mode：base / clean（禁贪污和以权谋私）/ grinder_base / grinder_dw（卷王有两张埋头就双埋头）。"""
    variant, mode, games, seed, policy = (*args, "")[:5]
    cfg = _cfg(variant, policy)
    out = []
    for g in games:
        rng = random.Random(f"{seed}/{g}")
        seats = list(cfg.origin_ids())
        rng.shuffle(seats)
        fs = seats.index("GRINDER") if mode.startswith("grinder") else rng.randrange(len(seats))
        gs = T.game_seed(seed, g)
        game = T.new_game(cfg, seats, gs)
        pool = ai.make_pool(cfg, random.Random(f"{gs}/mix"))
        focal_pool = ai.make_pool(cfg, random.Random(f"{gs}/focal"))
        if mode == "clean":
            focal_pool.allow_corrupt = False
        fpid = fs + 1

        def focal(gm, pid):
            # 这轮能用晋升卡升职就照常打（交给 AI）；升不了才双埋头攒政绩。
            # （以前是"有两张埋头就双埋头"：卷王保底两张，等于每轮都双埋头、从不打晋升卡，只能靠工龄升——测的是个傻卷王）
            if mode == "grinder_dw" and sum(d.card is Card.WORK for d in gm.hands[pid]) >= 2 \
                    and not can_promote_now(gm, pid):
                focal_pool.get(pid).observe(gm.public_state(), gm.private_state(pid))
                return [{"action": Card.WORK}, {"action": Card.WORK}]
            return ai.turn(gm, pid, focal_pool)

        deciders = {p: (lambda gm, pid: ai.turn(gm, pid, pool)) for p in game.players}
        deciders[fpid] = focal
        T.drive_game(game, deciders, gs, T.focal_order(len(seats), fpid))
        out.append((game.players[fpid].origin.value, 1 / len(game.winners) if fpid in game.winners else 0.0))
    return out


def run(variant: str, games: int, workers: int, seed: int = 41, policy: str = "") -> dict[str, Any]:
    chunk = max(1, games // (workers * 2))
    blocks = [list(range(i, min(games, i + chunk))) for i in range(0, games, chunk)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for res in ex.map(melee_block, [(variant, b, seed, policy) for b in blocks]) for r in res]
        focal = {}
        for mode in ("base", "clean", "grinder_base", "grinder_dw"):
            focal[mode] = [r for res in ex.map(focal_block, [(variant, mode, b, seed + 10, policy) for b in blocks])
                           for r in res]
    return {"variant": variant or "base", "games": games, "rows": rows, "focal": focal, "policy": policy}


def summarize(res: dict[str, Any]) -> dict[str, Any]:
    cfg = _cfg(res["variant"] if res["variant"] != "base" else "", res.get("policy", ""))
    rows = res["rows"]
    groups: dict[str, list[float]] = defaultdict(list)
    per_origin: dict[str, Counter] = defaultdict(Counter)
    wins: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        grp = group_of(r["routes"])
        if grp:
            groups[grp].append(r["win"])
        o = r["origin"]
        wins[o].append(r["win"])
        c = per_origin[o]
        c["merit"] += r["routes"].get("MERIT", 0)
        c["money"] += r["routes"].get("MONEY", 0) + r["routes"].get("BOTH", 0)
        c["work"] += r["cards"].get("WORK", 0)
        c["dirty"] += r["cards"].get("CORRUPT", 0) + r["cards"].get("GRAFT", 0)
        c["cards"] += sum(r["cards"].values())
    n_games = len(rows) // 6
    out = {"variant": res["variant"], "games": n_games, "policy": cfg.ai_policy,
           "president": sum(r["end"] == "president" for r in rows) / len(rows),
           "rounds": sum(r["rounds"] for r in rows) / len(rows),
           "groups": {g: {"share": len(groups[g]) / len(rows), "win": bs.mean_se(groups[g])} for g in GROUPS if groups[g]},
           "origins": {o: {"win": bs.mean_se(wins[o]),
                           "merit_share": per_origin[o]["merit"] / max(1, per_origin[o]["merit"] + per_origin[o]["money"]),
                           "work": per_origin[o]["work"] / max(1, per_origin[o]["cards"]),
                           "dirty": per_origin[o]["dirty"] / max(1, per_origin[o]["cards"])}
                       for o in cfg.origin_ids()},
           "focal": {m: bs.mean_se([w for _, w in v]) for m, v in res["focal"].items()}}
    allm = sum(c["merit"] for c in per_origin.values())
    alln = sum(c["merit"] + c["money"] for c in per_origin.values())
    out["merit_share"] = allm / max(1, alln)
    return out


def render(s: dict[str, Any]) -> str:
    cfg = _cfg(s["variant"] if s["variant"] != "base" else "", s.get("policy", ""))
    name = {o["id"]: o["name"] for o in cfg.origin_definitions}
    lines = [f"# 政绩 vs 金钱路线（规则 {s['variant']}，参照 AI = {cfg.ai_policy}，{s['games']} 局）", "",
             f"主席率 {100 * s['president']:.0f}%，平均局长 {s['rounds']:.2f} 轮，升职里走政绩的比例 {100 * s['merit_share']:.0f}%", "",
             "## 按主要升职路线分组", "", "| 分组 | 占比 | 胜率 |", "|---|---|---|"]
    for g, v in s["groups"].items():
        m, se = v["win"]
        lines.append(f"| {g} | {100 * v['share']:.1f}% | {bs.fmt_pct(m, se)} |")
    lines += ["", "## 各身份", "", "| 身份 | 胜率 | 升职走政绩 | 出牌·埋头工作 | 出牌·贪污 |", "|---|---|---|---|---|"]
    for o, v in s["origins"].items():
        m, se = v["win"]
        lines.append(f"| {name[o]} | {bs.fmt_pct(m, se)} | {100 * v['merit_share']:.0f}% | "
                     f"{100 * v['work']:.0f}% | {100 * v['dirty']:.0f}% |")
    f = s["focal"]

    def diff(a: str, b: str) -> str:
        (ma, sa), (mb, sb) = f[a], f[b]
        d, se = bs.diff_independent(ma, sa, mb, sb)
        mark = {1: " ↑", -1: " ↓", 0: ""}[bs.significant(d, se)]
        return f"{bs.fmt_pct(ma, sa)} vs {bs.fmt_pct(mb, sb)}，差 {100 * d:+.1f} ±{100 * bs.Z * se:.1f}{mark}"

    lines += ["", "## 因果实验（1 个座位改打法，其余参照 AI）", "",
              f"- 只走政绩（禁贪污和以权谋私） vs 照常打：{diff('clean', 'base')}",
              f"- 卷王升不了职时双埋头 vs 卷王照常打：{diff('grinder_dw', 'grinder_base')}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="")
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--policy", default="", help="参照 AI 换成这份权重文件（默认 cfg.ai_policy = policies/best.json）")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    md = render(summarize(run(args.variant, args.games, args.workers, policy=args.policy)))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
