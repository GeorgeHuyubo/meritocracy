"""对比工具：大模型局 vs Python 局，用**同一套指标函数**逐项比，找出两边结论不一样的原因。

    python3 crosscheck.py --a llm_runs/<混坐目录> --b control:llm_runs/<混坐目录>
    python3 crosscheck.py --a llm_runs/20261008-000213-base --b python:symmetric:2000
    python3 crosscheck.py --a search_runs/<目录>/search.jsonl --b search_runs/<目录>/learned.jsonl

来源写法：
    <目录>                  大模型对局（game_*.json）；旧格式（没有遥测）从战报文字里还原能还原的
    control:<目录>          那个目录里的 Python 对照局（control_*.json）
    <文件>.jsonl            probe.py 的结果（要带 --telemetry 跑才有局内指标）
    python:symmetric:N      现跑 N 局学习型 AI 自对弈（六个座位都算）
    python:focal:N[:陪练]   现跑 N 局焦点座位局（学习型 AI 坐焦点，陪练默认 learned）

被统计的座位：有焦点座位的局只算焦点座位，否则六个座位都算。
每个指标按身份给出 A、B、差值 ±95%（独立样本），** = 区间不含 0；最后按 |z| 列出差异最大的几项。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import ai
import balance_stats as bs
import telemetry as T
from config import DEFAULT_CONFIG, Config

CFG = DEFAULT_CONFIG
TOP = CFG.president_rank - 1
ROUTES = ("MERIT", "MONEY", "BOTH", "TENURE", "FAMILY")
CARDS = ("WORK", "CORRUPT", "GRAFT", "REPORT", "ATTACK", "PROMOTE_MERIT", "PROMOTE_MONEY", "PROMOTE_ANY")
ROUTE_TEXT = {"政绩卓著": "MERIT", "四处打点": "MONEY", "两手都硬": "BOTH", "按工龄": "TENURE",
              "一纸调令": "FAMILY"}


# --------------------------------------------------------------------------
# 读数据：统一成 "normalized game"
# --------------------------------------------------------------------------

def _rounds_from_telemetry(tel: list[dict]) -> list[dict[int, dict]]:
    out = []
    for r in tel:
        out.append({p["pid"]: {
            "rank_after": p["rank_after"],
            "promotion": "FAMILY" if p.get("family") else p["promotion"],
            "attacked_n": p["attacker_count"],
            "reported_n": p["report_count_players"],
            "cards": p["cards"],
        } for p in r["players"]})
    return out


def _rounds_from_legacy(gm: dict, cfg: Config) -> list[dict[int, dict]]:
    """旧格式大模型局：官职从"结束后：…"还原，升职路线从固定战报文案还原，挨打从出牌目标还原。"""
    names = {p["name"]: p["pid"] for p in gm["players"]}
    rank_idx = {cfg.rank_name(i): i for i in range(cfg.president_rank + 1)}
    by_round: dict[int, dict[int, dict]] = defaultdict(lambda: {
        pid: {"rank_after": None, "promotion": "NONE", "attacked_n": 0, "reported_n": 0, "cards": []}
        for pid in names.values()})
    for line in gm.get("history", []):
        m = re.match(r"第 (\d+) 轮（事件：[^）]*）：(.*)。结束后：(.*)$", line)
        if not m:
            continue
        rnd, msgs, ranks = int(m.group(1)), m.group(2), m.group(3)
        rd = by_round[rnd]
        for part in ranks.split("、"):
            if " " in part:
                nm, rk = part.split(" ", 1)
                if nm in names and rk in rank_idx:
                    rd[names[nm]]["rank_after"] = rank_idx[rk]
        for seg in msgs.split("；"):
            nm = seg.split(" ", 1)[0]
            if nm not in names or "晋升" not in seg and "调任" not in seg:
                continue
            for key, route in ROUTE_TEXT.items():
                if key in seg:
                    rd[names[nm]]["promotion"] = route
    for d in gm.get("decisions", []):
        rd = by_round[d["round"]]
        for p in d["picks"]:
            a, t = p.get("action"), p.get("target")
            if a and a != "PROMOTE_FAMILY":
                rd[d["pid"]]["cards"].append(a)
            if t in rd and a == "ATTACK":
                rd[t]["attacked_n"] += 1
            if t in rd and a == "REPORT":
                rd[t]["reported_n"] += 1
    return [by_round[r] for r in sorted(by_round)]


def _place_fallback(players: list[dict], winners: list[int]) -> dict[int, int]:
    order = sorted(players, key=lambda p: (p["pid"] not in winners, -p["rank"], -p.get("money", 0), -p.get("merit", 0)))
    return {p["pid"]: i + 1 for i, p in enumerate(order)}


def normalize_llm(gm: dict, cfg: Config = CFG) -> dict[str, Any]:
    winners = gm.get("winners") or [p["pid"] for p in gm["players"] if p["winner"]]
    places = {p["pid"]: p["place"] for p in gm["players"] if p.get("place")} or _place_fallback(gm["players"], winners)
    focal = gm.get("focal") or {}
    tel = gm.get("telemetry")
    return {
        "g": gm["game"], "rounds": gm["rounds"],
        "end": gm.get("end") or ("president" if gm.get("president") else "timeout"),
        "winners": winners,
        "players": {p["pid"]: {"origin": p["origin"], "controller": p.get("controller", "llm"),
                               "place": places[p["pid"]], "rank": p["rank"]} for p in gm["players"]},
        "units": [focal["pid"]] if focal.get("pid") else [p["pid"] for p in gm["players"]],
        "rounds_data": _rounds_from_telemetry(tel) if tel else _rounds_from_legacy(gm, cfg),
        "legacy": not tel,
        "decisions": [d for d in gm.get("decisions", []) if d.get("controller", "llm") == "llm"],
    }


def normalize_probe(r: dict) -> dict[str, Any]:
    """probe.py 的一行 / llm_play 的对照局（都是 play_probe_game 的输出）。"""
    summ = r.get("summary") or {}
    players = {p["pid"]: {"origin": p["origin"], "controller": p.get("controller"), "place": p["place"],
                          "rank": p["rank"]} for p in summ.get("players", [])} or {
        r["focal_pid"]: {"origin": r["focal_origin"], "controller": r["arm"], "place": r["place"], "rank": None}}
    return {
        "g": r["g"], "rounds": r["rounds"], "end": r["end"], "winners": r["winners"],
        "players": players, "units": [r["focal_pid"]],
        "rounds_data": _rounds_from_telemetry(r["telemetry"]) if r.get("telemetry") else None,
        "legacy": False, "decisions": [],
    }


def load_source(spec: str, workers: int = 8, seed: int = 11) -> tuple[str, list[dict[str, Any]]]:
    if spec.startswith("control:"):
        d = Path(spec.split(":", 1)[1])
        return f"Python 对照（{d.name}）", [normalize_probe(json.loads(p.read_text(encoding="utf-8")))
                                          for p in sorted(d.glob("control_*.json"))]
    if spec.startswith("python:"):
        parts = spec.split(":")
        kind, n = parts[1], int(parts[2])
        crowd = parts[3] if len(parts) > 3 else "learned"
        return f"Python {kind}", run_python(kind, n, crowd, workers, seed)
    p = Path(spec)
    if p.suffix == ".jsonl":
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        return f"probe {p.parent.name}/{p.stem}", [normalize_probe(r) for r in rows]
    games = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(p.glob("game_*.json"))]
    return f"大模型 {p.name}", [normalize_llm(gm) for gm in games]


def _py_block(args: tuple) -> list[dict[str, Any]]:
    kind, games, crowd, seed = args
    import probe
    import search_ai

    out = []
    for g in games:
        if kind == "focal":
            r = probe.play_probe_game(CFG, g, seed, "learned", crowd, search_ai.SearchConfig(budget=0),
                                      want_telemetry=True)
            out.append(normalize_probe(r))
            continue
        seats = list(CFG.origin_ids())
        random.Random(f"{seed}/{g}/sym").shuffle(seats)
        gs = T.game_seed(seed, g)
        game = T.new_game(CFG, seats, gs)
        pool = ai.make_pool(CFG, random.Random(f"{gs}/sym"))
        tel = T.drive_game(game, {p: (lambda gm, pid: ai.turn(gm, pid, pool)) for p in game.players}, gs)
        summ = T.game_summary(game)
        out.append({
            "g": g, "rounds": summ["rounds"], "end": summ["end"], "winners": summ["winners"],
            "players": {p["pid"]: {"origin": p["origin"], "controller": "learned", "place": p["place"],
                                   "rank": p["rank"]} for p in summ["players"]},
            "units": sorted(game.players), "rounds_data": _rounds_from_telemetry(tel),
            "legacy": False, "decisions": [],
        })
    return out


def run_python(kind: str, n: int, crowd: str = "learned", workers: int = 8, seed: int = 11) -> list[dict]:
    chunks = [list(range(i, min(n, i + 25))) for i in range(0, n, 25)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_py_block, [(kind, c, crowd, seed) for c in chunks]))
    return [g for block in res for g in block]


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------

def unit_rows(games: list[dict]) -> list[dict[str, Any]]:
    """每个被统计的座位一行：这个座位这一局的各项原始量。"""
    rows = []
    for gm in games:
        nw = max(1, len(gm["winners"]))
        rd = gm["rounds_data"]
        first_round = None
        first_set: set[int] = set()
        if rd:
            for r in rd:
                tops = {pid for pid, x in r.items() if x["rank_after"] is not None and x["rank_after"] >= TOP}
                if tops:
                    first_set = tops
                    break
        for pid in gm["units"]:
            pl = gm["players"][pid]
            win = (1.0 / nw) if pid in gm["winners"] else 0.0
            row = {"origin": pl["origin"], "win": win,
                   "win_pres": win if gm["end"] == "president" else 0.0,
                   "win_timeout": win if gm["end"] != "president" else 0.0,
                   "rounds": gm["rounds"], "place": pl["place"]}
            if rd:
                mine = [r[pid] for r in rd if pid in r]
                reach = any(x["rank_after"] is not None and x["rank_after"] >= TOP for x in mine)
                routes = Counter(x["promotion"] for x in mine if x["promotion"] not in ("NONE", None))
                cards = Counter(c for x in mine for c in x["cards"])
                row.update({
                    "reach": float(reach), "first": float(pid in first_set) / max(1, len(first_set)),
                    "promotions": sum(routes.values()), "routes": routes, "cards": cards,
                    "n_cards": sum(cards.values()), "n_rounds": len(mine),
                    "attacked": sum(x["attacked_n"] for x in mine),
                    "reported": sum(x["reported_n"] for x in mine),
                })
            rows.append(row)
    return rows


def _agreement(games: list[dict]) -> dict[str, list[float]]:
    """大模型和"同一手牌程序 AI 会怎么打"出的牌是否一样（按牌名多重集）。"""
    out: dict[str, list[float]] = defaultdict(list)
    for gm in games:
        for d in gm["decisions"]:
            if "base_picks" not in d:
                continue
            o = gm["players"][d["pid"]]["origin"]
            mine = sorted(p.get("action") for p in d["picks"] if p.get("action") != "PROMOTE_FAMILY")
            base = sorted(p.get("action") if isinstance(p, dict) else p for p in d["base_picks"])
            out[o].append(float(mine == base))
            out["ALL"].append(float(mine == base))
    return out


def metrics(games: list[dict]) -> dict[str, dict[str, tuple[float, float, int]]]:
    """{身份或 ALL: {指标: (值, SE, 样本数)}}"""
    rows = unit_rows(games)
    origins = sorted({r["origin"] for r in rows})
    agree = _agreement(games)
    out: dict[str, dict[str, tuple[float, float, int]]] = {}
    for oid in origins + ["ALL"]:
        sel = [r for r in rows if oid == "ALL" or r["origin"] == oid]
        if not sel:
            continue
        m: dict[str, tuple[float, float, int]] = {}
        for key in ("win", "win_pres", "win_timeout", "rounds", "place"):
            v, se = bs.mean_se([r[key] for r in sel])
            m[key] = (v, se, len(sel))
        tel = [r for r in sel if "reach" in r]
        if tel:
            n = len(tel)
            for key in ("reach", "first", "promotions"):
                v, se = bs.mean_se([r[key] for r in tel])
                m[key] = (v, se, n)
            v, se = bs.ratio_se([r["win"] * r["reach"] for r in tel], [r["reach"] for r in tel])
            m["win_given_reach"] = (v, se, n)
            v, se = bs.ratio_se([r["win"] * (r["first"] > 0) for r in tel], [float(r["first"] > 0) for r in tel])
            m["win_given_first"] = (v, se, n)
            for route in ROUTES:
                v, se = bs.ratio_se([r["routes"].get(route, 0) for r in tel], [r["promotions"] for r in tel])
                m[f"route_{route}"] = (v, se, n)
            v, se = bs.ratio_se([r["attacked"] for r in tel], [r["n_rounds"] for r in tel])
            m["attacked_per_round"] = (v, se, n)
            v, se = bs.ratio_se([r["reported"] for r in tel], [r["n_rounds"] for r in tel])
            m["reported_per_round"] = (v, se, n)
            for c in CARDS:
                v, se = bs.ratio_se([r["cards"].get(c, 0) for r in tel], [r["n_cards"] for r in tel])
                m[f"card_{c}"] = (v, se, n)
        if agree.get(oid):
            v, se = bs.mean_se(agree[oid])
            m["agree_with_python"] = (v, se, len(agree[oid]))
        out[oid] = m
    return out


LABELS = {
    "win": "胜率", "win_pres": "靠当主席赢", "win_timeout": "打满 12 轮比家底赢", "rounds": "局长（轮）",
    "place": "平均名次", "reach": "到过省级", "first": "第一个到省级", "promotions": "每局升职次数",
    "win_given_reach": "到省级后夺冠率", "win_given_first": "第一个到省级后夺冠率",
    "attacked_per_round": "每轮挨几刀", "reported_per_round": "每轮被几份举报",
    "agree_with_python": "和 Python AI 出同一手牌",
    **{f"route_{r}": f"升职走{n}" for r, n in zip(ROUTES, ("政绩", "钱", "两手", "工龄", "调令"))},
    **{f"card_{c}": f"出牌占比·{c}" for c in CARDS},
}
PCT = {k for k in LABELS if k not in ("rounds", "place", "promotions", "attacked_per_round", "reported_per_round")}


def _fmt(key: str, v: float, se: float | None = None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if key in PCT:
        return bs.fmt_pct(v, se)
    return f"{v:.2f}" if se is None or math.isnan(se) else f"{v:.2f} ±{bs.Z * se:.2f}"


def compare(ma: dict, mb: dict) -> list[dict[str, Any]]:
    out = []
    for oid in sorted(set(ma) & set(mb), key=lambda o: (o == "ALL", o)):
        for key in LABELS:
            if key not in ma[oid] or key not in mb[oid]:
                continue
            (va, sa, na), (vb, sb, nb) = ma[oid][key], mb[oid][key]
            if any(math.isnan(x) for x in (va, vb)):
                continue
            d, dse = bs.diff_independent(va, sa if not math.isnan(sa) else 0.0, vb, sb if not math.isnan(sb) else 0.0)
            z = d / dse if dse > 0 else 0.0
            out.append({"origin": oid, "metric": key, "a": va, "a_se": sa, "b": vb, "b_se": sb,
                        "diff": d, "diff_se": dse, "z": z, "n_a": na, "n_b": nb})
    return out


def render(label_a: str, label_b: str, games_a: list, games_b: list, rows: list[dict]) -> str:
    name = {o["id"]: o["name"] for o in CFG.origin_definitions}
    name["ALL"] = "合计"
    legacy = sum(g["legacy"] for g in games_a + games_b)
    lines = [f"# 对比：A = {label_a}（{len(games_a)} 局） vs B = {label_b}（{len(games_b)} 局）", ""]
    if legacy:
        lines.append(f"注意：{legacy} 局是旧格式（没有遥测），局内指标从战报文字和出牌记录推断；"
                     "挨打只数玩家的牌（不含事件查办），升职路线认不出调令以外的家族升职。")
        lines.append("")
    origins = [o for o in list(CFG.origin_ids()) + ["ALL"] if any(r["origin"] == o for r in rows)]
    keys = [k for k in LABELS if any(r["metric"] == k for r in rows)]
    for k in keys:
        lines += [f"## {LABELS[k]}", "", "| 身份 | A | B | A − B |", "|---|---|---|---|"]
        for o in origins:
            r = next((x for x in rows if x["origin"] == o and x["metric"] == k), None)
            if r is None:
                continue
            mark = " **" if abs(r["z"]) > bs.Z else ""
            scale = 100 if k in PCT else 1
            unit = " 点" if k in PCT else ""
            lines.append(f"| {name.get(o, o)} | {_fmt(k, r['a'], r['a_se'])} | {_fmt(k, r['b'], r['b_se'])} | "
                         f"{scale * r['diff']:+.2f}{unit} ±{scale * bs.Z * r['diff_se']:.2f}{mark} |")
        lines.append("")
    top = sorted((r for r in rows if r["origin"] != "ALL"), key=lambda r: -abs(r["z"]))[:15]
    lines += ["## 差异最大的几项（按 |z|）", ""]
    for r in top:
        scale = 100 if r["metric"] in PCT else 1
        lines.append(f"- {name.get(r['origin'], r['origin'])} · {LABELS[r['metric']]}："
                     f"A {_fmt(r['metric'], r['a'])}，B {_fmt(r['metric'], r['b'])}，"
                     f"差 {scale * r['diff']:+.2f}（z = {r['z']:+.1f}）")
    return "\n".join(lines)


def to_source(label: str, games: list[dict], family: str) -> dict[str, Any]:
    rows = unit_rows(games)
    focal = all(len(g["units"]) == 1 for g in games)
    return bs.origin_source(label, family, "focal" if focal else "symmetric", None,
                            T.rules_fingerprint(CFG), [(r["origin"], r["win"], r["place"]) for r in rows],
                            n_games=len(games),
                            notes="legacy" if any(g["legacy"] for g in games) else "")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default="", help="输出目录（默认写在 A 旁边）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv)
    la, ga = load_source(args.a, args.workers, args.seed)
    lb, gb = load_source(args.b, args.workers, args.seed)
    rows = compare(metrics(ga), metrics(gb))
    md = render(la, lb, ga, gb, rows)
    out = Path(args.out) if args.out else (Path(args.a) if Path(args.a).is_dir() else Path(args.a).parent)
    out.mkdir(parents=True, exist_ok=True)
    (out / "crosscheck.md").write_text(md, encoding="utf-8")
    (out / "crosscheck.json").write_text(json.dumps({"a": la, "b": lb, "rows": rows}, ensure_ascii=False,
                                                    indent=1), encoding="utf-8")
    print(md)
    print(f"\n写到 {out / 'crosscheck.md'}")


if __name__ == "__main__":
    main()
