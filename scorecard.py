"""综合记分卡：把几把"尺子"的身份强弱放在一张表上，只有多把尺子同向才下结论。

    python3 scorecard.py \\
        --src "py_learned:学习型自对弈=python:symmetric:6000" \\
        --src "search:搜索 vs 学习型=search_runs/<目录>/search.jsonl" \\
        --src "search:搜索 vs 手写=search_runs/<目录2>/search.jsonl" \\
        --src "llm:大模型混坐=llm_runs/<混坐目录>" \\
        --src "llm:旧·全大模型=llm_runs/20261008-000213-base" --legacy "旧·全大模型" \\
        --uplift search_runs/<目录>/summary.json

--src 写法："家族:名字=来源"，来源同 crosscheck.py（目录 / control:目录 / .jsonl / python:...），
也可以直接给一份来源 JSON（probe.py 写的 source_*.json）。家族：py_learned / py_smart / search / llm。

身份效应 d：
    对称设计（六个座位都算）：d = 胜率 − 1/6
    焦点设计（每局一个被测座位）：d = 胜率 − 这个来源六个身份的平均胜率
        （扣掉被测玩家本身的水平：搜索 AI 整体就比陪练强，不能算成"每个身份都强"）

判定：
    确认强 / 确认弱：至少 3 个来源的 95% 区间整体在 0 以上 / 以下，这些来源来自至少 2 个家族，
                    而且没有任何来源显著反向
    有争议：有来源显著偏强、也有来源显著偏弱
    偏强 / 偏弱：1~2 个来源显著，没有反向
    无证据：其余
--legacy 标出的来源只展示、不参与判定（比如用过跨局笔记本、没有规则指纹的旧大模型局）。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import balance_stats as bs
import crosscheck
import telemetry as T
from config import DEFAULT_CONFIG

CFG = DEFAULT_CONFIG
FAMILY_NAMES = {"py_learned": "学习型 AI", "py_smart": "手写 AI", "search": "搜索 AI", "llm": "大模型"}


def effects(src: dict[str, Any]) -> dict[str, tuple[float, float, int]]:
    """{身份: (d, SE, n)}"""
    per = src["per_origin"]
    origins = [o for o in CFG.origin_ids() if o in per]
    if src["design"] == "symmetric":
        return {o: (per[o]["mean"] - 1 / 6, per[o]["se"], per[o]["n"]) for o in origins}
    k = len(origins)
    base = sum(per[o]["mean"] for o in origins) / k
    out = {}
    for o in origins:
        var = ((k - 1) / k) ** 2 * per[o]["se"] ** 2 + sum(per[j]["se"] ** 2 for j in origins if j != o) / k ** 2
        out[o] = (per[o]["mean"] - base, math.sqrt(var), per[o]["n"])
    return out


def verdict(signs: list[tuple[int, str]], min_sources: int = 3, min_families: int = 2) -> str:
    pos = [f for s, f in signs if s > 0]
    neg = [f for s, f in signs if s < 0]
    if pos and neg:
        return "有争议"
    for lst, word in ((pos, "强"), (neg, "弱")):
        if not lst:
            continue
        if len(lst) >= min_sources and len(set(lst)) >= min_families:
            return f"确认{word}"
        return f"偏{word}"
    return "无证据"


def load(spec: str, workers: int) -> dict[str, Any]:
    fam_name, src_spec = spec.split("=", 1)
    family, name = fam_name.split(":", 1)
    p = Path(src_spec)
    if p.suffix == ".json" and p.exists():
        src = json.loads(p.read_text(encoding="utf-8"))
        src["source"], src["family"] = name, family
        return src
    label, games = crosscheck.load_source(src_spec, workers)
    src = crosscheck.to_source(name, games, family)
    src["notes"] = (src.get("notes") or "") + f" {label}"
    return src


def build(sources: list[dict[str, Any]], legacy: set[str], uplift: dict | None,
          min_sources: int, min_families: int) -> tuple[str, dict[str, Any]]:
    name = {o["id"]: o["name"] for o in CFG.origin_definitions}
    fp_now = T.rules_fingerprint(CFG)
    eff = {s["source"]: effects(s) for s in sources}
    table = {}
    for o in CFG.origin_ids():
        signs = []
        for s in sources:
            if s["source"] in legacy or o not in eff[s["source"]]:
                continue
            d, se, _ = eff[s["source"]][o]
            sg = bs.significant(d, se)
            if sg:
                signs.append((sg, s["family"]))
        table[o] = {"verdict": verdict(signs, min_sources, min_families),
                    "cells": {s["source"]: eff[s["source"]].get(o) for s in sources}}
    lines = ["# 身份平衡综合记分卡", "",
             "每格 = 身份效应 d（百分点）±95%；↑/↓ = 区间整体在 0 以上 / 以下。"
             "对称设计 d = 胜率 − 16.67%，焦点设计 d = 胜率 − 该来源六个身份的平均。", ""]
    head = "| 身份 | " + " | ".join(
        f"{s['source']}（{FAMILY_NAMES.get(s['family'], s['family'])}，{'焦点' if s['design'] == 'focal' else '对称'}"
        f"{'，仅展示' if s['source'] in legacy else ''}）" for s in sources) + " | 判定 |"
    if uplift:
        head = head[:-len(" 判定 |")] + " 搜索提升（AI 盲点） | 判定 |"
    lines += [head, "|---" * (len(sources) + 2 + (1 if uplift else 0)) + "|"]
    for o in CFG.origin_ids():
        cells = []
        for s in sources:
            c = table[o]["cells"][s["source"]]
            if not c:
                cells.append("—")
                continue
            d, se, n = c
            mark = {1: " ↑", -1: " ↓", 0: ""}[bs.significant(d, se)]
            cells.append(f"{100 * d:+.1f} ±{100 * bs.Z * se:.1f}{mark}（{n}）")
        if uplift:
            u = uplift.get(o)
            cells.append("—" if not u else f"{100 * u['delta']:+.1f} ±{100 * bs.Z * u['se']:.1f}"
                         + {1: " ↑", -1: " ↓", 0: ""}[bs.significant(u["delta"], u["se"])])
        lines.append(f"| {name[o]} | " + " | ".join(cells) + f" | **{table[o]['verdict']}** |")
    lines += ["", f"判定规则：至少 {min_sources} 个来源同向显著、来自至少 {min_families} 个家族、没有显著反向，才算\"确认\"。"]
    if uplift:
        lines.append("搜索提升 = 同一批对局里搜索 AI 减学习型 AI 坐这个身份的胜率。某个身份特别大，"
                     "说明普通 AI 把它打亏了，Python AI 关于它的结论要打折扣。")
    warn = [s["source"] for s in sources if s.get("rules_fp") and s["rules_fp"] != fp_now]
    if warn:
        lines.append(f"注意：{'、'.join(warn)} 的规则指纹和当前规则不一致（{fp_now}），不同规则下的结果放在一起比要小心。")
    small = [s["source"] for s in sources if s.get("n_games") and s["n_games"] < 300]
    if small:
        lines.append(f"注意：{'、'.join(small)} 样本少于 300 局，区间很宽。")
    for s in sources:
        lines.append(f"- {s['source']}：{s.get('n_games')} 局，{s.get('notes', '').strip()}")
    return "\n".join(lines), {"sources": sources, "table": table, "rules_fp": fp_now}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", action="append", required=True, help="家族:名字=来源")
    ap.add_argument("--legacy", action="append", default=[], help="只展示不参与判定的来源名字")
    ap.add_argument("--uplift", default="", help="probe.py 的 summary.json：加一列搜索提升")
    ap.add_argument("--min-sources", type=int, default=3)
    ap.add_argument("--min-families", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="scorecard")
    args = ap.parse_args(argv)
    sources = [load(s, args.workers) for s in args.src]
    uplift = None
    if args.uplift:
        s = json.loads(Path(args.uplift).read_text(encoding="utf-8"))
        uplift = next(iter(s["uplift"].values()), None)
    md, data = build(sources, set(args.legacy), uplift, args.min_sources, args.min_families)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scorecard.md").write_text(md, encoding="utf-8")
    (out / "scorecard.json").write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(md)
    print(f"\n写到 {out / 'scorecard.md'}")


if __name__ == "__main__":
    main()
