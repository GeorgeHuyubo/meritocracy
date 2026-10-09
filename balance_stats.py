"""平衡评估用的统计小工具（纯 Python，依赖里没有 numpy）。

口径：一局一个样本（同一局的几个座位不是独立的），区间一律 ±1.96·SE。
"""

from __future__ import annotations

import math
import random
from typing import Any, Iterable, Sequence

Z = 1.96


def mean_se(xs: Sequence[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def paired_diff(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """同一批对局两份打法的差（a - b）：逐局相减再求均值和 SE。"""
    assert len(a) == len(b)
    return mean_se([x - y for x, y in zip(a, b)])


def diff_independent(m1: float, se1: float, m2: float, se2: float) -> tuple[float, float]:
    return m1 - m2, math.sqrt(se1 ** 2 + se2 ** 2)


def ratio_se(nums: Sequence[float], dens: Sequence[float]) -> tuple[float, float]:
    """按局聚类的比值 Σnum / Σden 及其 delta 法 SE（比如"升职里走钱路的占比"）。"""
    n = len(nums)
    tn, td = sum(nums), sum(dens)
    if n == 0 or td == 0:
        return float("nan"), float("nan")
    r = tn / td
    if n == 1:
        return r, float("nan")
    dbar = td / n
    resid = [(x - r * d) for x, d in zip(nums, dens)]
    var = sum(e * e for e in resid) / (n - 1)
    return r, math.sqrt(var / n) / dbar


def bootstrap_mean_se(xs: Sequence[float], reps: int = 1000, seed: int = 0) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(xs)
    ms = []
    for _ in range(reps):
        ms.append(sum(xs[rng.randrange(n)] for _ in range(n)) / n)
    m = sum(xs) / n
    mm = sum(ms) / reps
    return m, math.sqrt(sum((x - mm) ** 2 for x in ms) / (reps - 1))


def ci(m: float, se: float) -> tuple[float, float]:
    return m - Z * se, m + Z * se


def significant(m: float, se: float) -> int:
    """+1 = 区间整体在 0 以上，-1 = 整体在 0 以下，0 = 跨 0 或没法算。"""
    if se is None or math.isnan(se) or math.isnan(m):
        return 0
    lo, hi = ci(m, se)
    return 1 if lo > 0 else (-1 if hi < 0 else 0)


def fmt_pct(m: float, se: float | None = None, digits: int = 1) -> str:
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "—"
    if se is None or math.isnan(se):
        return f"{100 * m:.{digits}f}%"
    return f"{100 * m:.{digits}f}% ±{100 * Z * se:.{digits}f}"


def origin_source(source: str, family: str, design: str, crowd: str | None,
                  rules_fp: str | None, rows: Iterable[tuple[str, float, float | None]],
                  notes: str = "", n_games: int | None = None) -> dict[str, Any]:
    """记分卡的一份"来源"：rows = (身份, 这个座位的胜率份额, 名次或 None)，每行一个座位样本。

    design="focal"：每局只有一个被测座位（行 = 局）；"symmetric"：六个座位都算（行 = 座位，
    同一局的六行加起来恰好是 1，SE 按独立样本算会略偏大——偏保守，可以接受）。
    """
    by: dict[str, dict[str, list[float]]] = {}
    for oid, share, place in rows:
        d = by.setdefault(oid, {"share": [], "place": []})
        d["share"].append(share)
        if place is not None:
            d["place"].append(place)
    per = {}
    for oid, d in sorted(by.items()):
        m, se = mean_se(d["share"])
        pm, pse = mean_se(d["place"]) if d["place"] else (float("nan"), float("nan"))
        per[oid] = {"n": len(d["share"]), "mean": m, "se": se, "placement": pm, "placement_se": pse}
    return {"source": source, "family": family, "design": design, "crowd": crowd,
            "rules_fp": rules_fp, "n_games": n_games, "notes": notes, "per_origin": per}
