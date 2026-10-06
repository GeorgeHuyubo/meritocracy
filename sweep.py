"""平衡实验：一次跑多个规则变体，每个变体一个进程并行，最后出一张对比表。

    python3 sweep.py --variants base,r1,loss5,r1+loss5
    python3 sweep.py --variants base,r1 --ablation-games 20000 --melee-games 3000   # 正式口径

变体名用 `+` 组合：`r1+loss5` = 同时套用 r1 和 loss5。新杠杆加到下面的 VARIANTS 里就行，
值可以是普通值，也可以是 `函数(基准配置) -> 值`（牌库这种复合字段用得上）。

每个变体算两样：
  * 消融（analysis.analyse_ablation）：禁掉一席的举报 / 攻击 / 贪污，看他和正常人的胜率差。
    差 > 0 = 不用反而赢得多（负收益）；差 < 0 = 这张牌强；≈ 0 = 定价合理
  * 六身份混战（每局六个出身随机发给六个座位）：主席率、平均轮数、各身份胜率的最强最弱差、
    举报和攻击的出牌率
"""

from __future__ import annotations

import argparse
import dataclasses
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis  # noqa: E402
from config import DEFAULT_CONFIG, Config  # noqa: E402


def deck(**counts: int):
    """改牌库里某几种牌的张数，其余不动。"""
    return lambda cfg: dict(cfg.card_deal_distribution, **counts)


VARIANTS: dict[str, dict[str, Any]] = {
    "base": {},
    # ---- 举报 ----
    "r1": {"card_deal_distribution": deck(REPORT=1)},
    "fee0": {"report_reward_fee": 0},    # 回到没有跑腿费（当前默认是 1）
    "fee2": {"report_reward_fee": 2},
    # ---- 攻击 ----
    "loss5": {"attack_block_merit_loss": Fraction(1, 5)},   # 穿小鞋掉 1/5 政绩
    "loss3": {"attack_block_merit_loss": Fraction(1, 3)},
    "up4": {"attack_steal_rank_bonus": Fraction(1, 4)},     # 目标每高一级多抢 1/4
    "up2": {"attack_steal_rank_bonus": Fraction(1, 2)},
    # 穿小鞋再扣几点，按目标官职倍率折算（基数 1 = 1/1/2/2，2 = 2/3/4/5，4 = 4/6/8/10）
    "blk1": {"attack_block_merit_penalty": 1},
    "blk2": {"attack_block_merit_penalty": 2},
    "blk4": {"attack_block_merit_penalty": 4},
    "blk6": {"attack_block_merit_penalty": 6},
    # 戴帽子扣成了，攻击者自己记一点功（按攻击者官职折算）
    "hat1": {"attack_hat_reward": 1},
    "hat2": {"attack_hat_reward": 2},
    "hat3": {"attack_hat_reward": 3},
    # ---- 牌库比例（举报一律保持 2 张）----
    "atk3": {"card_deal_distribution": deck(ATTACK=3)},
    "atk3_c2": {"card_deal_distribution": deck(ATTACK=3, CORRUPT=2)},
    "atk3_g1": {"card_deal_distribution": deck(ATTACK=3, GRAFT=1)},
    "c2": {"card_deal_distribution": deck(CORRUPT=2)},
    "w5": {"card_deal_distribution": deck(WORK=5)},
    "atk3_w5_c2": {"card_deal_distribution": deck(ATTACK=3, WORK=5, CORRUPT=2)},
    # ---- 出身 ----
    # 金钱路线拉回来（AI 估钱修好后贪污变成负收益）
    "c20": {"corrupt_card_distribution": [(18, 1), (19, 2), (20, 2), (21, 2), (22, 1)],
            "graft_card_distribution": [(9, 1), (10, 2), (11, 2), (12, 2), (13, 1)]},
    "c22": {"corrupt_card_distribution": [(20, 1), (21, 2), (22, 2), (23, 2), (24, 1)],
            "graft_card_distribution": [(10, 1), (11, 2), (12, 2), (13, 2), (14, 1)]},
    "seize23": {"report_seize_ratio": Fraction(2, 3)},   # 查实只没收本轮赃款的 2/3
    "seize12": {"report_seize_ratio": Fraction(1, 2)},   # 只没收举报人那份
    "mc90": {"promotion_money_costs": [14, 20, 27, 33]},
    "mc80": {"promotion_money_costs": [12, 18, 24, 30]},
    "mc110": {"promotion_money_costs": [17, 24, 33, 41]},   # 反方向：AI 变强后局结束得太快
    # 节奏：AI 变强后主席率 93%，门槛整体抬高
    "mt110": {"promotion_merit_costs": [20, 30, 40, 50]},
    "both110": {"promotion_merit_costs": [20, 30, 40, 50],
                "promotion_money_costs": [17, 24, 33, 41]},
    "both120": {"promotion_merit_costs": [22, 32, 43, 54],
                "promotion_money_costs": [18, 26, 36, 44]},
    # 身份：官二代、卷王偏强
    "off34": {"origin_patronage_merit_ratio": Fraction(3, 4)},
    "grind3": {"origin_grinder_overtime_multiplier": 3},
    "rich18": {"origin_old_money_start": 18},   # 富二代开局钱（默认 15）
    "rich20": {"origin_old_money_start": 20},
}


def build(name: str, base: Config = DEFAULT_CONFIG) -> Config:
    changes: dict[str, Any] = {}
    for part in name.split("+"):
        if part not in VARIANTS:
            raise SystemExit(f"没有这个变体：{part!r}（可选：{', '.join(VARIANTS)}）")
        for key, value in VARIANTS[part].items():
            changes[key] = value(base) if callable(value) else value
    return dataclasses.replace(base, **changes)


def run_variant(name: str, ablation_games: int, melee_games: int, seed: int) -> dict[str, Any]:
    cfg = build(name)
    t0 = time.time()
    out: dict[str, Any] = {"name": name}

    if ablation_games:
        abl = analysis.analyse_ablation(6, ablation_games, cfg, random.Random(seed))
        out["ablation"] = {
            label: (r["delta"], r["ci_half_width"]) for label, r in abl["cards"].items()
        }

    if melee_games:
        rng = random.Random(seed + 1)
        origin_ids = cfg.origin_ids()
        records = []
        for _ in range(melee_games):
            pool = list(origin_ids)
            rng.shuffle(pool)
            records.append(analysis.play(
                6, rng, cfg, analysis.build_assignment(["smart"] * 6, cfg, rng),
                origins=[pool[i % len(pool)] for i in range(6)] if pool else None,
            ))
        lead = analysis.analyse_leadership(records, cfg)
        out["president_pct"] = lead["president_pct"]
        out["avg_rounds"] = lead["avg_rounds"]
        if origin_ids:
            res = analysis.analyse_origin_results(records, cfg)
            rates = {oid: r["win_pct"] for oid, r in res.items()}
            out["origins"] = rates
            out["spread"] = round(max(rates.values()) - min(rates.values()), 2)
        offered = sum((r.offered for r in records), start=type(records[0].offered)())
        chosen = sum((r.chosen for r in records), start=type(records[0].chosen)())
        out["play_rate"] = {
            card: round(100 * chosen[card] / offered[card], 1) if offered[card] else 0.0
            for card in ("REPORT", "ATTACK")
        }
    out["seconds"] = round(time.time() - t0)
    return out


def fmt_delta(pair: tuple[float, float] | None) -> str:
    if pair is None:
        return "—"
    d, half = pair
    return f"{d:+.2f}±{half:.2f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="一次跑多个规则变体的平衡实验")
    ap.add_argument("--variants", default="base", help="逗号分隔；用 + 组合，例如 r1+loss5")
    ap.add_argument("--ablation-games", type=int, default=6000)
    ap.add_argument("--melee-games", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=0, help="并行进程数（默认 = 变体数）")
    args = ap.parse_args(argv)

    names = [n.strip() for n in args.variants.split(",") if n.strip()]
    for n in names:
        build(n)  # 名字写错了先报
    jobs = args.jobs or len(names)
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futures = [
            ex.submit(run_variant, n, args.ablation_games, args.melee_games, args.seed)
            for n in names
        ]
        results = [f.result() for f in futures]

    print(f"消融 {args.ablation_games} 局 / 六身份混战 {args.melee_games} 局，seed={args.seed}")
    print("消融差值：正 = 不用反而赢得多（负收益），负 = 这张牌强，≈0 = 定价合理\n")
    header = (f"{'变体':<16}{'举报Δ':>13}{'攻击Δ':>13}{'贪污Δ':>13}"
              f"{'主席率':>8}{'平均轮':>7}{'身份差':>7}{'举报出牌':>9}{'攻击出牌':>9}")
    print(header)
    for r in results:
        abl = r.get("ablation", {})
        pr = r.get("play_rate", {})
        print(
            f"{r['name']:<16}{fmt_delta(abl.get('举报')):>13}{fmt_delta(abl.get('攻击')):>13}"
            f"{fmt_delta(abl.get('贪污')):>13}"
            f"{r.get('president_pct', 0):>7.1f}%{r.get('avg_rounds', 0):>7.2f}"
            f"{r.get('spread', 0):>7.2f}{pr.get('REPORT', 0):>8.1f}%{pr.get('ATTACK', 0):>8.1f}%"
        )
    print("\n各身份胜率：")
    for r in results:
        if "origins" in r:
            cells = "  ".join(
                f"{(DEFAULT_CONFIG.origin(o) or {'name': o})['name']} {v:.1f}"
                for o, v in sorted(r["origins"].items(), key=lambda kv: -kv[1])
            )
            print(f"  {r['name']:<16}{cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
