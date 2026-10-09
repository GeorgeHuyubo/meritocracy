"""搜索型 AI：出牌前把几个候选打法各往后推演几十局，挑赢面最大的。

用途是**平衡探针**，不是上线对手：让它坐某个身份、和 5 个普通 AI 同桌，
看"这个身份打好了能有多强"、普通 AI 有没有把它打亏（probe.py）。

做法（信息集重采样 + rollout，每个决策点现算，不用训练）：
  * 候选：从我自己的学习型 AI 的出牌分布里抽样（围堵开 / 关各抽一半——允许它不跟全桌
    围堵、搭别人的便车），按"牌 + 顺序 + 目标"去重；能换牌时再加一个"换牌"候选。
  * 重采样：深拷贝对局，把我看不到的东西全换掉——对手手牌按发牌规则重发、
    对手的钱按我的估计 + 校准残差抽、本轮事件重抽、对手这轮已经选的牌清掉。
    对手在推演里用"影子 AI"：只喂公开信息长出来的记忆，不碰他们真实的私密观察。
  * 推演：每个候选在**同一批**重采样上打到终局（公共随机数），赢了得 1/冠军人数。
  * 预算：逐轮淘汰（successive halving）；最后只有比基础 AI 首选高出 margin、
    配对 t > t_min 才改选，防止被噪声带偏。

    python3 search_ai.py --calibrate-money 2000   # 拟合"真实存款 - 估计"的残差表
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import ai
import rules
import telemetry
from config import DEFAULT_CONFIG, Config
from game import Game
from models import Card, Selection

RESIDUALS_PATH = Path(__file__).resolve().parent / "policies" / "money_residuals.json"
REDRAW = ("REDRAW",)


@dataclass
class SearchConfig:
    k: int = 4                  # 留几个出牌候选（不含"换牌"）
    budget: int = 96            # 每个决策点总共推演几局
    margin: float = 0.02        # 改选门槛：平均收益至少高这么多
    t_min: float = 1.0          # 且配对 t 值超过它
    skip_top_prob: float = 0.90  # 基础 AI 首选的概率这么高、又不能换牌，就不搜了
    n_samples: int = 32         # 抽多少手基础 AI 的出牌来凑候选
    include_redraw: bool = True
    redraw_mode: str = "search"  # search = 换不换牌也搜；base = 照普通 AI 的规则换（只搜出牌，诊断用）
    max_redraws: int = ai.MAX_PANIC_REDRAWS
    money_model: str = "residual"  # residual（校准残差）| lognormal
    money_rel_sigma: float = 0.35
    money_abs_floor: float = 2.0
    seed: int = 0


@dataclass
class Candidate:
    key: tuple
    picks: list[dict]
    prior: float
    rewards: dict[int, float] = field(default_factory=dict)  # 重采样编号 -> 收益

    def mean(self, idx: list[int] | None = None) -> float:
        vals = [self.rewards[i] for i in (idx if idx is not None else self.rewards)]
        return sum(vals) / len(vals) if vals else 0.0


def key_str(key: tuple) -> str:
    if key == REDRAW:
        return "REDRAW"
    return "|".join(f"{a}@{t}" if t is not None else a for a, t in key) or "PASS"


def _pick_key(picks: list[dict]) -> tuple:
    return tuple((p["action"].value if isinstance(p["action"], Card) else str(p["action"]),
                  p.get("target")) for p in picks)


# --------------------------------------------------------------------------
# 对手存款的残差表
# --------------------------------------------------------------------------

def _bucket(rank: int, rnd: int, origin: str | None) -> str:
    return f"{rank}|{min(3, (rnd - 1) // 3)}|{int(origin == 'RICH')}"


def load_residuals(path: Path = RESIDUALS_PATH) -> dict[str, list[int]] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["residuals"]


def sample_money(est: float, rank: int, rnd: int, origin: str | None, rng: random.Random,
                 scfg: SearchConfig, residuals: dict[str, list[int]] | None) -> int:
    """对手存款的一个样本。est 已经补上本轮工资。"""
    if scfg.money_model == "residual" and residuals:
        table = residuals.get(_bucket(rank, rnd, origin)) or residuals.get(f"{rank}|*|*")
        if table:
            return max(0, round(est + rng.choice(table)))
    noisy = est * math.exp(rng.gauss(0.0, scfg.money_rel_sigma)) + rng.gauss(0.0, scfg.money_abs_floor)
    return max(0, round(noisy))


def calibrate_money(games: int, cfg: Config = DEFAULT_CONFIG, seed: int = 7,
                    cap: int = 600) -> dict[str, Any]:
    """学习型 AI 自对弈：每轮出牌前，每个人对每个对手的"真实存款 - (估计 + 本轮工资)"。"""
    by_bucket: dict[str, list[int]] = defaultdict(list)
    by_rank: dict[str, list[int]] = defaultdict(list)
    seen: dict[str, int] = defaultdict(int)
    rng = random.Random(seed)
    origins = list(cfg.origin_ids())
    for g in range(games):
        gs = telemetry.game_seed(seed, g)
        seats = origins[:]
        rng.shuffle(seats)
        game = telemetry.new_game(cfg, seats, gs)
        pool = ai.make_pool(cfg, random.Random(f"{gs}/pool"))

        def sample(gm: Game) -> None:
            pub = gm.public_state()
            for i in gm.players:
                agent = pool.get(i)
                agent.observe(pub, gm.private_state(i))
                for j, pj in gm.players.items():
                    if j == i:
                        continue
                    est = agent.models[j].money_est + cfg.salary(pj.rank)
                    oid = pj.origin.value if pj.origin else None
                    res = round(pj.money - est)
                    for key, table in ((_bucket(pj.rank, gm.round_number, oid), by_bucket),
                                       (f"{pj.rank}|*|*", by_rank)):
                        # 蓄水池抽样：每个桶最多留 cap 个，但对所有对局一视同仁（不偏向前几局）
                        seen[key] += 1
                        b = table[key]
                        if len(b) < cap:
                            b.append(res)
                        else:
                            j = rng.randrange(seen[key])
                            if j < cap:
                                b[j] = res

        def decider(gm: Game, pid: int) -> list[dict]:
            if pid == min(gm.players):
                sample(gm)
            return ai.turn(gm, pid, pool)

        telemetry.drive_game(game, {pid: decider for pid in game.players}, gs)
    table = {**by_bucket, **by_rank}
    summary = {k: {"n": seen[k], "mean": round(sum(v) / len(v), 2)} for k, v in sorted(table.items()) if v}
    return {"games": games, "rules_fp": telemetry.rules_fingerprint(cfg),
            "summary": summary, "residuals": table}


# --------------------------------------------------------------------------
# 搜索型 AI
# --------------------------------------------------------------------------

class Searcher:
    """一个座位的搜索型 AI。base_pool 里是我自己的真实学习型 AI（真实记忆），
    shadow 里是每个对手的"影子 AI"（只看公开信息）。"""

    def __init__(self, pid: int, cfg: Config = DEFAULT_CONFIG,
                 scfg: SearchConfig | None = None, base_rng_seed: Any = 0,
                 residuals: dict[str, list[int]] | None = None) -> None:
        self.pid = pid
        self.cfg = cfg
        self.scfg = scfg or SearchConfig()
        self.base_pool = ai.make_pool(cfg, random.Random(f"{base_rng_seed}/base"))
        self.shadow = ai.make_pool(cfg, random.Random(f"{base_rng_seed}/shadow"))
        self.rng = random.Random(f"{self.scfg.seed}/{base_rng_seed}/{pid}")
        self.residuals = residuals if residuals is not None else (
            load_residuals() if self.scfg.money_model == "residual" else None)
        self.log: list[dict[str, Any]] = []

    # ---- 入口 ----
    def turn(self, game: Game, pid: int) -> list[dict]:
        assert pid == self.pid
        if self.scfg.budget <= 0:
            return ai.turn(game, pid, self.base_pool)  # 预算 0 = 和基础 AI 逐位相同
        if self.scfg.redraw_mode == "base":
            for _ in range(self.scfg.max_redraws):
                if not ai.wants_redraw(game, pid, self.base_pool):
                    break
                game.redraw(pid)
        for n_redraw in range(self.scfg.max_redraws + 1):
            public, private = game.public_state(), game.private_state(pid)
            me = self.base_pool.get(pid)
            me.observe(public, private)
            for o in game.players:
                if o != pid:
                    self.shadow.get(o).observe(public, None)
            cands = self._candidates(public, private)
            can_redraw = (self.scfg.include_redraw and self.scfg.redraw_mode == "search"
                          and bool(private.get("redraw_affordable"))
                          and n_redraw < self.scfg.max_redraws)
            entry: dict[str, Any] = {"round": game.round_number, "n_redraw": n_redraw,
                                     "n_cands": len(cands), "can_redraw": can_redraw}
            if not can_redraw and (len(cands) <= 1 or cands[0].prior >= self.scfg.skip_top_prob):
                entry.update(searched=False, deviated=False)
                self.log.append(entry)
                return self._commit(game, pid, public, private, None)
            if can_redraw:
                cands.append(Candidate(REDRAW, [], 0.0))
            best, info = self._search(game, pid, cands)
            entry.update(searched=True, **info)
            self.log.append(entry)
            if best.key == REDRAW:
                game.redraw(pid)
                continue
            return self._commit(game, pid, public, private, best)
        public, private = game.public_state(), game.private_state(pid)
        return self._commit(game, pid, public, private, None)

    def _commit(self, game, pid, public, private, best: Candidate | None) -> list[dict]:
        """真实 AI 照常走一遍 decide（更新记忆、校准用的预估），搜到更好的就换成搜的那手。"""
        picks = self.base_pool.get(pid).decide(public, private)
        base = [{"action": c, "target": t} for c, t in picks]
        return base if best is None else best.picks

    # ---- 候选 ----
    def _candidates(self, public, private) -> list[Candidate]:
        me = self.base_pool.get(self.pid)
        counts: dict[tuple, int] = defaultdict(int)
        picks_of: dict[tuple, list[dict]] = {}
        on_counts: dict[tuple, int] = defaultdict(int)
        n = max(2, self.scfg.n_samples)
        no_pile = replace(self.cfg, ai_dogpile=False)
        for i in range(n):
            c = copy.deepcopy(me)
            c.rng = random.Random(self.rng.getrandbits(64))
            if hasattr(c, "record"):
                c.record = False
            dogpile = i % 2 == 0
            if not dogpile:
                c.cfg = no_pile
            picks = [{"action": card, "target": t} for card, t in c.decide(public, private)]
            key = _pick_key(picks)
            counts[key] += 1
            picks_of[key] = picks
            if dogpile:
                on_counts[key] += 1
        base_key = max(on_counts, key=lambda k: (on_counts[k], counts[k]))
        ranked = sorted(counts, key=lambda k: (k != base_key, -counts[k]))
        return [Candidate(k, picks_of[k], on_counts[k] / max(1, sum(on_counts.values()))
                          if k == base_key else counts[k] / n)
                for k in ranked[: self.scfg.k]]

    # ---- 搜索 ----
    def _search(self, game: Game, pid: int, cands: list[Candidate]) -> tuple[Candidate, dict]:
        base = cands[0]
        survivors = list(range(len(cands)))
        rungs = max(1, math.ceil(math.log2(len(cands))))
        per_rung = max(len(cands) * 2, self.scfg.budget // rungs)
        seeds: list[int] = []
        used = 0
        for r in range(rungs):
            n_each = max(2, per_rung // len(survivors))
            new = [self.rng.getrandbits(64) for _ in range(n_each)]
            start = len(seeds)
            seeds.extend(new)
            for ci in survivors:
                for j, s in enumerate(new, start):
                    cands[ci].rewards[j] = self._rollout(game, pid, cands[ci], s)
                    used += 1
            if r < rungs - 1 and len(survivors) > 2:
                idx = list(range(len(seeds)))
                ordered = sorted(survivors, key=lambda ci: -cands[ci].mean(idx))
                keep = ordered[: max(2, len(ordered) // 2)]
                if 0 not in keep:
                    keep = keep[:-1] + [0]
                survivors = keep
        idx = list(range(len(seeds)))
        best_i = max(survivors, key=lambda ci: cands[ci].mean(idx))
        diffs = [cands[best_i].rewards[j] - base.rewards[j] for j in idx]
        m = sum(diffs) / len(diffs)
        var = sum((d - m) ** 2 for d in diffs) / max(1, len(diffs) - 1)
        se = math.sqrt(var / len(diffs))
        t = m / se if se > 0 else (math.inf if m > 0 else 0.0)
        chosen = cands[best_i] if (best_i != 0 and m > self.scfg.margin and t > self.scfg.t_min) else base
        info = {}
        rd = next((c for c in cands if c.key == REDRAW), None)
        if rd is not None:
            common = [j for j in rd.rewards if j in base.rewards]
            if common:
                info["redraw_vs_base"] = round(sum(rd.rewards[j] - base.rewards[j] for j in common) / len(common), 4)
                info["redraw_n"] = len(common)
        info.update({
            "rollouts": used,
            "base": key_str(base.key),
            "chosen": key_str(chosen.key),
            "deviated": chosen is not base,
            "redraw": chosen.key == REDRAW,
            "gain": round(m if chosen is not base else 0.0, 4),
            "means": {key_str(cands[ci].key): round(cands[ci].mean(), 3) for ci in survivors},
        })
        return chosen, info

    # ---- 信息集重采样 ----
    def determinize(self, game: Game, pid: int, rng: random.Random) -> Game:
        cfg = self.cfg
        g = copy.deepcopy(game)
        g.rng = random.Random(rng.getrandbits(64))
        me = self.base_pool.get(pid)
        rnd = game.round_number
        for o, p in g.players.items():
            if o == pid:
                continue
            g.hands[o] = rules.deal_hand_for(p, g.rng, cfg)
            g.selections[o] = Selection()
            g.redraw_count.pop(o, None)
            g.redraw_spent.pop(o, None)
            g.ledger.pop(o, None)
            model = me.models.get(o) or ai.OpponentModel()
            est = model.money_est + cfg.salary(p.rank)
            p.money = sample_money(est, p.rank, rnd, p.origin.value if p.origin else None,
                                   rng, self.scfg, self.residuals)
        mine = game.players[pid]
        tipoff = mine.origin is not None and mine.origin.value == "OFFICIAL" and cfg.origin_official_tipoff
        if not tipoff:
            g.next_event = rules.pick_event(g.rng, cfg)
        return g

    def _sim_pool(self, game: Game, pid: int, rng: random.Random) -> ai.AgentPool:
        sim = ai.AgentPool(cfg=self.cfg, rng=random.Random(rng.getrandbits(64)),
                           policy=self.base_pool.policy,
                           policy_by_origin=self.base_pool.policy_by_origin)
        sim.agents = {pid: copy.deepcopy(self.base_pool.get(pid))}
        for o in game.players:
            if o != pid:
                sim.agents[o] = copy.deepcopy(self.shadow.get(o))
        for a in sim.agents.values():
            a.rng = sim.rng
            if hasattr(a, "record"):
                a.record = False
        return sim

    def _rollout(self, game: Game, pid: int, cand: Candidate, seed: int) -> float:
        rng = random.Random(seed)
        g = self.determinize(game, pid, rng)
        sim = self._sim_pool(g, pid, rng)
        for o in sorted(g.players):
            if o == pid:
                continue
            g.select_actions(o, ai.turn(g, o, sim))
            g.lock_action(o)
        if cand.key == REDRAW:
            g.redraw(pid)
            g.select_actions(pid, ai.turn(g, pid, sim))
        else:
            g.select_actions(pid, cand.picks)
        g.lock_action(pid)
        while True:
            g.force_lock_all()
            g.reveal_event()
            g.resolve()
            if g.is_over:
                break
            g.advance_round()
            for o in sorted(g.players):
                g.select_actions(o, ai.turn(g, o, sim))
                g.lock_action(o)
        return 1.0 / len(g.winners) if pid in g.winners else 0.0

    def stats(self) -> dict[str, Any]:
        s = [e for e in self.log if e.get("searched")]
        return {
            "decisions": len(self.log),
            "searched": len(s),
            "deviated": sum(1 for e in s if e.get("deviated")),
            "redraws_chosen": sum(1 for e in s if e.get("redraw")),
            "mean_gain": round(sum(e.get("gain", 0.0) for e in s) / len(s), 4) if s else 0.0,
            "rollouts": sum(e.get("rollouts", 0) for e in s),
        }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate-money", type=int, default=0, metavar="GAMES",
                    help="学习型 AI 自对弈这么多局，拟合对手存款的残差表，写到 policies/money_residuals.json")
    ap.add_argument("--out", default=str(RESIDUALS_PATH))
    args = ap.parse_args(argv)
    if args.calibrate_money:
        res = calibrate_money(args.calibrate_money)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False)
        for k, v in res["summary"].items():
            print(f"  {k:10s} n={v['n']:4d}  平均残差 {v['mean']:+.1f}")
        print(f"写到 {args.out}")


if __name__ == "__main__":
    main()
