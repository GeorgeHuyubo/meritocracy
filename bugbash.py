"""Bug bash：让 AI 大量对局，每一轮按**写下来的规则**独立重算一遍，和引擎的结算对账。

    python3 bugbash.py --games 200                 # 小样本试跑
    python3 bugbash.py --hours 8 --workers 10      # 过夜
    python3 bugbash.py --hours 8 --resume search_runs/bugbash/<目录>
    python3 bugbash.py --repro search_runs/bugbash/<目录> <编号>   # 重放某条发现，打印那一轮的完整状态

规则来源：llm_play.rules_text()（大模型看的规则）、static/app.js 的规则速查、config.py 的常量和注释。
裁判**不调用 rules.py 的结算函数**（不然只会把 bug 原样复制一遍），只读 config 的常量。
写下来的规则说不清的地方（几份说明互相矛盾、或者根本没写），裁判照引擎的口径走，另记一条「口径」。

查什么（每条发现的 kind 前缀）：
  ledger     流水账对不上钱 / 政绩的变化
  resolve    独立重算的结算结果和引擎不一样（生产、攻击、加班、举报没收、晋升、工龄……）
  spec       几份规则说明之间互相矛盾，或者规则没写、引擎自己定了口径
  msg        公报 / 私密提示和实际结算对不上（数字、升降级、"0 点"）
  preview    晋升卡的"现在可用"提示和实际结果对不上
  deal       发牌、换牌价格、开局送钱、工资、事件
  end        终局：主席、同轮多人登顶、打满轮数的名次
  snapshot   存盘再读出来状态变了
  privacy    公开状态里出现了不该公开的东西
  exception  合法操作报错（AI 出了非法牌也记在这里）

输出：<目录>/findings.jsonl（每条发现）+ findings.md（按类汇总、每类前几个例子）+ state.json（断点续跑）。
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ai  # noqa: E402
import analysis  # noqa: E402
import telemetry as T  # noqa: E402
from config import DEFAULT_CONFIG, Config  # noqa: E402
from game import Game, GameError  # noqa: E402
from models import Action, Card, GameEvent, Origin, PromotionKind  # noqa: E402

ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# 对局组合
# --------------------------------------------------------------------------

# 规则变体：默认规则之外，把各个开关都拨一遍（AI 都是默认规则下训的，打得不一定好，但规则得结算对）
VARIANTS: dict[str, dict[str, Any]] = {
    "base": {},
    "redfree": {"origin_red_shield_fee": None},
    "redsoft": {"origin_red_shield_fee_strict": False},
    "peasant1": {"origin_peasant_max_attackers": 1},
    "family": {"origin_red_family_card": True},
    "tipoff": {"origin_official_tipoff": True},
    "oldgossip": {"gossip_illegal_first": False},
    "nobribe": {"report_catches_bribery": False},
    "offtop": {"origin_patronage_top_ratio": Fraction(1)},
    "offtop45": {"origin_patronage_top_ratio": Fraction(4, 5)},
    "md3": {"merit_overflow_divisor": 3},
    "bribehalf": {"bribe_forfeit_ratio": Fraction(1, 2)},
    "noorigins": {"origins_enabled": False},
}
VARIANT_WEIGHTS = {"base": 8}  # 其余各 1

CONTROLLERS = ("learned", "hand", "scripted", "random")
CONTROLLER_WEIGHTS = (35, 20, 15, 30)


def variant_cfg(name: str) -> Config:
    return dataclasses.replace(DEFAULT_CONFIG, **VARIANTS[name])


@dataclass
class GameSpec:
    """第 g 局怎么打：全部由 (seed, g) 决定，复现只要这两个数。"""

    seed: int
    g: int
    variant: str
    n_players: int
    origins: list[str | None]
    controllers: list[str]

    @property
    def gseed(self) -> int:
        return self.seed * 100000 + self.g


def game_spec(seed: int, g: int) -> GameSpec:
    rng = random.Random(f"bugbash/{seed}/{g}")
    names = list(VARIANTS)
    variant = rng.choices(names, [VARIANT_WEIGHTS.get(n, 1) for n in names])[0]
    cfg = variant_cfg(variant)
    n = rng.choice([3, 4, 5, 6, 6, 6])
    pool = cfg.origin_ids()
    if not pool:
        origins: list[str | None] = [None] * n
    elif rng.random() < 0.3:
        origins = [rng.choice(pool) for _ in range(n)]  # 真人局挑出身是放回抽样，可能撞
    else:
        origins = rng.sample(pool, n) if n <= len(pool) else [rng.choice(pool) for _ in range(n)]
    if rng.random() < 0.35:
        controllers = ["learned"] * n
    else:
        controllers = rng.choices(CONTROLLERS, CONTROLLER_WEIGHTS, k=n)
    return GameSpec(seed, g, variant, n, origins, controllers)


# --------------------------------------------------------------------------
# 出牌的人
# --------------------------------------------------------------------------

Decider = Callable[[Game, int], list[dict]]


def random_decider(rng: random.Random) -> Decider:
    """随机合法玩家：随机换牌、随机挑 0~2 张、随机顺序、随机目标。专撞 AI 不会打的边角。"""

    def decide(game: Game, pid: int) -> list[dict]:
        p = game.players[pid]
        while rng.random() < 0.15:
            info = game._redraw_info(p)
            if not (info["redraw_available"] and info["redraw_affordable"]):
                break
            game.redraw(pid)
        hand = game.hands[pid]
        others = [q for q in game.players if q != pid]
        k = rng.choice([0, 1, 2, 2, 2, 2])
        idxs = rng.sample(range(len(hand)), min(k, len(hand)))
        promos = [i for i, d in enumerate(hand) if d.card.is_promotion]
        if promos and rng.random() < 0.5 and not any(hand[i].card.is_promotion for i in idxs):
            # 多打晋升卡：不然随机玩家几乎升不了职，晋升那一大块就没人测
            j = rng.choice(promos)
            idxs = (idxs[:1] if idxs else []) + [j]
            rng.shuffle(idxs)
        picks: list[dict] = [{"index": i, "target": rng.choice(others) if hand[i].card.needs_target else None}
                             for i in idxs]
        if game._family_card_blocker(p) is None and rng.random() < 0.3:
            picks.insert(rng.randrange(len(picks) + 1), {"action": Card.PROMOTE_FAMILY.value})
        return picks

    return decide


def scripted_decider(rng: random.Random) -> Decider:
    name = rng.choice(sorted(analysis.STRATEGIES))
    strat = analysis.STRATEGIES[name]

    def decide(game: Game, pid: int) -> list[dict]:
        me = game.players[pid]
        others = [q for q in game.players if q != pid]
        return strat(game, me, game.hands[pid], others, rng) or []

    return decide


def ai_decider(pool: ai.AgentPool) -> Decider:
    return lambda game, pid: ai.turn(game, pid, pool)


def build_deciders(spec: GameSpec, game: Game) -> dict[int, Decider]:
    cfg = game.cfg
    learned = ai.make_pool(cfg, random.Random(f"{spec.gseed}/learned"))
    hand = ai.make_pool(dataclasses.replace(cfg, ai_policy=""), random.Random(f"{spec.gseed}/hand"))
    out: dict[int, Decider] = {}
    for pid, ctl in zip(sorted(game.players), spec.controllers):
        rng = random.Random(f"{spec.gseed}/{pid}/{ctl}")
        out[pid] = {"learned": lambda r: ai_decider(learned), "hand": lambda r: ai_decider(hand),
                    "scripted": scripted_decider, "random": random_decider}[ctl](rng)
    return out


# --------------------------------------------------------------------------
# 裁判：按写下来的规则独立重算一轮
# --------------------------------------------------------------------------


def ffloor(x: Fraction) -> int:
    return max(0, math.floor(x))


def fceil(x: Fraction) -> int:
    return math.ceil(x)


@dataclass
class RP:
    """裁判自己的玩家账本（一轮）。"""

    pid: int
    name: str
    origin: str | None
    money: int
    merit: int
    rank: int
    tenure: int
    warnings: int
    acts: list[Action] = field(default_factory=list)
    # 本轮累计
    work_merit: int = 0
    graft_merit: int = 0
    work_gains: list[int] = field(default_factory=list)
    dirty: int = 0
    laundered: int = 0
    money_gained: int = 0
    attackers: list[int] = field(default_factory=list)
    reporters: list[int] = field(default_factory=list)
    storm: bool = False
    promo: dict[str, Any] | None = None  # 本轮升成的那一级
    pending: list[dict[str, Any]] = field(default_factory=list)  # 要等举报分到钱才知道够不够的晋升卡
    tried_merit: bool = False
    caught: bool = False
    bribe_attempt: int = 0  # 本轮"花钱买官"（钱够的贿赂升职）的那笔
    voided_bribe: int = 0  # 被查实作废的那笔买官（当作没升过，钱看抄完赃款还够不够再没收）
    family_bribe: int = 0
    stolen: int = 0
    hat: int = 0
    from_attacks: int = 0
    overtime_merit: int = 0
    overtime_pay: int = 0
    confiscated: int = 0
    bribe_lost: int = 0
    reward: int = 0
    shield_fee: int = 0
    warnings_issued: int = 0
    demoted: bool = False
    skip: str = ""  # 写下来的规则判不了的情况：这一轮不和引擎比这个人
    tags: set[str] = field(default_factory=set)

    @property
    def merit_gained(self) -> int:
        return self.work_merit + self.graft_merit


class Referee:
    """一轮的独立重算。只读 cfg 的常量，不调用 rules.py。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.notes: list[tuple[str, str, int, dict]] = []  # (kind, sig, pid, detail)：口径类发现

    # ---- 规则里的数 ----
    def is_origin(self, rp: RP, oid: str) -> bool:
        return self.cfg.origins_enabled and rp.origin == oid

    def merit_cost(self, rp: RP) -> int:
        cfg = self.cfg
        base = cfg.promotion_merit_costs[rp.rank]
        if self.is_origin(rp, "OFFICIAL"):
            ratio = cfg.origin_patronage_merit_ratio
            if rp.rank == cfg.president_rank - 1 and cfg.origin_patronage_top_ratio is not None:
                ratio = cfg.origin_patronage_top_ratio
            return fceil(Fraction(base) * ratio)
        return base

    def money_cost(self, rp: RP) -> int:
        return self.cfg.promotion_money_costs[rp.rank]

    def top_step(self, rp: RP) -> bool:
        return rp.rank == self.cfg.president_rank - 1 and bool(self.cfg.promotion_requires_both[rp.rank])

    def wmul(self, rank: int) -> Fraction:
        return self.cfg.work_multiplier(rank)

    def mmul(self, rank: int) -> Fraction:
        return self.cfg.money_multiplier(rank)

    def blocked(self, rp: RP) -> bool:
        """政绩升职被攻击暂缓？贫农：开关 None = 谁都拦不住；n = 不超过 n 个人拦不住。"""
        if not rp.attackers:
            return False
        if self.is_origin(rp, "PEASANT"):
            n = self.cfg.origin_peasant_max_attackers
            return not (n is None or len(rp.attackers) <= n)
        return True

    def note(self, kind: str, sig: str, rp: RP, **detail) -> None:
        self.notes.append((kind, sig, rp.pid, detail))

    # ---- 晋升 ----
    def _promote(self, rp: RP, kind: str, pay_merit: bool, pay_money: bool, card: Card, pos: int,
                 money_src: str = "") -> None:
        cfg = self.cfg
        tc, mc = self.merit_cost(rp), self.money_cost(rp)
        cost_t = tc if pay_merit else 0
        cost_m = mc if pay_money else 0
        rp.merit -= cost_t
        rp.money -= cost_m
        before = rp.merit
        # 升完一级，剩下的政绩一律 ÷N（向上取整）——不管怎么升上去的
        if pay_merit or cfg.promotion_always_decays_merit:
            rp.merit = fceil(Fraction(max(0, rp.merit), cfg.merit_overflow_divisor)) if rp.merit > 0 else 0
        decay = before - rp.merit
        rp.rank += 1
        rp.promo = {"kind": kind, "cost_m": cost_m, "cost_t": cost_t, "decay": decay, "card": card.value,
                    "pos": pos, "bribe": pay_money and kind != "FAMILY" and money_src == "bribe",
                    "family": card is Card.PROMOTE_FAMILY}

    def play_promotion(self, rp: RP, card: Card, pos: int, reported_before: bool, loot: int,
                       void_bribes: bool = False) -> None:
        """晋升卡按出牌顺序结算（规则速查「出牌顺序」「升职的克制关系」）。

        loot = 排在这张前面、这一轮贪来的钱里查实会被抄走的那份。查实的人（void_bribes）
        花不到它："赃款当场抄走，剩下的钱不够就升不上去（调令照样用掉）"。"""
        cfg = self.cfg
        if rp.rank >= cfg.president_rank:
            return
        tc, mc = self.merit_cost(rp), self.money_cost(rp)
        merit_ok = rp.merit >= tc
        money_ok = rp.money >= mc
        if card.can_use_merit and merit_ok:
            rp.tried_merit = True  # "凭政绩升职"算干正事（戴帽子豁免），和这张卡最后用没用上无关
        if rp.promo is not None:
            return  # 一轮最多升一级
        if card is Card.PROMOTE_FAMILY:
            # 一纸调令：政绩优先、其次金钱；攻击和举报都拦不住；不能升主席
            if rp.rank + 1 >= cfg.president_rank:
                return
            if merit_ok:
                self._promote(rp, "MERIT", True, False, card, pos)
                rp.promo["family"] = True
                return
            if money_ok:
                if void_bribes and rp.money - loot < mc:
                    return
                self._promote(rp, "MONEY", False, True, card, pos)
                rp.promo["family"] = True
                rp.family_bribe = mc
                return
            if reported_before:
                rp.pending.append({"card": card, "pos": pos, "money_at": rp.money, "merit_at": rp.merit, "loot": loot})
            return
        blocked = self.blocked(rp)
        if self.top_step(rp):
            # 主席那一步：钱和政绩都要够；走哪条路由卡决定
            merit_route = card.can_use_merit and not blocked
            money_route = card.can_use_money and not merit_route
            if not (merit_route or money_route):
                return  # 政绩升职被攻击按住：暂缓
            if not merit_ok:
                return
            if not money_ok:
                if reported_before:
                    rp.pending.append({"card": card, "pos": pos, "money_at": rp.money, "merit_at": rp.merit, "loot": loot,
                                       "top": True, "bribe": money_route})
                return
            if void_bribes and merit_route and rp.money - loot < mc:
                return
            if money_route:
                rp.bribe_attempt = mc
                if void_bribes:  # 查实了：买官作废，当作没升过
                    rp.voided_bribe = mc
                    return
            self._promote(rp, "BOTH", True, True, card, pos, money_src="bribe" if money_route else "top")
            return
        if card.can_use_merit and merit_ok and not blocked:
            self._promote(rp, "MERIT", True, False, card, pos)
            return
        if card.can_use_money:
            if money_ok:
                rp.bribe_attempt = mc
                if void_bribes:
                    rp.voided_bribe = mc
                    return
                self._promote(rp, "MONEY", False, True, card, pos, money_src="bribe")
                return
            if reported_before:
                rp.pending.append({"card": card, "pos": pos, "money_at": rp.money, "merit_at": rp.merit, "loot": loot,
                                   "bribe": True})

    # ---- 一轮 ----
    def caught_set(self, players: dict[int, RP], eff: dict[str, Any]) -> set[int]:
        """谁会被查实：被真人举报或者被反腐风暴扫到，并且这一轮贪了钱 / 花钱买官。"""
        cfg = self.cfg
        ids = sorted(players)
        storm: set[int] = set()
        if eff.get("storm_report"):
            guilty = {pid: players[pid].dirty for pid in ids if players[pid].dirty > 0}
            if guilty:
                quota = fceil(Fraction(len(ids)) * cfg.event_storm_fraction)
                quota = max(1, min(quota, len(guilty)))
                cutoff = sorted(guilty.values(), reverse=True)[quota - 1]
                storm = {pid for pid, amt in guilty.items() if amt >= cutoff}
        if eff.get("mass_report"):
            storm = set(ids)
        out = set()
        for pid in ids:
            rp = players[pid]
            rp.storm = pid in storm
            if not (rp.reporters or rp.storm):
                continue
            bribed = cfg.report_catches_bribery and (rp.bribe_attempt > 0 or rp.family_bribe > 0)
            if rp.dirty > 0 or bribed:
                out.add(pid)
        return out

    def resolve(self, players: dict[int, RP], event: GameEvent, salaries: dict[int, int]) -> dict[str, Any]:
        """被查实的买官当作没发生过（"官会被撤回来"）：先按都升了算一遍，看谁会被查实，再按作废重算。"""
        trial = copy.deepcopy(players)
        self.produce(trial, event, set())
        caught = self.caught_set(trial, event.effects)
        self.produce(players, event, caught)
        # 反腐风暴点谁按第一遍（都没作废）的贪污额定，重算之后不改名单（README「出牌顺序」）
        for pid in players:
            players[pid].storm = trial[pid].storm
        return self.settle(players, event, salaries, caught)

    def produce(self, players: dict[int, RP], event: GameEvent, void: set[int]) -> None:
        cfg = self.cfg
        ids = sorted(players)
        eff = event.effects
        wb, wm = int(eff.get("work_bonus", 0)), Fraction(eff.get("work_multiplier", 1))
        mb, mm = int(eff.get("money_bonus", 0)), Fraction(eff.get("money_multiplier", 1))

        # A. 先记谁打谁（攻击、举报都要指定别人；同一个人对同一目标打两张只算一次）
        for pid in ids:
            for a in players[pid].acts:
                tgt = a.target_id
                if a.card is Card.ATTACK and tgt in players and tgt != pid and not eff.get("attack_disabled"):
                    if pid not in players[tgt].attackers:
                        players[tgt].attackers.append(pid)
                if a.card is Card.REPORT and tgt in players and tgt != pid:
                    if pid not in players[tgt].reporters:
                        players[tgt].reporters.append(pid)

        # B. 每个人按自己排的顺序一张一张结算
        for pid in ids:
            rp = players[pid]
            reported_before = False
            loot = 0
            for pos, a in enumerate(rp.acts):
                c = a.card
                if c is Card.WORK:
                    g = ffloor((Fraction(a.value) + wb) * wm * self.wmul(rp.rank))
                    rp.work_merit += g
                    rp.work_gains.append(g)
                    rp.merit += g - self.rob(rp)
                elif c in (Card.CORRUPT, Card.GRAFT):
                    cash = ffloor((Fraction(a.value) + mb) * mm * self.mmul(rp.rank))
                    rp.money += cash
                    rp.money_gained += cash
                    rp.dirty += cash
                    washed = int(Fraction(cash) * cfg.origin_accountant_launder_ratio) \
                        if self.is_origin(rp, "ACCOUNTANT") else 0
                    rp.laundered += washed
                    loot += cash - washed
                    if c is Card.GRAFT:
                        # 顺带的政绩 = 牌面 × 1/4，再走官职倍率。重点项目只加「埋头工作」的基础政绩
                        gm = ffloor(Fraction(math.floor(Fraction(a.value) * cfg.graft_merit_ratio))
                                    * self.wmul(rp.rank))
                        rp.graft_merit += gm
                        rp.merit += gm - self.rob(rp)
                elif c is Card.REPORT:
                    reported_before = True
                elif c.is_promotion:
                    self.play_promotion(rp, c, pos, reported_before, loot, void_bribes=pid in void)

    def rob(self, rp: RP) -> int:
        """攻击作用在被攻击者的每张工作牌上：挣到的政绩当场被抢走一份，晋升卡用不上，和他怎么排无关。
        一共被抢 = 这一轮埋头工作 + 以权谋私挣的 × 1/2，按攻击人数平分、取整（零头留给他自己）。
        返回这张牌被抢走多少（累计的减去之前已经抢走的）。"""
        n = len(rp.attackers)
        if not n:
            return 0
        total = math.floor(Fraction(rp.merit_gained) * self.cfg.attack_steal_fraction) // n * n
        cut = total - rp.stolen
        rp.stolen = total
        return cut

    def judge(self, players: dict[int, RP], caught: set[int], late: dict[int, tuple[str, int]],
              proceeds: dict[int, int]) -> None:
        """D. 举报 / 反腐风暴：没收、行贿没收、警告、分赃。
        late = 等赃款的晋升卡里算行贿的（"bribe" 作废、"family" 一纸调令官不撤）：玩家 -> (类型, 这一级的钱)；
        proceeds = 上一遍每人分到的赃款。"""
        cfg = self.cfg
        ids = sorted(players)
        # D. 举报 / 反腐风暴
        for pid in ids:
            players[pid].caught = pid in caught or pid in late
        pending_bribe = {pid: players[pid].voided_bribe for pid in ids if players[pid].voided_bribe}
        seize: dict[int, int] = {}
        for pid in ids:
            rp = players[pid]
            if not rp.caught or not cfg.report_reward_enabled:
                continue
            s = fceil(Fraction(rp.dirty) * cfg.report_seize_ratio)
            if rp.laundered:
                s = min(s, rp.dirty - rp.laundered)
            s = max(math.floor(Fraction(rp.dirty) * cfg.report_reward_ratio), s)
            seize[pid] = min(s, rp.money)
        pool_total: dict[int, int] = {}
        for pid in ids:
            rp = players[pid]
            if not rp.caught:
                continue
            taken = seize.get(pid, 0)
            rp.money -= taken
            rp.confiscated = taken
            pool_total[pid] = taken
            bribe = pending_bribe.get(pid, 0)
            if bribe and rp.money >= bribe:  # 抄完赃款还买得起，才算行贿没收
                lost = min(math.floor(Fraction(bribe) * cfg.bribe_forfeit_ratio), rp.money)
                rp.money -= lost
                rp.bribe_lost = lost
                pool_total[pid] += lost
            kind, cost = late.get(pid, ("", 0))
            if kind == "bribe" and not bribe:
                # 等赃款的买官：钱要靠分来的赃款才凑得齐（按上一遍分到的算）
                if rp.money + proceeds.get(pid, 0) >= cost:
                    lost = math.floor(Fraction(cost) * cfg.bribe_forfeit_ratio)
                    rp.money -= lost  # 分赃在后面才到账，这里可能暂时是负的
                    rp.bribe_lost = lost
                    pool_total[pid] += lost
            family_paid = rp.family_bribe + (cost if kind == "family" else 0)
            if family_paid and cfg.report_catches_bribery:
                pool_total[pid] += math.floor(Fraction(family_paid) * cfg.bribe_forfeit_ratio)
            major = rp.dirty >= cfg.major_corruption_threshold
            issued = max(1, cfg.major_corruption_warnings) if major else 1
            rp.warnings_issued = issued
            rp.warnings += issued
            rp.tenure = 0
            while rp.warnings >= cfg.warnings_before_demotion:
                rp.warnings -= cfg.warnings_before_demotion
                if self.is_origin(rp, "RED"):
                    ratio = cfg.origin_red_shield_fee
                    fee = 0 if (ratio is None or rp.rank <= 0) else fceil(
                        Fraction(cfg.promotion_money_costs[rp.rank - 1]) * ratio)
                    if not (cfg.origin_red_shield_fee_strict and rp.money < fee):
                        fee = min(fee, max(0, rp.money))
                        rp.money -= fee
                        rp.shield_fee += fee
                        continue
                if rp.rank > cfg.base_rank:
                    rp.rank -= 1
                rp.tenure = 0
                rp.demoted = True
        for pid in ids:
            rp = players[pid]
            if not (rp.caught and rp.reporters and pool_total.get(pid, 0) > 0):
                continue
            fam_cost = rp.family_bribe + (late[pid][1] if late.get(pid, ("",))[0] == "family" else 0)
            family = math.floor(Fraction(fam_cost) * cfg.bribe_forfeit_ratio) \
                if cfg.report_catches_bribery else 0
            gross = rp.dirty + rp.bribe_lost + family
            payout = min(pool_total[pid], math.floor(Fraction(gross) * cfg.report_reward_ratio))
            share = payout // len(rp.reporters) if cfg.report_reward_split_evenly else payout
            share = max(0, share - cfg.report_reward_fee)
            for aid in rp.reporters:
                players[aid].money += share
                players[aid].reward += share

        for pid in late:
            players[pid].money = max(0, players[pid].money)

    def late_bribes(self, base: dict[int, RP], caught: set[int], got: dict[int, int]) -> dict[int, tuple[str, int]]:
        """举报作用在每张牌上：等赃款的晋升卡要是（加上分来的赃款）买得起，被举报就算行贿。
        base = 举报结算之前的状态（官职按出牌那一刻）。"""
        out: dict[int, tuple[str, int]] = {}
        for pid, rp in base.items():
            if not (rp.reporters or rp.storm) or rp.promo is not None:
                continue
            for pend in rp.pending:
                avail = pend["money_at"] + got.get(pid, 0) - (pend["loot"] if pid in caught else 0)
                if avail < self.money_cost(rp):
                    continue
                if pend.get("top") and pend["merit_at"] < self.merit_cost(rp):
                    continue
                if pend.get("bribe"):
                    out[pid] = ("bribe", self.money_cost(rp))
                elif pend["card"] is Card.PROMOTE_FAMILY:
                    out[pid] = ("family", self.money_cost(rp))
                break
        return out

    def settle(self, players: dict[int, RP], event: GameEvent, salaries: dict[int, int],
               caught: set[int]) -> dict[str, Any]:
        cfg = self.cfg
        ids = sorted(players)
        eff = event.effects

        # C. 政治攻击（所有人的牌都结算完之后）
        for tid in ids:
            tgt = players[tid]
            if not tgt.attackers:
                continue
            # 抢功抢的是这一轮埋头工作和以权谋私挣的政绩（2026-10-17 定的口径，两份说明都改了）
            worked = any(a.card in (Card.WORK, Card.GRAFT) for a in tgt.acts)
            if worked and tgt.merit_gained > 0:
                # 被抢的那份第 B 步在他的工作牌上就扣下了（tgt.stolen），这里分给攻击者
                share = tgt.stolen // len(tgt.attackers)
                for aid in tgt.attackers:
                    players[aid].merit += share
                    players[aid].from_attacks += share
            elif not tgt.tried_merit:
                # 戴帽子：按他的官职扣 4 / 6 / 8 / 10（规则里是一张固定的表），攻击者各记 1 / 1 / 2 / 2
                pen = min(ffloor(Fraction(cfg.attack_merit_penalty) * self.wmul(tgt.rank)), tgt.merit)
                if pen > 0:
                    tgt.merit -= pen
                    tgt.hat += pen
                    for aid in tgt.attackers:
                        r = ffloor(Fraction(cfg.attack_hat_reward) * self.wmul(players[aid].rank))
                        players[aid].merit += r
                        players[aid].from_attacks += r

        # C2. 卷王加班：两张埋头工作的政绩在攻击之后 ×K；K 倍工资
        for pid in ids:
            rp = players[pid]
            if self.is_origin(rp, "GRINDER") and len(rp.work_gains) >= 2:
                net = max(0, sum(rp.work_gains[:2]) - rp.stolen)
                extra = (cfg.origin_grinder_overtime_multiplier - 1) * net
                rp.overtime_merit = extra
                rp.merit += extra
                rp.overtime_pay = cfg.origin_grinder_overtime_multiplier * cfg.rank_salary[rp.rank]
                rp.money += rp.overtime_pay

        # D. 举报 / 反腐风暴。等赃款的买官也逃不掉举报，但它买不买得起要看分到多少赃款，
        # 所以反复判到"哪些等赃款的卡算行贿"和每人分到的赃款都不再变
        base = {pid: copy.deepcopy(players[pid]) for pid in ids}
        late: dict[int, tuple[str, int]] = {}
        proceeds: dict[int, int] = {}
        for attempt in range(len(ids) + 2):
            self.judge(players, caught, late, proceeds)
            got = {pid: players[pid].reward for pid in ids}
            nxt = self.late_bribes(base, caught, got)
            if attempt == len(ids) + 1 or (nxt == late and (not late or got == proceeds)):
                break
            for pid in ids:
                players[pid] = copy.deepcopy(base[pid])
            late, proceeds = nxt, got

        # E. 排在举报后面、要花举报分来的钱的晋升卡
        for pid in ids:
            rp = players[pid]
            for pend in rp.pending:
                if rp.promo is not None or rp.demoted:
                    break
                # 那一刻的钱 + 分来的赃款；自己也被查实的话，那一刻手上的赃款已经抄走
                avail = pend["money_at"] + rp.reward - (pend["loot"] if rp.caught else 0)
                card = pend["card"]
                # 现在手上的钱也得够（比如被查实交了护身符的钱）
                if min(avail, rp.money) < self.money_cost(rp):
                    continue
                # 政绩用不上排在后面挣的，也已经被攻击扣过：那一刻够、现在也还够才行
                if pend.get("top") and min(pend["merit_at"], rp.merit) < self.merit_cost(rp):
                    continue
                if rp.caught and pend.get("bribe"):
                    continue  # 被查实了，花钱的晋升作废（一纸调令、主席那一步走政绩的不算买官）
                kind = "BOTH" if pend.get("top") else "MONEY"
                self._promote(rp, kind, kind == "BOTH", True, card, pend["pos"])
                rp.tags.add("reward_funded")
                if card is Card.PROMOTE_FAMILY:
                    rp.promo["family"] = True

        # F. 工龄：官职没变 +1；升了或者被查实清零；攒满自动升一级（升不到主席）
        for pid in ids:
            rp = players[pid]
            if rp.promo is not None:
                rp.tenure = 0
            elif rp.caught:
                rp.tenure = 0
            else:
                rp.tenure += 1
                ceiling = cfg.president_rank if cfg.tenure_can_reach_president else cfg.president_rank - 1
                if rp.tenure >= cfg.tenure_required and rp.rank < ceiling:
                    before = rp.merit
                    rp.merit = fceil(Fraction(rp.merit, cfg.merit_overflow_divisor)) if rp.merit > 0 else 0
                    rp.rank += 1
                    rp.tenure = 0
                    rp.promo = {"kind": "TENURE", "cost_m": 0, "cost_t": 0, "decay": before - rp.merit,
                                "card": "", "pos": -1, "bribe": False, "family": False}

        # G. 坊间传闻
        gossip: list[int] = []
        if cfg.gossip_illegal_first:
            illegal = {pid: (0 if players[pid].caught else players[pid].dirty - players[pid].laundered)
                       + players[pid].reward for pid in ids}
            # 规则："合法收入（工资、加班费、会计洗白的钱）"，查实了也算（洗白的那份抄不走）
            legal = {pid: salaries.get(pid, 0) + players[pid].overtime_pay + players[pid].laundered
                     for pid in ids}
            if any(v > 0 for v in illegal.values()):
                best = max((illegal[pid], legal[pid]) for pid in ids)
                gossip = [pid for pid in ids if (illegal[pid], legal[pid]) == best]
        else:
            income = {pid: salaries.get(pid, 0) + players[pid].overtime_pay
                      + (0 if players[pid].caught else players[pid].dirty) + players[pid].reward for pid in ids}
            extra = {pid: (0 if players[pid].caught else players[pid].dirty) + players[pid].reward for pid in ids}
            if any(v > 0 for v in extra.values()):
                top = max(income.values())
                gossip = [pid for pid in ids if income[pid] == top and top > 0]
        return {"gossip": gossip}


# --------------------------------------------------------------------------
# 对账
# --------------------------------------------------------------------------

FIELD_CATEGORY = [
    ("base_merit", "production"), ("money_gained", "production"), ("corrupt_amount", "production"),
    ("laundered", "production"),
    ("merit_stolen_by_attackers", "attack"), ("attack_merit_loss", "attack"), ("merit_from_attacks", "attack"),
    ("overtime_merit", "overtime"), ("overtime_pay", "overtime"),
    ("report_effective", "report"), ("money_confiscated", "report"), ("bribe_lost", "report"),
    ("warnings_issued", "report"), ("shield_fee_paid", "report"), ("money_from_reports", "report"),
    ("warnings_after", "report"),
    ("promotion", "promotion"), ("promotion_money_cost", "promotion"), ("promotion_merit_cost", "promotion"),
    ("promotion_merit_decay", "promotion"), ("rank_after", "promotion"),
    ("tenure_after", "tenure"), ("money_after", "total"), ("merit_after", "total"),
]


def engine_fields(o, warnings_now: int | None = None) -> dict[str, Any]:
    """引擎的结算结果。warnings_after 只在查实时才写，所以警告数取玩家结算后的实际值。"""
    return {
        "base_merit": o.merit_gained - o.overtime_merit, "money_gained": o.money_gained,
        "corrupt_amount": o.corrupt_amount, "laundered": o.laundered,
        "merit_stolen_by_attackers": o.merit_stolen_by_attackers, "attack_merit_loss": o.attack_merit_loss,
        "merit_from_attacks": o.merit_from_attacks, "overtime_merit": o.overtime_merit,
        "overtime_pay": o.overtime_pay, "report_effective": o.report_effective,
        "money_confiscated": o.money_confiscated, "bribe_lost": o.bribe_lost,
        "warnings_issued": o.warnings_issued, "shield_fee_paid": o.shield_fee_paid,
        "money_from_reports": o.money_from_reports,
        "warnings_after": o.warnings_after if warnings_now is None else warnings_now,
        "promotion": o.promotion.value, "promotion_money_cost": o.promotion_money_cost,
        "promotion_merit_cost": o.promotion_merit_cost, "promotion_merit_decay": o.promotion_merit_decay,
        "rank_after": o.rank_after, "tenure_after": o.tenure_after, "money_after": o.money_after,
        "merit_after": o.merit_after,
    }


def referee_fields(rp: RP) -> dict[str, Any]:
    pr = rp.promo or {}
    return {
        "base_merit": rp.merit_gained, "money_gained": rp.money_gained, "corrupt_amount": rp.dirty,
        "laundered": rp.laundered, "merit_stolen_by_attackers": rp.stolen,
        "attack_merit_loss": rp.stolen + rp.hat, "merit_from_attacks": rp.from_attacks,
        "overtime_merit": rp.overtime_merit, "overtime_pay": rp.overtime_pay,
        "report_effective": rp.caught, "money_confiscated": rp.confiscated, "bribe_lost": rp.bribe_lost,
        "warnings_issued": rp.warnings_issued, "shield_fee_paid": rp.shield_fee,
        "money_from_reports": rp.reward, "warnings_after": rp.warnings,
        "promotion": pr.get("kind", "NONE") if pr.get("kind") != "FAMILY" else "MONEY",
        "promotion_money_cost": pr.get("cost_m", 0), "promotion_merit_cost": pr.get("cost_t", 0),
        "promotion_merit_decay": pr.get("decay", 0), "rank_after": rp.rank, "tenure_after": rp.tenure,
        "money_after": rp.money, "merit_after": rp.merit,
    }


def play_signature(rp: RP, event_id: str) -> str:
    cards = ">".join(a.card.value for a in rp.acts) or "-"
    flags = "".join(f for f, on in (("A", bool(rp.attackers)), ("R", bool(rp.reporters)), ("S", rp.storm),
                                    ("C", rp.caught)) if on)
    return f"{cards}[{flags}]"


ZERO_RE = re.compile(r"(?<![\d.])0 点")


class Bash:
    """一局的裁判：挂在对局循环上，每轮前后各看一眼。"""

    def __init__(self, spec: GameSpec, game: Game):
        self.spec = spec
        self.game = game
        self.cfg = game.cfg
        self.findings: list[dict[str, Any]] = []
        self.stats: Counter = Counter()
        self.prev: dict[int, tuple[int, int]] = {}
        self.redraws_seen: Counter = Counter()
        self.candidates: list[int] = []  # 这一轮登顶的人（同轮多人登顶判谁当选之前）
        self._wrap_redraw()
        settle = game._settle_presidents

        def settle_wrapped(outcome):
            self.candidates = list(outcome.presidents)
            return settle(outcome)

        game._settle_presidents = settle_wrapped  # type: ignore[method-assign]

    # ---- 发现 ----
    def find(self, kind: str, sig: str, rnd: int, pid: int | None, **detail) -> None:
        self.findings.append({"kind": kind, "sig": sig, "seed": self.spec.seed, "g": self.spec.g,
                              "variant": self.spec.variant, "round": rnd, "pid": pid,
                              "detail": detail})

    # ---- 发牌 / 换牌 ----
    def _wrap_redraw(self) -> None:
        game, orig = self.game, self.game.redraw

        def redraw(pid: int) -> int:
            p = game.players[pid]
            used = game.redraw_count.get(pid, 0)
            rank, money = p.rank, p.money
            base = self.cfg.redraw_costs[rank] if rank < len(self.cfg.redraw_costs) else 0
            free = self.cfg.origin_rich_free_redraws if (self.cfg.origins_enabled and p.origin is Origin.RICH) else 0
            want = None if base <= 0 else (0 if used < free else base * self.cfg.redraw_cost_growth ** (used - free))
            try:
                cost = orig(pid)
            except GameError as e:
                if want is not None and money >= want:
                    self.find("deal:redraw_refused", "钱够却换不了牌", game.round_number, pid,
                              error=str(e), want=want, money=money)
                raise
            self.stats["redraws"] += 1
            if want is None or cost != want:
                self.find("deal:redraw_cost", "换牌价格和规则不一样", game.round_number, pid,
                          rank=rank, used=used, rich=bool(free), cost=cost, want=want)
            if game.players[pid].money != money - cost:
                self.find("deal:redraw_money", "换牌扣钱和报价不一样", game.round_number, pid,
                          cost=cost, before=money, after=game.players[pid].money)
            self.check_hand(pid, "redraw")
            return cost

        game.redraw = redraw  # type: ignore[method-assign]

    def check_hand(self, pid: int, when: str) -> None:
        cfg, hand = self.cfg, self.game.hands[pid]
        rnd = self.game.round_number
        if len(hand) != cfg.hand_size:
            self.find("deal:hand_size", "手牌张数不对", rnd, pid, n=len(hand), when=when)
        counts = Counter(d.card.value for d in hand)
        grinder = cfg.origins_enabled and self.game.players[pid].origin is Origin.GRINDER
        for name, n in counts.items():
            limit = cfg.card_deal_distribution.get(name, 0)
            if n > limit and not (grinder and name == "WORK" and n <= max(limit, cfg.origin_grinder_min_work)):
                self.find("deal:too_many", "同名牌比牌库里还多", rnd, pid, card=name, n=n, limit=limit)
        if grinder and counts.get("WORK", 0) < cfg.origin_grinder_min_work:
            self.find("deal:grinder_min", "卷王手里埋头工作不够保底", rnd, pid, n=counts.get("WORK", 0))
        dists = {"WORK": cfg.work_card_distribution, "CORRUPT": cfg.corrupt_card_distribution,
                 "GRAFT": cfg.graft_card_distribution}
        for d in hand:
            if d.card.value in dists:
                if d.value not in {v for v, _ in dists[d.card.value]}:
                    self.find("deal:value", "牌面点数不在分布里", rnd, pid, card=d.card.value, value=d.value)
            elif d.value != 0:
                self.find("deal:value", "非生产牌有点数", rnd, pid, card=d.card.value, value=d.value)

    # ---- 开局 ----
    def on_start(self) -> None:
        cfg = self.cfg
        for pid, p in self.game.players.items():
            rich = cfg.origins_enabled and p.origin is Origin.RICH
            # 第一轮工资已经发了：开局钱 = 富二代白拿的 + 第一轮工资
            want = (cfg.origin_old_money_start if rich else 0) + cfg.rank_salary[0]
            if p.money != want:
                self.find("deal:start_money", "开局钱不对（富二代白拿 + 第一轮工资）", 1, pid,
                          money=p.money, want=want, rich=rich)
            self.prev[pid] = (p.money - self.game.salary_paid.get(pid, 0), p.merit)

    def on_deal(self) -> None:
        game, cfg = self.game, self.cfg
        for pid, p in game.players.items():
            self.check_hand(pid, "deal")
            want = cfg.rank_salary[p.rank]
            if game.salary_paid.get(pid, 0) != want:
                self.find("deal:salary", "工资和官职对不上", game.round_number, pid,
                          paid=game.salary_paid.get(pid, 0), want=want, rank=p.rank)
        ev = game.next_event
        if ev is not None:
            w = {d["id"]: d["weight"] for d in cfg.event_definitions}
            if w.get(ev.id, 0) <= 0:
                self.find("deal:event", "抽到了权重 0 的事件", game.round_number, None, event=ev.id)

    # ---- 每轮 ----
    def before_resolve(self) -> dict[str, Any]:
        game = self.game
        self.candidates = []
        pre = {pid: dataclasses.replace(p) for pid, p in game.players.items()}
        previews = {pid: game._card_preview(p) for pid, p in game.players.items()}
        actions = {pid: list(sel.picks) for pid, sel in game.selections.items()}
        return {"pre": pre, "previews": previews, "actions": actions, "salaries": dict(game.salary_paid),
                "event": game.current_event}

    def after_resolve(self, ctx: dict[str, Any], outcome) -> None:
        game, cfg = self.game, self.cfg
        rnd = outcome.round_number
        names = {pid: p.name for pid, p in game.players.items()}
        self.stats["rounds"] += 1

        # ---- A. 流水账 ----
        for pid, o in outcome.outcomes.items():
            rows = o.ledger_lines(cfg.rank_name(o.rank_after))
            dm = sum(r["money"] for r in rows)
            dt = sum(r["merit"] for r in rows)
            m0, t0 = self.prev[pid]
            p = game.players[pid]
            if m0 + dm != p.money or t0 + dt != p.merit:
                self.find("ledger:mismatch", "流水账加起来和钱 / 政绩的变化对不上", rnd, pid,
                          money=[m0, dm, p.money], merit=[t0, dt, p.merit], rows=rows,
                          cards=[c.value for c in o.cards])
            if o.money_after != p.money and pid not in game.winners and not outcome.presidents:
                self.find("ledger:after", "结算记录的 money_after 和玩家实际的钱不一样", rnd, pid,
                          money_after=o.money_after, money=p.money)
            self.prev[pid] = (p.money, p.merit)

        # ---- B. 独立重算 ----
        ref = Referee(cfg)
        rps: dict[int, RP] = {}
        for pid, p in ctx["pre"].items():
            rps[pid] = RP(pid=pid, name=p.name, origin=p.origin.value if p.origin else None, money=p.money,
                          merit=p.merit, rank=p.rank, tenure=p.tenure, warnings=p.warnings,
                          acts=list(ctx["actions"].get(pid, [])))
        try:
            res = ref.resolve(rps, ctx["event"], ctx["salaries"])
        except Exception as e:  # 裁判自己崩了也要知道
            self.find("exception:referee", type(e).__name__, rnd, None, tb=traceback.format_exc(limit=6))
            return
        for kind, sig, pid, detail in ref.notes:
            self.find(kind, sig, rnd, pid, **detail)
        presidents_ref = [pid for pid in sorted(rps) if rps[pid].rank >= cfg.president_rank]
        all_diffs: dict[int, list[tuple[str, str, Any, Any]]] = {}
        ef_all: dict[int, dict[str, Any]] = {}
        for pid, rp in rps.items():
            o = outcome.outcomes[pid]
            if rp.skip:
                self.stats["skipped"] += 1
                continue
            self.stats["compared"] += 1
            ef, rf = engine_fields(o, game.players[pid].warnings), referee_fields(rp)
            if pid in self.candidates and pid not in game.winners:
                rf["rank_after"] = ef["rank_after"]  # 同轮多人登顶落选，下面 end 那段单独查
            all_diffs[pid] = [(f, cat, ef[f], rf[f]) for f, cat in FIELD_CATEGORY if ef[f] != rf[f]]
            ef_all[pid] = ef
        downstream = {"merit_from_attacks", "money_from_reports", "money_after", "merit_after"}
        for pid, rp in rps.items():
            diffs = all_diffs.get(pid)
            if diffs and {f for f, *_ in diffs} <= downstream:
                hit = [t for t in rps if (all_diffs.get(t) or rps[t].skip) and t != pid
                       and (pid in rps[t].attackers or pid in rps[t].reporters)]
                if hit:
                    # 只是被打 / 被举报的那个人算得不一样，连带出来的
                    self.stats["cascade"] += 1
                    continue
            o = outcome.outcomes[pid]
            if diffs:
                first, cat = diffs[0][0], diffs[0][1]
                sig = f"{first}|{play_signature(rp, ctx['event'].id)}"
                # 升职当场 ÷N，在攻击之前（2026-10-17 定的口径：按出牌顺序）
                kind = f"resolve:{cat}"
                self.find(kind, sig, rnd, pid, event=ctx["event"].id, origin=rp.origin,
                          diffs={f: {"engine": e, "rules": r} for f, _, e, r in diffs},
                          pre=_pstate(ctx["pre"][pid]), tags=sorted(rp.tags),
                          acts=[(a.card.value, a.target_id, a.value) for a in rp.acts],
                          attackers=rp.attackers, reporters=rp.reporters, storm=rp.storm,
                          notes=list(o.private_notes))
        # 坊间传闻（有人算得不一样、或者有人判不了的时候，传闻也会跟着不一样，不单记）
        money_fields = {"corrupt_amount", "laundered", "report_effective", "money_from_reports", "overtime_pay"}
        upstream = any(rp.skip for rp in rps.values()) or any(
            f in money_fields for d in all_diffs.values() for f, *_ in d)
        if upstream and sorted(outcome.wealth_top_ids) != sorted(res["gossip"]):
            self.stats["cascade"] += 1
        elif sorted(outcome.wealth_top_ids) != sorted(res["gossip"]):
            self.find("resolve:gossip", "坊间传闻点名的人和规则不一样", rnd, None,
                      engine=outcome.wealth_top_ids, rules=res["gossip"], event=ctx["event"].id,
                      players={pid: {"dirty": rp.dirty, "laundered": rp.laundered, "caught": rp.caught,
                                     "reward": rp.reward, "salary": ctx["salaries"].get(pid, 0),
                                     "overtime_pay": rp.overtime_pay} for pid, rp in rps.items()})

        # ---- C. 公报 / 提示 ----
        self.check_messages(outcome, rps, names)

        # ---- D. 晋升卡提示 ----
        for pid, rp in rps.items():
            acts = ctx["actions"].get(pid, [])
            if not acts or not acts[0].card.is_promotion or acts[0].card is Card.PROMOTE_FAMILY:
                continue
            if rp.attackers or rp.reporters or rp.storm or rp.caught or rp.skip \
                    or outcome.outcomes[pid].report_effective:
                continue
            pv = ctx["previews"][pid].get(acts[0].card.value)
            if pv is None:
                continue
            o = outcome.outcomes[pid]
            if sum(1 for a in acts if a.card.is_promotion) != 1:
                continue
            ok = o.promotion in (PromotionKind.MERIT, PromotionKind.MONEY, PromotionKind.BOTH)
            self.stats["preview_checked"] += 1
            if bool(pv["usable"]) != ok:
                self.find("preview:mismatch", f"{acts[0].card.value} 提示{'可用' if pv['usable'] else '不可用'}，"
                          f"实际{'升了' if ok else '没升'}", rnd, pid, preview=pv, pre=_pstate(ctx["pre"][pid]),
                          promotion=o.promotion.value, notes=list(o.private_notes))

        # ---- E. 终局 ----
        self.check_end(outcome, sorted(self.candidates))

        # ---- F. 存盘 / 隐私 ----
        self.check_snapshot(rnd)
        self.check_privacy(rnd)

    def check_messages(self, outcome, rps: dict[int, RP], names: dict[int, str]) -> None:
        cfg, game = self.cfg, self.game
        rnd = outcome.round_number
        msgs = list(outcome.public_messages)
        for m in msgs:
            if ZERO_RE.search(m):
                self.find("msg:zero", "公报里出现 '0 点'", rnd, None, msg=m)
        by_name = {v: k for k, v in names.items()}

        def about(m: str) -> int | None:
            # 消息里最先出现的那个名字
            hits = [(m.find(n), pid) for n, pid in by_name.items() if n in m]
            return min(hits)[1] if hits else None

        for pid, o in outcome.outcomes.items():
            name = names[pid]
            mine = [m for m in msgs if about(m) == pid]
            # 公报按发生先后播：把这个人的升、降一条条连起来，每一步都要接得上，最后落在实际官职上
            cur = o.rank_before
            for m in mine:
                hit = re.search(r"(?:晋升为|调任)(\S+?)。", m)
                if hit and "落选" not in m:
                    if hit.group(1) != cfg.rank_name(cur + 1):
                        self.find("msg:promotion_wrong", "公报说升到的官职接不上前面的官职", rnd, pid, msg=m,
                                  rank=[o.rank_before, o.rank_after], msgs=mine)
                    cur += 1
                hit = re.search(r"由(\S+?)降为(\S+?)。", m)
                if hit:
                    if (hit.group(1), hit.group(2)) != (cfg.rank_name(cur), cfg.rank_name(cur - 1)):
                        self.find("msg:demotion_wrong", "公报说的降级接不上前面的官职", rnd, pid, msg=m,
                                  rank=[o.rank_before, o.rank_after], msgs=mine)
                    cur -= 1
            if cur != o.rank_after:
                self.find("msg:rank_story", "公报里的升降连起来和实际官职对不上", rnd, pid, msgs=mine,
                          rank=[o.rank_before, o.rank_after])
            for m in mine:
                route = {"政绩卓著": "MERIT", "四处打点": "MONEY", "两手都硬": "BOTH", "按工龄": "TENURE"}
                for word, kind in route.items():
                    if word in m and o.promotion.value != kind and name in m.split(word)[0]:
                        self.find("msg:promotion_route", "公报里写的升职方式和实际不一样", rnd, pid, msg=m,
                                  promotion=o.promotion.value)
                hit = re.search(r"赃款 (\d+) 全部没收", m)
                if hit and int(hit.group(1)) != o.money_confiscated:
                    self.find("msg:confiscated", "公报里没收的数和实际不一样", rnd, pid, msg=m,
                              confiscated=o.money_confiscated)
                hit = re.search(r"行贿的 (\d+) 打了水漂", m)
                if hit and int(hit.group(1)) != o.bribe_lost:
                    self.find("msg:bribe", "公报里行贿打水漂的数和实际不一样", rnd, pid, msg=m, bribe_lost=o.bribe_lost)
                hit = re.search(r"再记 (\d+) 次就要降级", m)
                if hit and int(hit.group(1)) != cfg.warnings_before_demotion - o.warnings_after:
                    self.find("msg:warnings", "公报里'再记几次降级'和实际警告数对不上", rnd, pid, msg=m,
                              warnings_after=o.warnings_after)
                if "【穿小鞋】" in m and "本要凭政绩升职" in m and (o.promotion is not PromotionKind.NONE
                                                                  and o.promotion is not PromotionKind.TENURE):
                    self.find("msg:smear_but_promoted", "公报说升职暂缓，实际升上去了", rnd, pid, msg=m,
                              promotion=o.promotion.value)
            # 被抢功 / 戴帽子的数字
            for m in msgs:
                if f"{name} 埋头苦干" in m:
                    hit = re.search(r"抢走 (\d+) 点政绩", m)
                    if hit and int(hit.group(1)) != o.merit_stolen_by_attackers:
                        self.find("msg:stolen", "公报里被抢的数和实际不一样", rnd, pid, msg=m,
                                  stolen=o.merit_stolen_by_attackers)
                if f"【戴帽子】{name} " in m:
                    hit = re.search(r"扣掉 (\d+) 点政绩", m)
                    hat = o.attack_merit_loss - o.merit_stolen_by_attackers - o.merit_wiped_by_attack
                    if hit and int(hit.group(1)) != hat:
                        self.find("msg:hat", "公报里戴帽子扣的数和实际不一样", rnd, pid, msg=m, hat=hat)
            for note in o.private_notes:
                hit = re.search(r"分得赃款 (\d+)", note)
                if hit and int(hit.group(1)) != o.money_from_reports:
                    self.find("msg:reward_note", "私密提示里分到的赃款和实际不一样", rnd, pid, note=note,
                              reward=o.money_from_reports)
            got = sum(int(x) for n in o.private_notes for x in re.findall(r"抢到了 (\d+) 点政绩", n))
            hats = sum(int(x) for n in o.private_notes for x in re.findall(r"你记了 (\d+) 点功", n))
            if got + hats != o.merit_from_attacks:
                self.find("msg:attack_note", "私密提示里抢到 / 记功的数加起来和实际不一样", rnd, pid,
                          notes=list(o.private_notes), merit_from_attacks=o.merit_from_attacks)
            if o.promotion is PromotionKind.NONE and o.promotion_card_played and not o.private_notes \
                    and not o.promotion_frozen_by_report:
                self.find("msg:silent_promo_fail", "打了晋升卡没升上去，私密提示一句没说", rnd, pid,
                          cards=[c.value for c in o.cards])

    def check_end(self, outcome, presidents_ref: list[int]) -> None:
        game, cfg = self.game, self.cfg
        rnd = outcome.round_number
        cands = sorted(set(presidents_ref) | set(outcome.presidents))
        if cands:
            if not game.is_over:
                self.find("end:no_stop", "有人当上主席游戏没结束", rnd, None, presidents=cands)
            if len(cands) > 1 and cfg.president_tiebreak:
                key = {pid: (game.players[pid].money, game.players[pid].merit) for pid in cands}
                best = max(key.values())
                want = sorted(pid for pid in cands if key[pid] == best)
                if sorted(game.winners) != want:
                    self.find("end:tiebreak", "同轮多人登顶，当选的人和规则（先比钱再比政绩）不一样", rnd, None,
                              winners=game.winners, want=want, key=key)
                for pid in cands:
                    if pid not in want and game.players[pid].rank != cfg.president_rank - 1:
                        self.find("end:loser_rank", "同轮登顶落选的人没退回省级", rnd, pid,
                                  rank=game.players[pid].rank)
                if len(want) < len(cands) and not any("主席只有一个" in m for m in outcome.public_messages):
                    self.find("msg:tiebreak_missing", "同轮多人登顶，公报没说谁当选", rnd, None,
                              msgs=outcome.public_messages)
            elif sorted(game.winners) != cands:
                self.find("end:winner", "当上主席的人不是赢家", rnd, None, winners=game.winners, presidents=cands)
        elif rnd >= cfg.max_rounds:
            if not game.is_over:
                self.find("end:no_stop", f"打满 {cfg.max_rounds} 轮游戏没结束", rnd, None)
            # 规则：先比金钱、再比官职、最后比政绩
            key = {pid: (p.money, p.rank, p.merit) for pid, p in game.players.items()}
            best = max(key.values())
            want = sorted(pid for pid in key if key[pid] == best)
            if sorted(game.winners) != want:
                self.find("end:timeout", "打满轮数的赢家和规则（钱 > 官职 > 政绩）不一样", rnd, None,
                          winners=game.winners, want=want, key=key)
            standing = game.final_standing()
            order = sorted(key, key=lambda pid: key[pid], reverse=True)
            if [key[p] for p in standing] != [key[p] for p in order]:
                self.find("end:standing", "终局名次和规则排序不一样", rnd, None, standing=standing, want=order)
        elif game.is_over:
            self.find("end:early", "没人登顶也没打满轮数，游戏却结束了", rnd, None)
        if game.is_over and game.winners and game.final_standing()[: len(game.winners)] != \
                [p for p in game.final_standing() if p in game.winners]:
            self.find("end:standing_winner", "赢家没排在名次最前面", rnd, None,
                      standing=game.final_standing(), winners=game.winners)

    def check_snapshot(self, rnd: int) -> None:
        game = self.game
        try:
            snap = game.to_snapshot()
            g2 = Game.from_snapshot(json.loads(json.dumps(snap)), cfg=game.cfg, rng=random.Random(0))
            snap2 = g2.to_snapshot()
        except Exception as e:
            self.find("snapshot:error", type(e).__name__, rnd, None, tb=traceback.format_exc(limit=6))
            return
        a, b = json.loads(json.dumps(snap)), snap2
        # results 是"本次进程里结算过的轮次"，读档后只剩读档之后的（storage 按轮次 upsert，不会丢）
        a.pop("results", None)
        for k in a:
            if a[k] != b.get(k):
                self.find("snapshot:diff", f"存盘再读出来 {k} 变了", rnd, None, key=k)
        pa, pb = game.public_state(), g2.public_state()
        for k in pa:
            if json.dumps(pa[k], sort_keys=True, default=str) != json.dumps(pb.get(k), sort_keys=True, default=str):
                self.find("snapshot:public", f"存盘再读出来公开状态 {k} 变了", rnd, None, key=k)

    SECRET_KEYS = {"money", "hand", "picks", "corrupt_amount", "money_confiscated", "money_from_reports",
                   "money_after", "token", "laundered", "bribe_lost", "shield_fee_paid", "targets", "cards"}

    def check_privacy(self, rnd: int) -> None:
        state = self.game.public_state()
        bad: list[str] = []

        def walk(x: Any, path: str) -> None:
            if isinstance(x, dict):
                for k, v in x.items():
                    p = f"{path}.{k}"
                    if k in self.SECRET_KEYS and not path.startswith(".promotion_costs") \
                            and not path.startswith(".reveal"):
                        bad.append(p)
                    walk(v, p)
            elif isinstance(x, list):
                for i, v in enumerate(x):
                    walk(v, path)

        walk(state, "")
        if not self.game.is_over and state.get("reveal"):
            bad.append(".reveal（游戏没结束）")
        for p in sorted(set(bad)):
            self.find("privacy:key", f"公开状态里有 {p}", rnd, None, path=p)


def _pstate(p) -> dict[str, Any]:
    return {"money": p.money, "merit": p.merit, "rank": p.rank, "tenure": p.tenure, "warnings": p.warnings,
            "origin": p.origin.value if p.origin else None}


# --------------------------------------------------------------------------
# 跑一局
# --------------------------------------------------------------------------


def play(spec: GameSpec, stop_round: int | None = None) -> tuple[Bash, dict[str, Any] | None]:
    """打一局，全程挂着裁判。stop_round：复现时打到这一轮结算完就停，返回那一轮的上下文。"""
    cfg = variant_cfg(spec.variant)
    game = Game(game_id="bb", cfg=cfg, rng=random.Random(f"{spec.gseed}/start"))
    for name in T.NAMES[: spec.n_players]:
        game.add_player(name, is_ai=True)
    for pid, oid in zip(sorted(game.players), spec.origins):
        game.players[pid].origin = Origin(oid) if oid else None
    T.reseed_round(game, spec.gseed)
    bash = Bash(spec, game)
    game.start_game()
    bash.on_start()
    bash.on_deal()
    deciders = build_deciders(spec, game)
    last_ctx = None
    while not game.is_over:
        rnd = game.round_number
        for pid in sorted(game.players):
            try:
                picks = deciders[pid](game, pid) or []
            except Exception as e:
                bash.find("exception:decider", f"{spec.controllers[pid - 1]}: {type(e).__name__}: {e}"[:120],
                          rnd, pid, tb=traceback.format_exc(limit=8))
                picks = []
            if not game.selections[pid].picks:
                try:
                    game.select_actions(pid, picks)
                except GameError as e:
                    bash.find("exception:illegal_pick", f"{spec.controllers[pid - 1]} 出了非法牌：{e}", rnd, pid,
                              picks=[{k: (v.value if isinstance(v, Card) else v) for k, v in p.items()}
                                     for p in picks], hand=[d.view() for d in game.hands[pid]])
                    game.select_actions(pid, [])
            game.lock_action(pid)
        game.force_lock_all()
        game.reveal_event()
        ctx = bash.before_resolve()
        try:
            outcome = game.resolve()
        except Exception as e:
            bash.find("exception:resolve", f"{type(e).__name__}: {e}"[:120], rnd, None,
                      tb=traceback.format_exc(limit=10),
                      actions={pid: [(a.card.value, a.target_id, a.value) for a in acts]
                               for pid, acts in ctx["actions"].items()})
            return bash, None
        bash.after_resolve(ctx, outcome)
        ctx["outcome"] = outcome
        if stop_round is not None and rnd >= stop_round:
            return bash, ctx
        last_ctx = ctx
        if not game.is_over:
            T.reseed_round(game, spec.gseed)
            game.advance_round()
            bash.on_deal()
    bash.stats["games"] += 1
    return bash, last_ctx


def scenario(players: list[dict[str, Any]], picks: dict[int, list[tuple]], event_id: str = "CALM",
             cfg: Config = DEFAULT_CONFIG) -> tuple[Bash, Any]:
    """摆一个局面打一轮（测试用）：players[i] = {origin, money, merit, rank, tenure, warnings}，
    picks[pid] = [(牌, 点数, 目标), ...]（按出牌顺序；手牌就是这几张，不够 6 张补埋头工作）。"""
    from models import DealtCard
    import rules

    game = Game(game_id="sc", cfg=cfg, rng=random.Random(0))
    for i, _ in enumerate(players):
        game.add_player(T.NAMES[i], is_ai=True)
    game.start_game()
    spec = GameSpec(0, 0, "base", len(players), [p.get("origin") for p in players], ["random"] * len(players))
    bash = Bash(spec, game)
    for pid, st in zip(sorted(game.players), players):
        p = game.players[pid]
        p.origin = Origin(st["origin"]) if st.get("origin") else None
        p.money, p.merit, p.rank = st.get("money", 0), st.get("merit", 0), st.get("rank", 0)
        p.tenure, p.warnings = st.get("tenure", 0), st.get("warnings", 0)
        game.salary_paid[pid] = 0
        bash.prev[pid] = (p.money, p.merit)
        mine = picks.get(pid, [])
        hand = [DealtCard(Card(c), v) for c, v, _ in mine]
        hand += [DealtCard(Card.WORK, 6)] * (cfg.hand_size - len(hand))
        game.hands[pid] = hand
        game.select_actions(pid, [{"index": i, "action": c, "target": t} for i, (c, _, t) in enumerate(mine)])
        game.lock_action(pid)
    game.next_event = rules.event_by_id(event_id, cfg)
    game.reveal_event()
    ctx = bash.before_resolve()
    outcome = game.resolve()
    bash.after_resolve(ctx, outcome)
    return bash, outcome


def run_chunk(args: tuple[int, list[int]]) -> dict[str, Any]:
    seed, games = args
    findings: list[dict[str, Any]] = []
    stats: Counter = Counter()
    for g in games:
        spec = game_spec(seed, g)
        try:
            bash, _ = play(spec)
        except Exception as e:  # 对局循环本身崩了
            findings.append({"kind": "exception:game", "sig": f"{type(e).__name__}: {e}"[:120], "seed": seed,
                             "g": g, "variant": spec.variant, "round": None, "pid": None,
                             "detail": {"tb": traceback.format_exc(limit=10)}})
            continue
        findings.extend(bash.findings)
        stats.update(bash.stats)
        stats[f"variant:{spec.variant}"] += 1
    return {"games": games, "findings": findings, "stats": dict(stats)}


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------


KEEP = 20


def summarize(findings: list[dict[str, Any]], stats: Counter, out: Path, counts: Counter | None = None,
              examples: int = 3) -> str:
    """counts：每种情形（kind\tsig）一共出现几次（文件里只存了前 KEEP 个例子）。"""
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in findings:
        by_kind[f["kind"]].append(f)
    if counts is None:
        counts = Counter(f"{f['kind']}\t{f['sig']}" for f in findings)
    total = Counter()
    sig_total: dict[str, Counter] = defaultdict(Counter)
    for key, n in counts.items():
        k, sig = key.split("\t", 1)
        total[k] += n
        sig_total[k][sig] += n
    lines = [f"# Bug bash 结果（{stats.get('games', 0)} 局，{stats.get('rounds', 0)} 轮）", "",
             f"逐人对账 {stats.get('compared', 0)} 次，规则判不了跳过 {stats.get('skipped', 0)} 次；"
             f"晋升卡提示核对 {stats.get('preview_checked', 0)} 次；换牌 {stats.get('redraws', 0)} 次。", "",
             "| 类 | 条数 | 不同情形 |", "|---|---|---|"]
    order = sorted(total, key=lambda k: (k.split(":")[0] == "spec", -total[k]))
    for k in order:
        lines.append(f"| {k} | {total[k]} | {len(sig_total[k])} |")
    for k in order:
        lines += ["", f"## {k}（{total[k]} 条）", ""]
        sigs = sig_total[k]
        for sig, n in sigs.most_common(12):
            lines.append(f"- **{n}×** `{sig}`")
            for f in [f for f in by_kind[k] if f["sig"] == sig][:examples]:
                d = json.dumps(f["detail"], ensure_ascii=False, default=str)
                lines.append(f"  - 编号 {f['id']}：seed {f['seed']} 第 {f['g']} 局（{f['variant']}）"
                             f"第 {f['round']} 轮 玩家 {f['pid']}：{d[:600]}")
        if len(sigs) > 12:
            lines.append(f"- ……另有 {len(sigs) - 12} 种情形")
    md = "\n".join(lines)
    (out / "findings.md").write_text(md, encoding="utf-8")
    return md


def load_state(out: Path) -> dict[str, Any]:
    f = out / "state.json"
    return json.loads(f.read_text()) if f.exists() else {"done": [], "stats": {}, "next_id": 0, "seed": None,
                                                         "counts": {}}


def run(out: Path, seed: int, games: int | None, hours: float | None, workers: int, chunk: int = 50) -> Counter:
    out.mkdir(parents=True, exist_ok=True)
    state = load_state(out)
    if state["seed"] is not None:
        seed = state["seed"]
    state["seed"] = seed
    done = set(state["done"])
    stats = Counter(state["stats"])
    next_id = state["next_id"]
    # 同一种情形（kind + sig）只存前 KEEP 个例子，其余只计数——一秒一百多局，全存的话一晚上几千万行
    counts: Counter = Counter(state.get("counts", {}))
    findings_path = out / "findings.jsonl"
    deadline = time.time() + hours * 3600 if hours else None
    total_chunks = (games + chunk - 1) // chunk if games else None

    def chunks():
        c = 0
        while total_chunks is None or c < total_chunks:
            if c not in done:
                lo = c * chunk
                hi = lo + chunk if games is None else min(games, lo + chunk)
                yield c, list(range(lo, hi))
            c += 1

    gen = chunks()
    last_summary = last_save = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        running = {}

        def submit():
            if deadline and time.time() > deadline:
                return False
            try:
                c, gs = next(gen)
            except StopIteration:
                return False
            running[ex.submit(run_chunk, (seed, gs))] = c
            return True

        for _ in range(workers * 2):
            if not submit():
                break
        while running:
            fut = next(as_completed(list(running)))
            c = running.pop(fut)
            res = fut.result()
            with open(findings_path, "a", encoding="utf-8") as fh:
                for f in res["findings"]:
                    key = f"{f['kind']}\t{f['sig']}"
                    counts[key] += 1
                    if counts[key] > KEEP:
                        continue
                    f["id"] = next_id
                    next_id += 1
                    fh.write(json.dumps(f, ensure_ascii=False, default=str) + "\n")
            stats.update(res["stats"])
            done.add(c)
            submit()
            # 存盘别太勤：每秒写十几次几百 KB，杀毒软件跟着扫，整台机器会被拖死
            if time.time() - last_save > 30 or not running:
                state.update(done=sorted(done), stats=dict(stats), next_id=next_id, counts=dict(counts))
                (out / "state.json").write_text(json.dumps(state))
                last_save = time.time()
            if time.time() - last_summary > 600:
                summarize(load_findings(out), stats, out, counts)
                last_summary = time.time()
    summarize(load_findings(out), stats, out, counts)
    return stats


def load_findings(out: Path) -> list[dict[str, Any]]:
    f = out / "findings.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]


def repro(out: Path, fid: int) -> None:
    f = next((x for x in load_findings(out) if x["id"] == fid), None)
    if f is None:
        raise SystemExit(f"没有编号 {fid}")
    spec = game_spec(f["seed"], f["g"])
    print(f"# 编号 {fid}：{f['kind']}  {f['sig']}")
    print(f"规则变体 {spec.variant}，{spec.n_players} 人，出身 {spec.origins}，出牌的人 {spec.controllers}")
    print(json.dumps(f["detail"], ensure_ascii=False, indent=1, default=str))
    rnd = f["round"] or 1
    bash, ctx = play(spec, stop_round=rnd)
    if ctx is None or "outcome" not in ctx:
        print("（没能重放到这一轮）")
        return
    o = ctx["outcome"]
    print(f"\n## 第 {rnd} 轮  事件 {o.event.name}")
    for pid, p in sorted(ctx["pre"].items()):
        acts = [(a.card.value, a.target_id, a.value) for a in ctx["actions"].get(pid, [])]
        oc = o.outcomes[pid]
        print(f"- {pid} {p.name}（{p.origin.value if p.origin else '-'}）结算前 {_pstate(p)}  出牌 {acts}")
        print(f"    引擎：{engine_fields(oc)}")
        print(f"    私密提示：{oc.private_notes}")
    print("\n公报：")
    for m in o.public_messages + o.wealth_broadcast:
        print("  " + m)
    again = [x for x in bash.findings if x["round"] == rnd and x["kind"] == f["kind"]]
    print(f"\n重放这一轮又查出 {len(again)} 条同类发现" + ("（复现成功）" if again else "（没复现！）"))


def main(argv: list[str] | None = None) -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":
        # AI 里有按 set 遍历的地方：固定哈希种子，同一个 (seed, 局号) 才能原样重放
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.argv])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=None)
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--resume", default="")
    ap.add_argument("--repro", nargs=2, metavar=("目录", "编号"))
    args = ap.parse_args(argv)
    if args.repro:
        repro(Path(args.repro[0]), int(args.repro[1]))
        return
    if not args.games and not args.hours:
        args.games = 200
    out = Path(args.resume or args.out or ROOT / "search_runs" / "bugbash" / time.strftime("%Y%m%d-%H%M%S"))
    print(f"输出目录：{out}")
    stats = run(out, args.seed, args.games, args.hours, args.workers)
    print((out / "findings.md").read_text(encoding="utf-8")[:4000])
    print(f"\n{stats.get('games', 0)} 局跑完，详见 {out / 'findings.md'}")


if __name__ == "__main__":
    main()
