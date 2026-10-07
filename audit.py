"""AI 体检：AI 出牌时心里的预测 vs 真实结果，外加明显失误盘点、简单打法能不能剥削它。

    python3 audit.py --games 3000            # 六个思考型 AI、六身份混战
    python3 audit.py --games 3000 --exploit 2000
    python3 audit.py --duel endgame_scale=0.1 --duel-games 4000   # 权重对决
    python3 audit.py --vs-old /path/to/old/ai.py               # 新旧 AI 同桌
    python3 audit.py --features 400 [--bots]                       # 读牌特征表

判断 AI 笨不笨的尺子：一个足够聪明的 AI，多一个选项永远不会更差。所以

  1. 校准：它估"贪了会被查实 30%"，实际是不是 30%？估低了就会贪得太多。
  2. 失误：资源够、手里有卡却没升；晋升卡打出去怎么都不可能成；有人当轮登顶而我手里的
     干扰牌没往他身上打……这些不用算期望值，一眼就是错的。
  3. 剥削：一个死板的脚本打法（只干活、只贪污……）坐进五个 AI 中间，要是能稳定赢过公平线，
     说明 AI 有漏洞可钻。
"""

from __future__ import annotations

import argparse
import dataclasses
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ai  # noqa: E402
import analysis  # noqa: E402
from config import DEFAULT_CONFIG, Config  # noqa: E402

# 体检里说"手写 AI / 思考型 AI"就是手写的那个：不跟着 cfg.ai_policy 切到学习型
HAND_CFG = dataclasses.replace(DEFAULT_CONFIG, ai_policy="")
from game import Game  # noqa: E402
from models import Card, Origin, PromotionKind  # noqa: E402

N = 6
BUCKETS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.01]


def bucket(p: float) -> str:
    for lo, hi in zip(BUCKETS, BUCKETS[1:]):
        if p < hi:
            return f"{lo:.1f}-{min(hi, 1.0):.1f}"
    return "1.0"


class Tally:
    """可以跨进程合并的计数器集合。"""

    def __init__(self) -> None:
        # 校准：key -> [预测概率之和, 实际发生次数, 样本数]
        self.cal: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
        self.count: Counter = Counter()
        self.examples: dict[str, list[str]] = defaultdict(list)

    def cal_add(self, key: str, p: float, hit: bool) -> None:
        row = self.cal[key]
        row[0] += p
        row[1] += 1.0 if hit else 0.0
        row[2] += 1.0

    def example(self, kind: str, text: str, limit: int = 4) -> None:
        if len(self.examples[kind]) < limit:
            self.examples[kind].append(text)

    def merge(self, other: "Tally") -> None:
        for k, row in other.cal.items():
            mine = self.cal[k]
            for i in range(3):
                mine[i] += row[i]
        self.count.update(other.count)
        for k, lst in other.examples.items():
            room = 8 - len(self.examples[k])
            self.examples[k].extend(lst[:max(0, room)])

    def to_plain(self) -> dict[str, Any]:
        return {"cal": dict(self.cal), "count": dict(self.count), "examples": dict(self.examples)}

    @classmethod
    def from_plain(cls, d: dict[str, Any]) -> "Tally":
        t = cls()
        for k, row in d["cal"].items():
            t.cal[k] = list(row)
        t.count.update(d["count"])
        for k, lst in d["examples"].items():
            t.examples[k] = list(lst)
        return t


def _can_promote_now(cfg: Config, rank: int, money: int, merit: int, origin, cards) -> str:
    """资源 + 手里的卡，够不够这一轮升一级（按结算前的状态，不算本轮产出）。

    返回 "merit"（能走政绩，零举报风险）/ "money"（只能掏钱，有举报风险）/ ""（不够）。
    """
    import rules

    mc = rules.money_cost_at(rank, origin, cfg)
    tc = rules.merit_cost_at(rank, origin, cfg)
    if mc is None:
        return ""
    promos = [c for c in cards if c.is_promotion and c is not Card.PROMOTE_FAMILY]
    if cfg.needs_both(rank):
        return "money" if promos and money >= mc and tc is not None and merit >= tc else ""
    if any(c.can_use_merit for c in promos) and tc is not None and merit >= tc:
        return "merit"
    if any(c.can_use_money for c in promos) and money >= mc:
        return "money"
    return ""


def audit_games(args: tuple) -> dict[str, Any]:
    start, n_games, seed = args[:3]
    policy = ai.load_policy(args[3]) if len(args) > 3 and args[3] else None
    cfg = HAND_CFG
    t = Tally()
    for g in range(start, start + n_games):
        rng = random.Random(seed * 100003 + g)
        game = Game(game_id=f"audit{g}", cfg=cfg, rng=rng)
        for i in range(N):
            game.add_player(f"P{i + 1}")
        pool_ids = list(cfg.origin_ids())
        rng.shuffle(pool_ids)
        for pid, oid in zip(sorted(game.players), pool_ids):
            game.players[pid].origin = Origin(oid)
        pool = ai.AgentPool(cfg=cfg, rng=rng, policy=policy)
        game.start_game()
        t.count["games"] += 1

        while not game.is_over:
            rnd = game.round_number
            before = {pid: (p.rank, p.money, p.merit) for pid, p in game.players.items()}
            ranks_now = {pid: (p.rank, p.merit) for pid, p in game.players.items()}
            decisions: dict[int, dict[str, Any]] = {}
            for pid in sorted(game.players):
                picks = ai.turn(game, pid, pool)
                agent = pool.get(pid)
                hand = [d.card for d in game.hands[pid]]
                fam = (game.private_state(pid).get("family_card") or {}).get("usable")
                pub = game.public_state()
                believed_closing = {
                    o["id"]: agent._about_to_win(o, agent.models.get(o["id"], ai.OpponentModel()))
                    for o in pub["players"] if o["id"] != pid
                }
                # 有人被判要登顶、我手里有干扰牌：按"拦下之后我是不是头号挑战者"分组，看我拦没拦
                closers = [x for x, b in believed_closing.items() if b]
                if closers and any(c in (Card.ATTACK, Card.REPORT) for c in hand):
                    lead = max(closers, key=lambda x: game.players[x].rank)
                    rest = [x for x in game.players if x != lead]
                    order = sorted(rest, key=lambda x: (game.players[x].rank, game.players[x].merit,
                                                        game.players[x].money), reverse=True)
                    place = order.index(pid) + 2  # 加上要登顶的那个人，我是全场第几
                    bucket_name = f"全场第{place}" if place <= 3 else "全场第4名及以后"
                    hit = any(p.get("target") in closers for p in picks)
                    t.cal_add(f"拦人|{bucket_name}", 0.0, hit)
                    c = agent._contender(lead)
                    behind = -__import__("math").log(max(c, 1e-6)) / max(agent.w.contender_decay, 1e-6)
                    cb = ("头号挑战者" if behind < 0.25 else "落后约半级" if behind < 0.75
                          else "落后约一级" if behind < 1.5 else "落后两级以上")
                    t.cal_add(f"拦人|{cb}", c, hit)
                    runner_up_miss = lead if place == 2 and not hit else None
                else:
                    runner_up_miss = None
                decisions[pid] = {
                    "runner_up_miss": runner_up_miss,
                    "picks": picks, "hand": hand, "family": fam,
                    "pred": dict(agent.last_prediction),
                    "believed_closing": believed_closing,
                }
                game.select_actions(pid, picks)
                if picks:
                    game.lock_action(pid)
            game.force_lock_all()
            event = game.reveal_event()
            outcome = game.resolve()
            top = cfg.president_rank
            winners_now = [
                pid for pid, p in game.players.items()
                if p.rank >= top and before[pid][0] < top
            ]
            leader_key = max(ranks_now.values())

            # ---- 第二名有牌却没拦：为什么 ----
            for pid, d in decisions.items():
                lead = d.get("runner_up_miss")
                if lead is None:
                    continue
                played = [Card(p["action"]) for p in d["picks"]]
                others_hit = any(p.get("target") == lead
                                 for q, dd in decisions.items() if q != pid for p in dd["picks"])
                lead_won = lead in winners_now
                if before[pid][0] == top - 1 and any(c.is_promotion for c in played):
                    why = "我自己也在冲主席"
                elif any(c.is_promotion for c in played):
                    why = "我在用晋升卡升职"
                elif others_hit:
                    why = "别人已经在拦他"
                else:
                    why = "就是没拦（在发展）"
                t.count[f"ru|{why}"] += 1
                t.count[f"ru|{why}|他登顶了"] += lead_won
                t.count["ru|total"] += 1

            # ---- "他这轮要登顶"判得准不准（只看省级的对手）----
            # 真相 = 按他的真实钱、政绩和手牌，没人拦的话这一轮够不够登顶
            import rules
            able = {}
            for opp, p in game.players.items():
                r0, m0, me0 = before[opp]
                if r0 != top - 1:
                    continue
                org = p.origin.value if p.origin else None
                mc = rules.money_cost_at(r0, org, cfg)
                tc = rules.merit_cost_at(r0, org, cfg)
                hand = game.hands[opp]
                has_card = any(x.card.is_promotion and x.card is not Card.PROMOTE_FAMILY for x in hand)
                best_work = max((rules.work_merit(x.value, r0, None, cfg) for x in hand
                                 if x.card is Card.WORK), default=0)
                best_cash = max((x.value * cfg.money_multiplier(r0) for x in hand
                                 if x.card in (Card.CORRUPT, Card.GRAFT)), default=0)
                able[opp] = has_card and (
                    (m0 >= mc and me0 >= tc)
                    or (m0 >= mc and me0 + best_work >= tc)
                    or (me0 >= tc and m0 + best_cash >= mc)
                )
            for pid, d in decisions.items():
                for opp, believed in d["believed_closing"].items():
                    if opp not in able:
                        continue
                    won = opp in winners_now
                    t.count["closing_pairs"] += 1
                    t.count["closing_flag"] += believed
                    t.count["closing_won"] += won
                    t.count["closing_true_pos"] += believed and won
                    t.count["closing_able"] += able[opp]
                    t.count["closing_flag_able"] += believed and able[opp]

            for pid, d in decisions.items():
                o = outcome.outcomes[pid]
                me = game.players[pid]
                rank0, money0, merit0 = before[pid]
                played = [Card(p["action"]) for p in d["picks"]]
                pred = d["pred"]
                t.count["decisions"] += 1
                where = f"局{g} 第{rnd}轮 P{pid}({cfg.rank_name(rank0)}, 钱{money0}, 政绩{merit0})"

                # ---- 1a 校准：贪了会不会被查实 ----
                if pred.get("p_caught") is not None and o.corrupt_amount > 0:
                    p = float(pred["p_caught"])
                    by_player = o.report_effective and o.report_count_players > 0
                    t.cal_add(f"贪污查实|预测{bucket(p)}", p, o.report_effective)
                    t.cal_add(f"贪污查实|{cfg.rank_name(rank0)}", p, o.report_effective)
                    lead = "明面领先" if ranks_now[pid] >= leader_key else "不领先"
                    t.cal_add(f"贪污查实|{lead}", p, o.report_effective)
                    t.cal_add("贪污查实|全部", p, o.report_effective)
                    t.cal_add("贪污查实(只算玩家举报)|全部", p, by_player)
                    t.count["corrupt_plays"] += 1

                # ---- 1a 校准：举报命中率 ----
                for target, (pc, pb, p) in (pred.get("reports") or {}).items():
                    ot = outcome.outcomes[target]
                    bribing = (ot.pending_bribe > 0 or ot.family_bribe > 0
                               or ot.promotion in (PromotionKind.MONEY, PromotionKind.BOTH))
                    t.cal_add(f"他在贪|预测{bucket(pc)}", pc, ot.corrupt_amount > 0)
                    t.cal_add("他在贪|全部", pc, ot.corrupt_amount > 0)
                    t.cal_add(f"他在买官|预测{bucket(pb)}", pb, bribing)
                    t.cal_add(f"他在买官|目标{cfg.rank_name(before[target][0])}", pb, bribing)
                    t.cal_add(f"举报查实|预测{bucket(p)}", p, ot.report_effective)
                    t.cal_add(f"举报查实|目标{cfg.rank_name(before[target][0])}", p, ot.report_effective)
                    t.cal_add("举报查实|全部", p, ot.report_effective)

                # ---- 攻击结果 ----
                n_attack = sum(1 for c in played if c is Card.ATTACK)
                if n_attack:
                    t.count["attacks"] += n_attack
                    if o.attacks_landed == 0 and o.merit_from_attacks == 0:
                        t.count["attack_blank"] += n_attack
                    t.count["attack_merit_gained"] += o.merit_from_attacks

                # ---- 1b 失误：够了没升 ----
                origin = me.origin.value if me.origin else None
                hand_cards = d["hand"]
                route = _can_promote_now(cfg, rank0, money0, merit0, origin, hand_cards)
                if route:
                    t.count[f"could_promote_{route}"] += 1
                    if not any(c.is_promotion for c in played) and o.promotion is PromotionKind.NONE:
                        t.count[f"blunder_skip_promotion_{route}"] += 1
                        kind = ("政绩够（零风险）、手里有卡却没升" if route == "merit"
                                else "钱够、手里有卡却没买官（有举报风险）")
                        closing = [w for w, b in d["believed_closing"].items() if b]
                        t.example(kind, f"{where} 手牌 {[c.value for c in hand_cards]} -> 出 "
                                        f"{[(p['action'], p.get('target')) for p in d['picks']]}"
                                        + (f"  （AI 认为 P{closing} 要登顶）" if closing else ""))

                # ---- 1b 失误：晋升卡白打（没被攻击/举报拦，就是自己算错）----
                promo_played = [c for c in played if c.is_promotion and c is not Card.PROMOTE_FAMILY]
                if promo_played:
                    t.count["promo_cards_played"] += 1
                    if (o.promotion is PromotionKind.NONE and not o.merit_promotion_blocked
                            and not o.promotion_frozen_by_report and not o.report_effective):
                        producers = [c for c in played if c.is_production]
                        if producers:
                            t.count["promo_gamble_failed"] += 1  # 赌这轮产出能补上，没赌中
                        else:
                            t.count["blunder_dead_promotion"] += 1
                            closing = [w for w, b in d["believed_closing"].items() if b]
                            t.example("晋升卡白打（没生产牌、本来就不够）",
                                      f"{where} 手牌 {[c.value for c in hand_cards]} -> 出 "
                                      f"{[(p['action'], p.get('target')) for p in d['picks']]}"
                                      + (f"  （AI 认为 P{closing} 要登顶）" if closing else ""))
                if Card.PROMOTE_FAMILY in played:
                    t.count["family_played"] += 1
                    if not o.family_promotion:
                        t.count["blunder_family_wasted"] += 1
                        t.count["family_wasted_caught" if o.report_effective else "family_wasted_short"] += 1
                        t.example("一纸调令白用", f"{where} 出 {[c.value for c in played]}")

                # ---- 1b 失误：有人当轮登顶，我手里的干扰牌没往他身上打 ----
                for w in winners_now:
                    if w == pid:
                        continue
                    held = [c for c in d["hand"] if c in (Card.ATTACK, Card.REPORT)]
                    if not held:
                        t.count["winner_no_card"] += 1
                        continue
                    shot = [p for p in d["picks"] if p.get("target") == w]
                    believed = d["believed_closing"].get(w, False)
                    t.count["winner_could_interfere"] += 1
                    t.count["winner_believed_closing"] += believed
                    if not shot:
                        t.count["blunder_missed_block"] += 1
                        t.count["missed_block_believed" if believed else "missed_block_unaware"] += 1
                        wr, wm, wme = before[w]
                        others = [x for x, b in d["believed_closing"].items() if b and x != w]
                        if believed and any(p.get("target") in others for p in d["picks"]):
                            t.count["missed_block_hit_other_closer"] += 1
                        t.example("有人当轮登顶，手里的干扰牌没打他",
                                  f"{where} 手里 {[c.value for c in held]}，P{w}({cfg.rank_name(wr)}, 钱{wm}, 政绩{wme}) "
                                  f"登顶；AI 当时{'看出来了' if believed else '没看出来'}"
                                  + (f"（还认为 P{others} 也要登顶）" if others else "")
                                  + f"，出了 {[(p['action'], p.get('target')) for p in d['picks']]}", limit=8)
            if not game.is_over:
                game.advance_round()
    return t.to_plain()


def exploit(args: tuple) -> tuple[str, float, int]:
    """1 个脚本打法 + 5 个 AI（手写，或给了权重文件就用学习型），轮换座位。返回夺冠份额。"""
    name, n_games, seed = args[:3]
    theta = ai.load_policy(args[3]) if len(args) > 3 and args[3] else None
    cfg = HAND_CFG
    rng = random.Random(seed)
    share = 0.0
    for g in range(n_games):
        seat = g % N
        names = ["smart"] * N
        names[seat] = name
        pool_ids = list(cfg.origin_ids())
        rng.shuffle(pool_ids)
        assign = analysis.build_assignment(names, cfg, rng)
        if theta is not None:
            for s in range(N):
                if s != seat:
                    pool = ai.AgentPool(cfg=cfg, rng=rng, policy=theta)
                    assign[s] = (lambda p: lambda game, me, hand, others, _r: ai.turn(game, me.id, p))(pool)
        rec = analysis.play(N, rng, cfg, assign, origins=pool_ids[:N])
        pid = sorted(rec.cards)[seat]
        if pid in rec.winners:
            share += 1.0 / len(rec.winners)
    return name, 100.0 * share / n_games, n_games


def duel(args: tuple) -> tuple[str, float, int]:
    """1 个改了权重的 AI + 5 个标准 AI，轮换座位。夺冠份额 > 16.67% = 这个改动让 AI 更强。

    各变体用同一串种子（同样的发牌、同样的出身分配），差异主要来自那一席的决策。
    """
    label, over, n_games, seed, crowd_over = args
    cfg = HAND_CFG
    rng = random.Random(seed)
    weights = dataclasses.replace(ai.Weights(), **over)
    crowd_weights = dataclasses.replace(ai.Weights(), **crowd_over)

    def variant(cfg: Config, rng: random.Random, w):
        pool = ai.AgentPool(cfg=cfg, rng=rng, weights=w)
        return lambda game, me, hand, others, _rng: ai.turn(game, me.id, pool)

    share = 0.0
    for g in range(n_games):
        seat = g % N
        assign = [variant(cfg, rng, crowd_weights) for _ in range(N)]
        assign[seat] = variant(cfg, rng, weights)
        pool_ids = list(cfg.origin_ids())
        rng.shuffle(pool_ids)
        rec = analysis.play(N, rng, cfg, assign, origins=pool_ids[:N])
        pid = sorted(rec.cards)[seat]
        if pid in rec.winners:
            share += 1.0 / len(rec.winners)
    return label, 100.0 * share / n_games, n_games


def _load_ai(path: str):
    """把另一份 ai.py（比如改之前的快照）当成独立模块载进来，和现在的 AI 同桌打。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("ai_other", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ai_other"] = mod  # dataclass 要能在 sys.modules 里找到自己的模块
    spec.loader.exec_module(mod)
    return mod


def versus(args: tuple[str, str, int, int]) -> tuple[str, float, int]:
    """1 个 A 版 AI + 5 个 B 版 AI，轮换座位。mode = "new_in_old" / "old_in_new"。"""
    mode, old_path, n_games, seed = args
    cfg = HAND_CFG
    old = _load_ai(old_path)
    rng = random.Random(seed)

    def factory(mod):
        def make(cfg: Config, rng: random.Random):
            pool = mod.AgentPool(cfg=cfg, rng=rng)
            return lambda game, me, hand, others, _rng: mod.turn(game, me.id, pool)
        return make

    lone, crowd = (factory(ai), factory(old)) if mode == "new_in_old" else (factory(old), factory(ai))
    share = 0.0
    for g in range(n_games):
        seat = g % N
        assign = [crowd(cfg, rng) for _ in range(N)]
        assign[seat] = lone(cfg, rng)
        pool_ids = list(cfg.origin_ids())
        rng.shuffle(pool_ids)
        rec = analysis.play(N, rng, cfg, assign, origins=pool_ids[:N])
        pid = sorted(rec.cards)[seat]
        if pid in rec.winners:
            share += 1.0 / len(rec.winners)
    return mode, 100.0 * share / n_games, n_games


def learned_vs_smart(args: tuple[str, str, int, int]) -> tuple[str, float, int]:
    """学习型 AI 和手写 AI 同桌。mode = "learned_in_smart"（1 学 5 手写）/ "smart_in_learned"。"""
    mode, policy_path, n_games, seed = args[:4]
    other = ai.load_policy(args[4]) if len(args) > 4 and args[4] else None  # 对照组：另一份权重，默认手写
    cfg = HAND_CFG
    theta = ai.load_policy(policy_path)
    rng = random.Random(seed)

    def make(policy):
        def f(cfg: Config, rng: random.Random):
            pool = ai.AgentPool(cfg=cfg, rng=rng, policy=policy)
            return lambda game, me, hand, others, _rng: ai.turn(game, me.id, pool)
        return f

    lone, crowd = (make(theta), make(other)) if mode == "learned_in_smart" else (make(other), make(theta))
    share = 0.0
    for g in range(n_games):
        seat = g % N
        assign = [crowd(cfg, rng) for _ in range(N)]
        assign[seat] = lone(cfg, rng)
        ids = list(cfg.origin_ids())
        rng.shuffle(ids)
        rec = analysis.play(N, rng, cfg, assign, origins=ids[:N])
        pid = sorted(rec.cards)[seat]
        if pid in rec.winners:
            share += 1.0 / len(rec.winners)
    return mode, 100.0 * share / n_games, n_games


DYNAMIC_TABLES = {
    "全是只举报的": ["reporter"] * 5,
    "全是从不举报、只干活的": ["worker"] * 5,
    "全是只攻击的": ["attacker"] * 5,
    "全是只贪的": ["corrupt"] * 5,
    "全是手写 AI": ["smart"] * 5,
}


def dynamic(args: tuple[str, str, str, int, int]) -> tuple[str, str, dict[str, float]]:
    """把被测 AI 放进一桌"怪人"里，看它的打法会不会跟着变。"""
    who, policy_path, table, n_games, seed = args
    cfg = HAND_CFG
    rng = random.Random(seed)
    theta = ai.load_policy(policy_path) if who == "学习型" else None
    mix: Counter = Counter()
    share = 0.0
    for g in range(n_games):
        seat = g % N
        pool = ai.AgentPool(cfg=cfg, rng=rng, policy=theta)
        names = list(DYNAMIC_TABLES[table])
        assign = analysis.build_assignment(names[:seat] + ["smart"] + names[seat:], cfg, rng)
        assign[seat] = lambda game, me, hand, others, _rng: ai.turn(game, me.id, pool)
        ids = list(cfg.origin_ids())
        rng.shuffle(ids)
        rec = analysis.play(N, rng, cfg, assign, origins=ids[:N])
        pid = sorted(rec.cards)[seat]
        if pid in rec.winners:
            share += 1.0 / len(rec.winners)
        for card, n in rec.cards[pid].items():
            mix[card] += n
    total = sum(mix.values()) or 1
    out = {c: 100.0 * mix[c] / total for c in ("WORK", "CORRUPT", "GRAFT", "REPORT", "ATTACK")}
    out["胜率"] = 100.0 * share / n_games
    return who, table, out


def force_block_duel(args: tuple[bool, str, int, int]) -> tuple[bool, float, int]:
    """1 个学习型（可选"头号挑战者必拦"）+ 5 个学习型。比较强制拦人到底是赚是亏。"""
    force, policy_path, n_games, seed = args
    cfg = HAND_CFG
    theta = ai.load_policy(policy_path)
    rng = random.Random(seed)

    def make(flag: bool):
        pool = ai.AgentPool(cfg=cfg, rng=rng, policy=theta)

        def play(game, me, hand, others, _rng):
            pool.get(me.id).force_block = flag
            return ai.turn(game, me.id, pool)
        return play

    share = 0.0
    for g in range(n_games):
        seat = g % N
        assign = [make(False) for _ in range(N)]
        assign[seat] = make(force)
        ids = list(cfg.origin_ids())
        rng.shuffle(ids)
        rec = analysis.play(N, rng, cfg, assign, origins=ids[:N])
        pid = sorted(rec.cards)[seat]
        if pid in rec.winners:
            share += 1.0 / len(rec.winners)
    return force, 100.0 * share / n_games, n_games


def parse_weights(spec: str) -> dict[str, Any]:
    """'endgame_economy_discount=0.5,noise=0.01' -> {...}，按 Weights 里的类型转换。"""
    fields = {f.name: f.type for f in dataclasses.fields(ai.Weights)}
    out: dict[str, Any] = {}
    for item in filter(None, spec.split(",")) if spec != "base" else []:
        key, _, val = item.partition("=")
        if key not in fields:
            raise SystemExit(f"Weights 里没有 {key!r}")
        out[key] = int(val) if fields[key] in ("int", int) else float(val)
    return out


def features(args: tuple[int, int, int, bool]) -> dict[tuple[str, str], list[int]]:
    """每轮每一对（观察者 AI, 对手）：AI 看得到的特征 -> 对手这一轮实际有没有贪、有没有掏钱升职。

    ai.py 里 CORRUPT_BY_* / BRIBE_BY_* 那些因子表就是按这张表拟合的。
    bots=True 时每局混一个脚本打法（只贪 / 爬线 / 只干活），让"个人档案"这一项有区分度。
    """
    import rules  # noqa: F401

    start, n_games, seed, bots = args[:4]
    policy = ai.load_policy(args[4]) if len(args) > 4 and args[4] else None
    cfg = HAND_CFG
    tab: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for g in range(start, start + n_games):
        rng = random.Random(seed * 100003 + g)
        game = Game(game_id=f"feat{g}", cfg=cfg, rng=rng)
        for i in range(N):
            game.add_player(f"P{i + 1}")
        ids = list(cfg.origin_ids())
        rng.shuffle(ids)
        for pid, oid in zip(sorted(game.players), ids):
            game.players[pid].origin = Origin(oid)
        pool = ai.AgentPool(cfg=cfg, rng=rng, policy=policy)
        game.start_game()
        record = {pid: [0, 0] for pid in game.players}
        bot_seat = sorted(game.players)[g % N]
        bot_name = (["corrupt", "climber", "worker"] + ["smart"] * 3)[g % N] if bots else "smart"
        bot = None if bot_name == "smart" else analysis.STRATEGIES[bot_name]
        while not game.is_over:
            rnd = game.round_number
            feats = []
            for pid in sorted(game.players):
                if bot is not None and pid == bot_seat:
                    picks = bot(game, game.players[pid], game.hands[pid],
                                [x for x in game.players if x != pid], rng) or []
                else:
                    picks = ai.turn(game, pid, pool)
                    agent = pool.get(pid)
                    pub = game.public_state()
                    for o in pub["players"]:
                        if o["id"] == pid:
                            continue
                        m = agent.models.get(o["id"], ai.OpponentModel())
                        mc, tc = agent._costs(o["rank"], o.get("origin"))
                        ratio = (m.money_est / mc) if mc else 0
                        dirty, seen = record[o["id"]]
                        f = {
                            "个人档案": f"{min(8, int((dirty + 1) / (seen + 4) * 10)) / 10:.1f}+",
                            "官职": cfg.rank_name(o["rank"]),
                            "上轮被传闻点名": rnd - m.last_seen_corrupting <= 1,
                            "估钱/门槛": ("≥1" if ratio >= 1 else "0.6-1" if ratio >= 0.6
                                         else "0.3-0.6" if ratio >= 0.3 else "<0.3"),
                            "政绩够门槛": tc is not None and o["merit"] >= tc,
                            "不干活比例": f"{min(4, int(m.quiet_rate * 5)) / 5:.1f}+",
                            "轮次": "1-3" if rnd <= 3 else ("4-7" if rnd <= 7 else "8+"),
                            "出身": o.get("origin"),
                        }
                        if o["rank"] == cfg.president_rank - 1:
                            f["省级·AI判要登顶"] = agent._about_to_win(o, m)
                        feats.append((o["id"], f))
                game.select_actions(pid, picks)
                if picks:
                    game.lock_action(pid)
            game.force_lock_all()
            game.reveal_event()
            out = game.resolve()
            for pid in game.players:
                record[pid][1] += 1
                record[pid][0] += (pid in out.wealth_top_ids) or out.outcomes[pid].warnings_issued > 0
            for oid, f in feats:
                o = out.outcomes[oid]
                c = o.corrupt_amount > 0
                b = (o.pending_bribe > 0 or o.family_bribe > 0
                     or o.promotion in (PromotionKind.MONEY, PromotionKind.BOTH))
                for k, v in f.items():
                    row = tab[(k, str(v))]
                    row[0] += 1
                    row[1] += c
                    row[2] += b
            if not game.is_over:
                game.advance_round()
    return dict(tab)


def report(t: Tally) -> None:
    c = t.count
    print("=" * 60)
    print(f"AI 体检   {c['games']} 局 × 6 个思考型 AI（六身份混战）  共 {c['decisions']} 次出牌")
    print("=" * 60)

    def cal_table(prefix: str, title: str) -> None:
        rows = [(k[len(prefix) + 1:], v) for k, v in t.cal.items() if k.startswith(prefix + "|")]
        if not rows:
            return
        print(f"\n{title}")
        print(f"  {'分组':<14}{'样本':>8}{'AI 预测':>10}{'实际':>10}{'差':>9}")
        order = {"全部": -1}
        for key, (ps, hits, n) in sorted(rows, key=lambda kv: (order.get(kv[0], 0), kv[0])):
            pred, real = ps / n, hits / n
            flag = "  <<< 估低了" if real - pred > 0.08 else ("  <<< 估高了" if pred - real > 0.08 else "")
            print(f"  {key:<14}{int(n):>8}{pred:>10.1%}{real:>10.1%}{real - pred:>+9.1%}{flag}")

    print("\n1a. 校准：AI 出牌时的预测 vs 真实结果")
    cal_table("贪污查实", "  贪污之后被查实的概率（含反腐风暴等事件）")
    cal_table("贪污查实(只算玩家举报)", "  其中只算被玩家举报查实")
    cal_table("举报查实", "  举报命中率")
    cal_table("他在贪", "  举报时估的「他这轮在贪」vs 他真的贪了没有")
    cal_table("他在买官", "  举报时估的「他这轮在买官」vs 他真的掏钱升职了没有")
    cal_table("拦人", "  有人要登顶、我手里有干扰牌时，我拦他的比例（看'实际'那一列）")

    if c["ru|total"]:
        print(f"\n  全场第 2 名有干扰牌却没拦要登顶的人（{c['ru|total']} 次）：")
        for why in ("我自己也在冲主席", "我在用晋升卡升职", "别人已经在拦他", "就是没拦（在发展）"):
            n = c[f"ru|{why}"]
            won = c[f"ru|{why}|他登顶了"]
            print(f"    {why:<16}{n:>6}  ({n / c['ru|total']:.0%})   其中他真登顶了 {won}"
                  f"（{won / n:.0%}）" if n else f"    {why:<16}{0:>6}")

    print("\n1b. 明显失误")

    def rate(a: str, b: str) -> str:
        return f"{c[a]} / {c[b]} = {c[a] / c[b]:.1%}" if c[b] else "—"

    print(f"  政绩够（零风险）、手里有卡，却没升          : {rate('blunder_skip_promotion_merit', 'could_promote_merit')}")
    print(f"  钱够、手里有卡，却没买官（有举报风险，未必错）: {rate('blunder_skip_promotion_money', 'could_promote_money')}")
    print(f"  晋升卡打出去，没生产牌、本来就不够（白打）  : {rate('blunder_dead_promotion', 'promo_cards_played')}")
    print(f"  晋升卡赌这轮产出能补上，没赌中              : {rate('promo_gamble_failed', 'promo_cards_played')}")
    print(f"  一纸调令白用                                : {rate('blunder_family_wasted', 'family_played')}"
          f"   （被查实抄走 {c['family_wasted_caught']}，本来就不够 {c['family_wasted_short']}）")
    print(f"  攻击打空（没抢到、没拦下）                  : {rate('attack_blank', 'attacks')}"
          f"   平均每刀抢到 {c['attack_merit_gained'] / max(1, c['attacks']):.2f} 政绩")
    print(f"  「他这轮要登顶」的判断（对手在省级时，{c['closing_pairs']} 次）：")
    print(f"    按「他真的当轮登顶了」算：准确率 {rate('closing_true_pos', 'closing_flag')}"
          f"，查全率 {rate('closing_true_pos', 'closing_won')}")
    print(f"    按「他的钱+政绩+手牌没人拦就够登顶」算：准确率 {rate('closing_flag_able', 'closing_flag')}"
          f"，查全率 {rate('closing_flag_able', 'closing_able')}")
    print(f"  有人当轮登顶时，手里有干扰牌的旁观者        : {c['winner_could_interfere']} 人次"
          f"（其中 AI 事先看出来的 {rate('winner_believed_closing', 'winner_could_interfere')}）")
    print(f"    ……干扰牌没往登顶的人身上打                : {rate('blunder_missed_block', 'winner_could_interfere')}"
          f"   （看出来了还没打 {c['missed_block_believed']}，"
          f"其中打了另一个'要登顶'的 {c['missed_block_hit_other_closer']}；没看出来 {c['missed_block_unaware']}）")
    for kind, lst in t.examples.items():
        print(f"\n  例子 · {kind}")
        for line in lst:
            print(f"    {line}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--exploit", type=int, default=0, help="每个脚本打法的单挑局数（0 = 不跑）")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--duel", action="append", default=[],
                    help="权重变体，如 endgame_economy_discount=0.5（可写多个；自动加一个 base 对照）")
    ap.add_argument("--duel-games", type=int, default=3000)
    ap.add_argument("--crowd", default="", help="另外 5 个 AI 用的权重（默认标准权重），格式同 --duel")
    ap.add_argument("--features", type=int, default=0,
                    help="跑这么多局，打印 AI 读牌用的特征 vs 实际行为（拟合 ai.py 因子表用）")
    ap.add_argument("--bots", action="store_true", help="--features 时每局混一个脚本打法")
    ap.add_argument("--audit-policy", default="", help="体检学习型 AI（六个座位都用这份权重）")
    ap.add_argument("--policy", default="", help="学习型 AI 的权重文件：和手写 AI 同桌对决（用 --duel-games 局数）")
    ap.add_argument("--dynamic", type=int, default=0,
                    help="动态性测试：被测 AI 坐进各种怪桌，每桌跑这么多局（要和 --policy 一起用才有学习型对照）")
    ap.add_argument("--policy-b", default="", help="和 --policy 一起用：对照组换成另一份学习型权重（默认手写）")
    ap.add_argument("--force-block", action="store_true",
                    help="和 --policy 一起用：学习型'头号挑战者必拦'vs 照常，各坐进 5 个学习型里")
    ap.add_argument("--vs-old", default="", help="另一份 ai.py 的路径：新版 1 打旧版 5、旧版 1 打新版 5")
    args = ap.parse_args(argv)

    jobs = []
    if args.games:
        per = max(1, args.games // args.workers)
        starts = list(range(0, args.games, per))
        jobs = [(s, min(per, args.games - s), args.seed, args.audit_policy) for s in starts]
    scripted = ["worker", "builder", "safe_climber", "climber", "corrupt", "challenger", "reporter"]
    if args.features:
        per = max(1, args.features // args.workers)
        tot: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for t in ex.map(features, [(i * per, per, args.seed, args.bots, args.audit_policy)
                                       for i in range(args.workers)]):
                for k, r in t.items():
                    for i in range(3):
                        tot[k][i] += r[i]
        cur = None
        for (k, v), (n, c, b) in sorted(tot.items()):
            if k != cur:
                print(f"\n{k:<10}{'样本':>8}{'真在贪':>8}{'真买官':>8}")
                cur = k
            print(f"  {v:<12}{n:>8}{c / n:>8.1%}{b / n:>8.1%}")
        return 0
    if (args.duel or args.vs_old or args.policy or args.dynamic) and not args.exploit \
            and args.games == 2000:
        jobs = []  # 只跑对决时不默认跑体检
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(audit_games, j) for j in jobs]
        efuts = [ex.submit(exploit, (s, args.exploit, args.seed + i, args.audit_policy))
                 for i, s in enumerate(scripted)] if args.exploit else []
        pfuts = [ex.submit(learned_vs_smart, (m, args.policy, args.duel_games, args.seed + k, args.policy_b))
                 for k, m in enumerate(["learned_in_smart", "smart_in_learned"] * 2)] \
            if args.policy and not args.dynamic and not args.force_block else []
        ffuts = [ex.submit(force_block_duel, (flag, args.policy, args.duel_games, args.seed + k))
                 for k in range(2) for flag in (False, True)] if args.force_block else []
        whos = ["手写"] + (["学习型"] if args.policy else [])
        yfuts = [ex.submit(dynamic, (w, args.policy, t, args.dynamic, args.seed))
                 for w in whos for t in DYNAMIC_TABLES] if args.dynamic else []
        vfuts = [ex.submit(versus, (m, args.vs_old, args.duel_games, args.seed + k))
                 for k, m in enumerate(["new_in_old", "old_in_new"] * 2)] if args.vs_old else []
        crowd = parse_weights(args.crowd) if args.crowd else {}
        dfuts = [ex.submit(duel, (spec, parse_weights(spec), args.duel_games, args.seed, crowd))
                 for spec in (["base"] if args.duel else []) + args.duel]
        total = Tally()
        for f in futs:
            total.merge(Tally.from_plain(f.result()))
        if jobs:
            report(total)
        if efuts:
            print("\n1c. 剥削测试：1 个脚本打法 + 5 个思考型 AI（公平线 16.67%）")
            for f in efuts:
                name, pct, n = f.result()
                half = 1.96 * (pct / 100 * (1 - pct / 100) / n) ** 0.5 * 100
                flag = "  <<< 能剥削 AI" if pct - half > 16.67 else ""
                print(f"  {name:<14}{pct:>7.2f}%  ±{half:.2f}{flag}")
        if pfuts:
            agg_p: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
            for f in pfuts:
                mode, pct, n = f.result()
                agg_p[mode][0] += pct * n
                agg_p[mode][1] += n
            b = args.policy_b or "手写"
            print(f"\n{args.policy} vs {b}（公平线 16.67%）")
            for mode, label in (("learned_in_smart", f"1 个 A + 5 个 B：A"),
                                ("smart_in_learned", f"1 个 B + 5 个 A：B")):
                tot, n = agg_p[mode]
                pct = tot / n
                half = 1.96 * (pct / 100 * (1 - pct / 100) / n) ** 0.5 * 100
                print(f"  {label:<26}{pct:>7.2f}%  ±{half:.2f}   （{n} 局）")
        if ffuts:
            agg_f: dict[bool, list[float]] = defaultdict(lambda: [0.0, 0])
            for f in ffuts:
                flag, pct, n = f.result()
                agg_f[flag][0] += pct * n
                agg_f[flag][1] += n
            print("\n强制拦人对照：1 个学习型 + 5 个学习型（公平线 16.67%）")
            for flag, label in ((False, "照常（自己判断拦不拦）"), (True, "头号挑战者必拦")):
                tot, n = agg_f[flag]
                pct = tot / n
                half = 1.96 * (pct / 100 * (1 - pct / 100) / n) ** 0.5 * 100
                print(f"  {label:<20}{pct:>7.2f}%  ±{half:.2f}   （{n} 局）")
        if yfuts:
            print(f"\n动态性测试：被测 AI 坐进各种怪桌，它自己打出的牌里各占多少（%）")
            print(f"  {'谁':<6}{'桌子':<20}{'干活':>7}{'贪污':>7}{'以权谋私':>9}{'举报':>7}{'攻击':>7}{'胜率':>8}")
            for f in yfuts:
                who, table, o = f.result()
                print(f"  {who:<6}{table:<20}{o['WORK']:>7.1f}{o['CORRUPT']:>7.1f}{o['GRAFT']:>9.1f}"
                      f"{o['REPORT']:>7.1f}{o['ATTACK']:>7.1f}{o['胜率']:>8.1f}")
        if vfuts:
            agg: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
            for f in vfuts:
                mode, pct, n = f.result()
                agg[mode][0] += pct * n
                agg[mode][1] += n
            print(f"\n新旧 AI 对决（公平线 16.67%）")
            for mode, label in (("new_in_old", "1 个新版 + 5 个旧版：新版"),
                                ("old_in_new", "1 个旧版 + 5 个新版：旧版")):
                tot, n = agg[mode]
                pct = tot / n
                half = 1.96 * (pct / 100 * (1 - pct / 100) / n) ** 0.5 * 100
                print(f"  {label:<24}{pct:>7.2f}%  ±{half:.2f}   （{n} 局）")
        if dfuts:
            who = f"5 个 [{args.crowd}] AI" if args.crowd else "5 个标准 AI"
            print(f"\n权重对决：1 个改了权重的 AI + {who}（公平线 16.67%，各 {args.duel_games} 局）")
            for f in dfuts:
                name, pct, n = f.result()
                half = 1.96 * (pct / 100 * (1 - pct / 100) / n) ** 0.5 * 100
                flag = "  <<< 更强" if pct - half > 16.67 else ("  <<< 更弱" if pct + half < 16.67 else "")
                print(f"  {name:<44}{pct:>7.2f}%  ±{half:.2f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
