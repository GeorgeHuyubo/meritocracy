"""让大模型（Claude）当玩家打几局，出一份小样本平衡报告 + 玩家视角的反馈。

    python3 llm_play.py --games 10                       # 6 个 Sonnet 玩家、六身份各一，跑 10 局
    python3 llm_play.py --games 100 --parallel 3         # 同时跑 3 局
    python3 llm_play.py --report llm_runs/<目录>          # 只出报告（会再请模型把所有反馈归纳一遍）
    python3 llm_play.py --games 2 --backend mock         # 不调模型，用程序 AI 冒充，测流程

调用方式是 `claude -p`（Claude Code 的非交互模式），所以要在**你自己的终端**里跑
（Claude Code 会话里嵌套调用会被沙盒拦下）。

每个大模型玩家看到的信息和真人一样：规则、自己的出身技能、公开局面、最近几轮发生了什么、
自己的手牌和钱。另外附上程序估算的对手存款（钱是暗的，标明是估计）。**不给出牌建议**。
格式不对重试一次，还不行就退回程序 AI 并记一笔（报告里会写退回了几次）。

每局结束后问每个玩家：哪个身份最强 / 最弱、哪条规则不公平、体验几分、为什么。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ai  # noqa: E402
from config import DEFAULT_CONFIG, Config  # noqa: E402
from game import Game, GameError  # noqa: E402
from models import Card, Origin  # noqa: E402
import telemetry as T  # noqa: E402

N = 6
RUN_DIR = Path(__file__).resolve().parent / "llm_runs"
CARD_NAMES = {
    "WORK": "埋头工作", "CORRUPT": "中饱私囊", "GRAFT": "以权谋私", "REPORT": "匿名举报",
    "ATTACK": "政治攻击", "PROMOTE_MERIT": "政绩升职", "PROMOTE_MONEY": "贿赂升职",
    "PROMOTE_ANY": "通用升职", "PROMOTE_FAMILY": "一纸调令",
}


# ---------------------------------------------------------------------------
# 调模型
# ---------------------------------------------------------------------------


# 网关的临时性错误（限流 / 过载 / 5xx / 网络）：退避重试，不算"退回程序 AI"
TRANSIENT = ("rate", "limit", "overload", "429", "529", "500", "502", "503", "504",
             "timeout", "Timeout", "ECONN", "network", "Network", "API Error", "temporarily")
MAX_TRIES = 8


def _transient(msg: str) -> bool:
    return any(t in msg for t in TRANSIENT)


def _backoff(attempt: int) -> None:
    base = float(os.environ.get("LLM_BACKOFF_BASE", "10"))
    time.sleep(min(300, base * 2 ** attempt))  # 10 秒、20、40……最长 5 分钟


class ClaudeCLI:
    """每次调用起一个 `claude -p`，在空的临时目录里跑（不加载仓库的上下文）。"""

    def __init__(self, model: str, timeout: int = 240) -> None:
        self.model = model
        self.timeout = timeout
        self.cwd = tempfile.mkdtemp(prefix="meritocracy_llm_")
        self.calls = 0
        self.seconds = 0.0
        self.retries = 0
        self.restarts = 0

    def ask(self, prompt: str) -> str:
        for attempt in range(MAX_TRIES):
            try:
                return self._ask_once(prompt)
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                if attempt == MAX_TRIES - 1 or not (_transient(str(exc))
                                                   or isinstance(exc, subprocess.TimeoutExpired)):
                    raise
                self.retries += 1
                _backoff(attempt)
        raise RuntimeError("重试次数用完")

    def _ask_once(self, prompt: str) -> str:
        t0 = time.time()
        # 提示词走标准输入：几百份反馈拼起来会超过命令行参数的长度上限
        proc = subprocess.run(
            ["claude", "-p", "--model", self.model, "--output-format", "json", "--max-turns", "1"],
            input=prompt, cwd=self.cwd, capture_output=True, text=True, timeout=self.timeout,
        )
        self.calls += 1
        self.seconds += time.time() - t0
        for line in reversed(proc.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                env = json.loads(line)
                if env.get("is_error"):
                    raise RuntimeError(env.get("result") or "claude 返回错误")
                return str(env.get("result", ""))
        raise RuntimeError(f"claude 没有返回结果：{proc.stderr.strip()[-300:]}")


class ClaudeSession:
    """一个玩家一整局只起一个 claude 进程（stream-json 输入输出），上下文一直留着。

    开局把规则和身份放进系统提示（读一次），之后每轮只发"这轮的新局面和手牌"；
    之前几轮发生过什么它自己记得，赛后反馈也在同一个会话里问。
    换掉 Claude Code 自带的大段系统提示、关掉工具，每次调用快很多。

    过夜跑的可靠性：网关临时出错就退避重试；进程死掉 / 卡住就重开一个
    （系统提示照旧，前面几轮的记忆没了，但每轮发的内容本来就带着当前局面和上一轮经过）。
    stderr 写进临时文件——接成管道又没人读的话，写满 64KB 进程会卡死。
    """

    def __init__(self, model: str, system_prompt: str, extra: list[str], timeout: int = 240) -> None:
        self.model, self.system_prompt, self.extra = model, system_prompt, list(extra)
        self.timeout = timeout
        self.calls = 0
        self.seconds = 0.0
        self.retries = 0
        self.restarts = 0
        self.cwd = tempfile.mkdtemp(prefix="meritocracy_llm_")
        self._start()

    def _start(self) -> None:
        import queue
        import threading

        self.err = open(Path(self.cwd) / "stderr.log", "a", encoding="utf-8")
        self.proc = subprocess.Popen(
            ["claude", "-p", "--input-format", "stream-json", "--output-format", "stream-json",
             "--verbose", "--model", self.model, "--system-prompt", self.system_prompt, *self.extra],
            cwd=self.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err,
            text=True, bufsize=1,
        )
        self.lines: "queue.Queue[str | None]" = queue.Queue()
        proc, lines = self.proc, self.lines

        def pump() -> None:
            for line in proc.stdout:  # type: ignore[union-attr]
                lines.put(line)
            lines.put(None)

        threading.Thread(target=pump, daemon=True).start()

    def _restart(self) -> None:
        self.restarts += 1
        try:
            self.proc.kill()
        except Exception:  # noqa: BLE001
            pass
        self._start()

    def ask(self, text: str) -> str:
        note = ""
        for attempt in range(MAX_TRIES):
            try:
                return self._ask_once(note + text)
            except _SessionDied:
                if attempt == MAX_TRIES - 1:
                    raise RuntimeError("会话进程反复退出") from None
                self._restart()
                note = "（你的会话中断后重开了，前面几轮的经过可能不记得了，下面是当前局面。）\n"
            except RuntimeError as exc:
                if attempt == MAX_TRIES - 1 or not _transient(str(exc)):
                    raise
                self.retries += 1
                _backoff(attempt)
        raise RuntimeError("重试次数用完")

    def _ask_once(self, text: str) -> str:
        import queue

        t0 = time.time()
        msg = {"type": "user", "message": {"role": "user", "content": text}}
        try:
            self.proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")  # type: ignore[union-attr]
            self.proc.stdin.flush()  # type: ignore[union-attr]
        except (BrokenPipeError, OSError, ValueError):
            raise _SessionDied() from None
        deadline = t0 + self.timeout
        while True:
            try:
                line = self.lines.get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
                raise _SessionDied() from None  # 卡住了：当成进程死掉，重开
            if line is None:
                raise _SessionDied()
            line = line.strip()
            if not line.startswith("{"):
                continue
            ev = json.loads(line)
            if ev.get("type") == "result":
                self.calls += 1
                self.seconds += time.time() - t0
                if ev.get("is_error"):
                    raise RuntimeError(str(ev.get("result") or "会话返回错误"))
                return str(ev.get("result", ""))

    def close(self) -> None:
        try:
            self.proc.stdin.close()  # type: ignore[union-attr]
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.proc.kill()
        try:
            self.err.close()
        except Exception:  # noqa: BLE001
            pass


class _SessionDied(Exception):
    """常驻会话的进程退出了或者卡住了。"""


class MockBackend:
    """不调模型：出牌交给程序 AI（在 LLMPlayer 里处理），反馈给一段固定文本。用来测流程。"""

    calls = 0
    seconds = 0.0
    retries = 0
    restarts = 0

    def ask(self, prompt: str) -> str:
        self.calls += 1
        if "赛后反馈" in prompt:
            return json.dumps({
                "strongest": "官二代", "weakest": "小镇做题家·会计", "fun": 6,
                "unfair": "（mock）", "experience": "（mock）", "suggestion": "（mock）",
                "lessons": ["（mock 心得）"],
            }, ensure_ascii=False)
        return "MOCK"


def extract_json(text: str) -> dict[str, Any]:
    """从模型的回答里抠出第一个完整的 JSON 对象（允许它在前后多说几句）。"""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError("回答里没有 JSON")


# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------


def variant_cfg(variant: str) -> Config:
    """sweep.py 里的规则变体；后备的程序 AI 用这个变体续训出来的那份（有的话）。"""
    if not variant or variant == "base":
        return DEFAULT_CONFIG
    import sweep

    return sweep.build(variant, own_ai=True)


def rules_text(cfg: Config) -> str:
    ranks = [cfg.rank_name(r) for r in range(cfg.president_rank + 1)]
    steps = "；".join(
        f"{ranks[r]}→{ranks[r + 1]}：政绩 {cfg.merit_cost(r)} 或 金钱 {cfg.money_cost(r)}"
        + ("（这一步**两样都要**）" if cfg.needs_both(r) else "")
        for r in range(cfg.president_rank)
    )
    origins = "\n".join(
        f"  - {o['name']}「{o['skill']}」：{o['description']}" for o in cfg.origin_definitions
    )
    import rules

    hat = " / ".join(f"{ranks[r]} {rules.work_merit(cfg.attack_merit_penalty, r, None, cfg)}"
                     for r in range(cfg.president_rank))
    hat_reward = " / ".join(f"{ranks[r]} {rules.work_merit(cfg.attack_hat_reward, r, None, cfg)}"
                            for r in range(cfg.president_rank))
    return f"""【游戏：Meritocracy（官场升职）】6 人，最多 {cfg.max_rounds} 轮。官职：{' < '.join(ranks)}。
目标：第一个当上国家主席的人获胜（同一轮多人登顶，家底厚的那个当选）；打满 {cfg.max_rounds} 轮没人登顶就比官职和资源。
升职门槛（升一级要打一张对应的晋升卡）：{steps}。升职后多余的政绩会打折，钱只扣门槛。

每轮：每人发 6 张牌，最多打出 {cfg.picks_per_round} 张，按你排的顺序一张一张结算，然后公布全局事件和结算结果。
牌：
  - 埋头工作：加政绩（官越大倍率越高）
  - 中饱私囊：加不少钱，但这是贪污；以权谋私：钱少一点、顺带一点政绩，同样算贪污
  - 匿名举报（选一个人）：他这轮如果贪污或花钱买官，就被查实——本轮赃款没收（举报人分一半，扣 1 块跑腿费），
    记一次降职警告（攒满 {cfg.warnings_before_demotion} 次降一级），买的官作废、钱打水漂；他这轮没有经济问题就白打
  - 政治攻击（选一个人）：抢走他这轮埋头工作政绩的一半（几个人攻击同一人就平分这一半）；
    他要是靠政绩升职，就把这次升职暂缓（贫农例外，见出身）；
    他这轮一点政绩都没产（没干活）就"戴帽子"：扣他政绩 {hat}（按他的官职），每个攻击者记功 {hat_reward}（按攻击者官职）
  - 晋升卡（够门槛才能升一级，一轮最多升一级）：
      政绩升职 = 只花政绩（政绩不够就白打，不会掏钱；被攻击会暂缓）
      贿赂升职 = 只花钱（钱不够就白打；被举报查实就作废、钱打水漂）
      通用升职 = 先试政绩，政绩不够或被攻击按住时改成掏钱
    主席那一步钱和政绩都要够；打政绩升职就怕攻击，打贿赂升职就怕举报
工龄：连续 {cfg.tenure_required} 轮官职没变（没升没降），自动升一级，不用晋升卡；记一次降职警告会把工龄清零；
工龄{'也能' if cfg.tenure_can_reach_president else '升不到'}国家主席。
钱是暗的（别人看不到你有多少钱），政绩和官职是公开的。每轮公布"坊间传闻"：点名本轮到手钱最多的人（工资+没被查实的贪污+举报分到的赃款），不报金额。
全局事件（反腐风暴会查办贪得多的人、经济好坏会让金钱收益翻倍或减半、重点项目让埋头工作加政绩……）在出牌**之后**才公布。
克制关系：攻击克制走政绩路线的人，举报克制走金钱路线（贪污、买官）的人。

【胜负的关键】只有一个人能赢：别人当上主席，你和其余所有人都算输（不管你当时离主席多近）。
所以把快赢的人拦下来，和自己升官一样重要——拦住了，游戏才继续，你才有机会。
- 看谁离主席最近，不是看谁的政绩数字最大：省级玩家只要钱和政绩都凑够、再打一张晋升卡就赢了。
- 怎么拦：他想靠政绩升（政绩升职），政治攻击能把这次升职暂缓；他想靠钱买（贿赂升职），
  匿名举报能让它作废、钱打水漂；他打通用升职的话，攻击和举报一起压上去才稳。
- 一个人往往拦不住，几个人一起压上去才稳；手里没有攻击 / 举报时，可以花钱换牌去找。
- 但如果你自己这一轮就能当上主席，那就去冲。

出身（每人一个，有专属技能）：
{origins}"""


def player_prompt(cfg: Config, game: Game, pid: int, agent: ai.SmartAgent,
                  history: list[str], memory: list[str]) -> str:
    pub = game.public_state()
    priv = game.private_state(pid)
    names = {p["id"]: p["name"] for p in pub["players"]}
    me = game.players[pid]
    my_origin = cfg.origin(priv.get("origin")) or {}

    rows = []
    for p in pub["players"]:
        o = cfg.origin(p.get("origin")) or {}
        # 按出身算（官二代的政绩门槛打了折）——用通用门槛的话，大家会低估官二代的进度
        import rules
        tc = rules.merit_cost_at(p["rank"], p.get("origin"), cfg)
        mc = rules.money_cost_at(p["rank"], p.get("origin"), cfg)
        est = ""
        if p["id"] != pid:
            m = agent.models.get(p["id"])
            if m is not None:
                est = f"，估计存款约 {m.money_est:.0f}（程序估算，不一定准）"
        rows.append(
            f"  - {p['name']}（id={p['id']}，{o.get('name', '?')}）：{cfg.rank_name(p['rank'])}，"
            f"政绩 {p['merit']}" + (f"/{tc}（他升下一级要：政绩 {tc} 或 金钱 {mc}）" if tc else "")
            + f"，降职警告 {p.get('warnings', 0)}{est}" + ("  ← 你" if p["id"] == pid else "")
        )
    threat_rows = []
    top = cfg.president_rank
    for p in pub["players"]:
        if p["id"] == pid:
            continue
        mc = rules.money_cost_at(p["rank"], p.get("origin"), cfg)
        tc = rules.merit_cost_at(p["rank"], p.get("origin"), cfg)
        if tc is None:
            continue
        m = agent.models.get(p["id"], ai.OpponentModel())
        steps = top - p["rank"]
        gap_merit = max(0, tc - p["merit"])
        gap_money = max(0.0, (mc or 0) - m.money_est)
        flag = ""
        if p["rank"] == top - 1:
            near = p["merit"] + agent._expected_work(p["rank"]) >= tc
            if agent._about_to_win(p, m) or near:
                flag = "  ⚠ 这一轮就可能登顶"
        need = "钱和政绩都要" if cfg.needs_both(p["rank"]) else "政绩或金钱任一"
        threat_rows.append((steps, gap_merit, f"  - {p['name']}（id={p['id']}）：离主席还差 {steps} 级；"
                            f"本级（{need}）还差政绩 {gap_merit}、估计还差金钱 {gap_money:.0f}{flag}"))
    threat_rows.sort(key=lambda r: (r[0], r[1]))
    threat = "\n".join(r[2] for r in threat_rows)
    hand = "\n".join(
        f"  [{i}] {CARD_NAMES.get(d['card'], d['card'])}" + (f"（点数 {d['value']}）" if d.get("value") else "")
        for i, d in enumerate(priv["hand"])
    )
    fam = priv.get("family_card") or {}
    family_line = ("\n你还有一张没用过的「一纸调令」（每局一次，不占出牌位）：要用就在 JSON 里写 \"family\": true。"
                   if fam.get("usable") else "")
    tip = priv.get("tipoff_event")
    tip_line = f"\n家里透风：本轮的全局事件是「{tip.get('name')}」——{tip.get('description', '')}" if tip else ""
    np_ = priv.get("next_promotion") or {}
    redraw_line = ""
    if priv.get("redraw_available"):
        cost, nxt = priv.get("redraw_cost", 0), priv.get("redraw_next_cost", 0)
        redraw_line = (f"\n换牌：可以花 {cost} 金钱把这 6 张全部换掉重抽（同一轮再换要 {nxt}）"
                       + ("" if priv.get("redraw_affordable") else "——你现在钱不够")
                       + "。想换就只回 {\"redraw\": true}，我会把新手牌发给你再决定。")
    hist = "\n".join(history[-4:]) or "（第一轮，还没有历史）"
    mem = "\n".join(memory[-3:]) or "（无）"
    return f"""{rules_text(cfg)}

====================
你是 {names[pid]}，出身「{my_origin.get('name', '?')}」。现在是第 {pub['round']} / {pub['max_rounds']} 轮。
你自己：{cfg.rank_name(me.rank)}，政绩 {me.merit}，金钱 {me.money}（只有你知道），降职警告 {me.warnings}。
下一级门槛：政绩 {np_.get('merit', cfg.merit_cost(me.rank))} / 金钱 {np_.get('money', cfg.money_cost(me.rank))}。{tip_line}

场上所有人：
{chr(10).join(rows)}

谁离赢最近（程序按公开信息算的，钱是估计）：
{threat}

最近几轮发生了什么：
{hist}

你前几轮的想法（自己的笔记）：
{mem}

你这轮的手牌：
{hand}{family_line}{redraw_line}

请像一个想赢的真人玩家那样分析局势：先看谁离赢最近、你这一轮能不能赢；
不能赢的话，是自己发展，还是去拦那个快赢的人（他赢了你就输了）。然后决定出牌。
最多出 {cfg.picks_per_round} 张，按结算顺序排列；举报和攻击要写目标的 id。
**只输出一个 JSON**，不要别的文字：
{{"threat": 你认为离赢最近的对手id, "picks": [{{"index": 手牌序号, "target": 目标id或null}}, ...], "family": false, "reason": "一两句话：谁最可能先赢、你这轮能不能赢、为什么这样出"}}"""


def session_system_prompt(cfg: Config, name: str, origin_name: str, notebook: str = "") -> str:
    memo = (f"""

【你过去对局积累的心得（笔记本）】
这是你之前打过的很多局总结下来的经验，参考着用，但不一定全对，以眼前的局面为准：
{notebook}
""" if notebook.strip() else "")
    return f"""你在玩一个桌游，扮演其中一名玩家，目标是赢。不要使用任何工具，直接回答。

{rules_text(cfg)}{memo}

====================
你是 {name}，出身「{origin_name}」。每轮我会告诉你最新的局面和你的手牌，
你像一个想赢的真人玩家那样分析局势：先看谁离赢最近、你这一轮能不能赢；
不能赢的话，是自己发展，还是去拦那个快赢的人（他赢了你就输了）。然后出牌。
最多出 {cfg.picks_per_round} 张，按结算顺序排列；举报和攻击要写目标的 id。
出牌时**只输出一个 JSON**，不要别的文字：
{{"threat": 你认为离赢最近的对手id, "picks": [{{"index": 手牌序号, "target": 目标id或null}}, ...], "family": false, "reason": "一两句话：谁最可能先赢、你这轮能不能赢、为什么这样出"}}"""


def session_round_prompt(cfg: Config, game: Game, pid: int, agent: ai.SmartAgent,
                         last_round: str) -> str:
    """会话模式每轮发的内容：上一轮发生了什么 + 现在的局面 + 手牌（规则和更早的经过它记得）。"""
    full = player_prompt(cfg, game, pid, agent, [last_round] if last_round else [], [])
    body = full[full.index("现在是第"):]
    body = body.replace("你前几轮的想法（自己的笔记）：\n（无）\n\n", "")
    return "【新一轮】" + body


def feedback_prompt(cfg: Config, game: Game, pid: int, history: list[str], memory: list[str]) -> str:
    pub = game.public_state()
    names = {p["id"]: p["name"] for p in pub["players"]}
    me = game.players[pid]
    finals = "\n".join(
        f"  - {p['name']}（{(cfg.origin(p.get('origin')) or {}).get('name', '?')}）：{cfg.rank_name(p['rank'])}"
        + ("  ★ 冠军" if p["id"] in game.winners else "")
        for p in pub["players"]
    )
    return f"""{rules_text(cfg)}

====================
你刚以 {names[pid]}（出身「{(cfg.origin(me.origin.value if me.origin else None) or {}).get('name', '?')}」）的身份打完一局。
结局：{game.game_over_reason}
最终官职：
{finals}

这局的经过（每轮公开信息）：
{chr(10).join(history)}

你每轮的想法：
{chr(10).join(memory)}

请以玩家的身份给游戏设计者写赛后反馈（要具体、说真实感受，可以批评）。**只输出一个 JSON**：
{{"strongest": "你觉得最强的出身", "strongest_why": "为什么",
  "weakest": "你觉得最弱的出身", "weakest_why": "为什么",
  "unfair": "你觉得哪条规则不公平或有问题（没有就写无）",
  "fun": 1到10的整数（这局好不好玩）, "experience": "这局的体验：哪里爽、哪里憋屈",
  "suggestion": "你最想改的一条规则"}}"""


FEEDBACK_IN_SESSION = """【对局结束】{reason}
最后一轮：{last}

请以玩家的身份给游戏设计者写赛后反馈（要具体、说真实感受，可以批评）。**只输出一个 JSON**：
{{"strongest": "你觉得最强的出身", "strongest_why": "为什么",
  "weakest": "你觉得最弱的出身", "weakest_why": "为什么",
  "unfair": "你觉得哪条规则不公平或有问题（没有就写无）",
  "fun": 1到10的整数（这局好不好玩）, "experience": "这局的体验：哪里爽、哪里憋屈",
  "suggestion": "你最想改的一条规则"}}"""


LESSONS_IN_SESSION = """【对局结束】{reason}
最后一轮：{last}

这局打完了。请总结 1~3 条**下一局能用上的心得**（具体、可执行：什么局面该怎么打、哪个身份该怎么用 /
怎么对付、哪种判断这局被证明是错的）。**只输出一个 JSON**：
{{"lessons": ["……", "……"]}}"""

FEEDBACK_AND_LESSONS_IN_SESSION = FEEDBACK_IN_SESSION.replace(
    '"suggestion": "你最想改的一条规则"}}',
    '"suggestion": "你最想改的一条规则",\n  "lessons": ["下一局能用上的心得，1~3 条，具体可执行"]}}')

NOTEBOOK_LIMIT = 2600  # 字；初始版（入门套路）约 1800 字，留出学新东西的空间


class Notebook:
    """六个座位共用的跨局笔记本：每局开局读最新版，打完把六份心得合并进去（加锁，同时结束的几局不会互相覆盖）。"""

    def __init__(self, out_dir: Path, model: str, backend_name: str, seed: str = "") -> None:
        import threading

        self.path = out_dir / "notebook.md"
        self.history = out_dir / "notebook_history"
        self.history.mkdir(exist_ok=True)
        # 初始版：从 Python AI 的训练经验写的入门套路（llm_notebook_seed.md），让它一上来不是纯新手
        if not self.path.exists() and seed and Path(seed).exists():
            text = Path(seed).read_text(encoding="utf-8")
            self.path.write_text(text, encoding="utf-8")
            (self.history / "seed.md").write_text(text, encoding="utf-8")
        self.lock = threading.Lock()
        self.mock = backend_name == "mock"
        self.curator = None if self.mock else ClaudeCLI(model, timeout=300)
        self.version = len(list(self.history.glob("after_game_*.md")))

    def read(self) -> tuple[str, int]:
        with self.lock:
            text = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
            return text, self.version

    def merge(self, game_no: int, summary: str, lessons: list[str], cfg: Config) -> str:
        with self.lock:
            old = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
            if self.mock:
                new = (old + "\n" + f"- 第 {game_no} 局：" + "；".join(lessons[:2]))[-NOTEBOOK_LIMIT:]
            else:
                prompt = (rules_text(cfg) + f"""

====================
你在维护一本**跨局心得笔记本**：同一个 AI 打了很多局这个游戏，每局结束后六个座位各写几条心得，你负责把它们
合并进笔记本，让下一局的自己打得更好。

现在的笔记本：
{old or "（还是空的）"}

刚结束的这一局：{summary}
六个座位这局写的心得：
""" + "\n".join(f"- {x}" for x in lessons) + f"""

请输出**更新后的完整笔记本**（Markdown，{NOTEBOOK_LIMIT} 字以内，可以比现在长一点但别超），分三节：
## 通用策略
## 各身份怎么打 / 怎么对付
## 容易犯的错
要求：合并重复的；被多局证明有效的留下、措辞更精炼；和新证据矛盾的改掉或删掉；具体、可执行，不要空话。
只输出笔记本正文。""")
                new = self.curator.ask(prompt).strip()[:NOTEBOOK_LIMIT + 500]
            self.path.write_text(new, encoding="utf-8")
            self.version += 1
            (self.history / f"after_game_{game_no:03d}_v{self.version:03d}.md").write_text(new, encoding="utf-8")
            return new


NOTEBOOK: "Notebook | None" = None


# ---------------------------------------------------------------------------
# 一局
# ---------------------------------------------------------------------------


def round_summary(cfg: Config, game: Game, outcome) -> str:
    names = {p.id: p.name for p in game.players.values()}
    msgs = list(outcome.public_messages) + list(outcome.wealth_broadcast)
    ranks = "、".join(f"{names[p.id]} {cfg.rank_name(p.rank)}" for p in game.ordered_players())
    return (f"第 {outcome.round_number} 轮（事件：{outcome.event.name}）："
            + ("；".join(msgs) if msgs else "风平浪静") + f"。结束后：{ranks}")


MAX_REDRAWS = 3


def decide(cfg: Config, game: Game, pid: int, backend, pool: ai.AgentPool,
           history: list[str], memory: list[str], stats: Counter, mock: bool,
           session: "ClaudeSession | None" = None,
           extra: dict[str, Any] | None = None) -> tuple[list, str, int]:
    """返回 (出牌, 理由, 这轮换了几次牌)。换不换牌由大模型自己决定（mock 时交给程序 AI）。

    extra 给了的话顺手记下：同一手牌程序 AI 会怎么打（base_picks）、当时的手牌。"""
    agent = pool.get(pid)
    if mock:
        picks = ai.turn(game, pid, pool)  # 程序 AI 冒充：连换牌一起它来
        stats["prompt_chars"] += len(player_prompt(cfg, game, pid, agent, history, memory))
        if extra is not None:
            extra["base_picks"] = [{"action": n, "target": b.get("target")}
                                   for n, b in zip(T.card_names(picks), picks)]
        return picks, "（mock：程序 AI 出牌）", 0
    base = ai.choose(game, pid, pool)  # 只为让程序 AI 的观察跟上（提示里的估计存款要用），不换牌
    if extra is not None:
        extra["base_picks"] = [{"action": n, "target": b.get("target")}
                               for n, b in zip(T.card_names(base), base)]

    def make_prompt(after_redraw: bool) -> str:
        if session is not None:
            p = session_round_prompt(cfg, game, pid, agent, "" if after_redraw else
                                     (history[-1] if history else ""))
            return p.replace("【新一轮】", "【换牌后的新手牌】") if after_redraw else p
        return player_prompt(cfg, game, pid, agent, history, memory)

    ask = (session or backend).ask
    prompt = make_prompt(False)
    redraws = errors = 0
    for _ in range(MAX_REDRAWS + 3):
        try:
            ans = extract_json(ask(prompt))
            if ans.get("redraw"):
                priv = game.private_state(pid)
                if redraws < MAX_REDRAWS and priv.get("redraw_affordable"):
                    stats["redraw_spent"] += game.redraw(pid)
                    stats["redraws"] += 1
                    redraws += 1
                    prompt = make_prompt(True)
                else:
                    prompt = "（现在不能再换牌了：钱不够或者这轮已经换了好几次。请直接出牌，只输出出牌的 JSON。）"
                continue
            hand = game.private_state(pid)["hand"]
            picks = []
            for p in ans.get("picks", [])[: cfg.picks_per_round]:
                i = int(p["index"])
                if not 0 <= i < len(hand):
                    raise ValueError(f"手牌序号 {i} 不存在")
                picks.append({"index": i, "action": hand[i]["card"], "target": p.get("target")})
            if ans.get("family"):
                picks.append({"action": "PROMOTE_FAMILY", "target": None})
            game.select_actions(pid, picks)  # 不合法会抛 GameError
            stats["llm_ok"] += 1
            reason = str(ans.get("reason", ""))[:300]
            if ans.get("threat") is not None:
                reason = f"[威胁:{ans.get('threat')}] " + reason
            if extra is not None:
                extra["hand"] = [d["card"] for d in hand]
            return picks, reason, redraws
        except (ValueError, KeyError, TypeError, GameError, RuntimeError,
                subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            errors += 1
            stats["llm_retry" if errors == 1 else "llm_fallback"] += 1
            if errors >= 2:
                break
            prompt = f"（上一次的回答有问题：{exc}。请严格只输出符合格式的 JSON。）" if session else \
                prompt + f"\n\n（上一次的回答有问题：{exc}。请严格只输出符合格式的 JSON。）"
    return ai.choose(game, pid, pool), "（模型没给出合法出牌，这一轮由程序 AI 代打）", redraws


def play_one(args: tuple) -> dict[str, Any]:
    """打一局。args 的前 9 项含义不变；第 10 项是可选的 opts：

        llm_seats  大模型坐几个座位（默认 6 = 全大模型桌；1 = 混坐：1 个大模型 + 5 个 Python 陪练）
        crowd      混坐时陪练用什么 AI：learned（policies/best.json）| smart（手写）
        control    混坐时每局再跑几份 Python 对照局（同一个种子，大模型的座位换成学习型 AI）
        base_seed  混坐时的焦点分层种子（第 g 局被测身份 = ORIGINS[g % 6]、座位轮换，见 telemetry.py）

    混坐局的发牌、事件按 (种子, 轮次) 重新播种，陪练先出、大模型最后出，所以和对照局逐局可比。
    """
    g, seed, model, backend_name, out_dir = args[:5]
    extra = list(args[5]) if len(args) > 5 else []
    variant = args[6] if len(args) > 6 else "base"
    want_feedback = args[7] if len(args) > 7 else True
    opts = dict(args[9]) if len(args) > 9 and args[9] else {}
    llm_seats = int(opts.get("llm_seats", N))
    mixed = llm_seats < N
    if mixed and llm_seats != 1:
        raise ValueError("混坐目前只支持 1 个大模型座位（--llm-seats 1）")
    crowd = opts.get("crowd", "learned")
    cfg = variant_cfg(variant)
    rng = random.Random(seed)
    mock = backend_name == "mock"
    backend = MockBackend() if mock else ClaudeCLI(model)
    sessions: dict[int, ClaudeSession] = {}
    names = ["老张", "老李", "老王", "老赵", "老刘", "老陈"]
    focal: dict[str, Any] | None = None
    if mixed:
        import probe  # 延迟导入：probe 依赖 search_ai，全大模型桌用不上

        base_seed = int(opts.get("base_seed", seed // 1000))
        focal_origin, focal_seat, seats = T.focal_assignment(g, base_seed, cfg)
        gs = T.game_seed(base_seed, g)
        game = T.new_game(cfg, seats, gs, game_id=f"llm{g}", names=names)
        llm_pids = {focal_seat + 1}
        focal = {"pid": focal_seat + 1, "origin": focal_origin, "seat": focal_seat, "base_seed": base_seed}
        # 大模型座位的"观察助手"（提示里的估计存款）和对照局里坐这个座位的学习型 AI 同一个种子
        pool = probe.arm_pool(cfg, "learned", random.Random(f"{gs}/base"))
        crowd_pool = probe.arm_pool(cfg, crowd, random.Random(f"{gs}/crowd"))
        order = T.focal_order(N, focal_seat + 1)
    else:
        gs = seed
        game = Game(game_id=f"llm{g}", cfg=cfg, rng=rng)
        for name in names:
            game.add_player(name, is_ai=True)
        origins = list(cfg.origin_ids())
        rng.shuffle(origins)
        for pid, oid in zip(sorted(game.players), origins):
            game.players[pid].origin = Origin(oid)
        pool = ai.make_pool(cfg, rng)
        crowd_pool = None
        llm_pids = set(game.players)
        order = sorted(game.players)
        game.start_game()
    controller = {pid: ("llm" if pid in llm_pids else crowd) for pid in game.players}
    notebook_text, notebook_version = NOTEBOOK.read() if NOTEBOOK is not None else ("", None)
    if backend_name == "session":
        for pid in sorted(llm_pids):
            p = game.players[pid]
            oname = (cfg.origin(p.origin.value) or {}).get("name", "?")
            sessions[pid] = ClaudeSession(
                model, session_system_prompt(cfg, p.name, oname, notebook_text), extra)

    history: list[str] = []
    memory: dict[int, list[str]] = defaultdict(list)
    decisions: list[dict[str, Any]] = []
    tele: list[dict[str, Any]] = []
    stats: Counter = Counter()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N) as ex:
        while not game.is_over:
            rnd = game.round_number
            # 陪练先在主线程按座位出牌（混坐时），大模型的座位再并行去问
            for pid in order:
                if pid in llm_pids:
                    continue
                picks = ai.turn(game, pid, crowd_pool)
                game.select_actions(pid, picks)
                game.lock_action(pid)
                decisions.append({"round": rnd, "pid": pid, "controller": crowd,
                                  "picks": [{"action": n, "target": p.get("target")}
                                            for n, p in zip(T.card_names(picks), picks)],
                                  "reason": "", "redraws": game.redraw_count.get(pid, 0)})
            extras = {pid: {} for pid in llm_pids}
            futs = {pid: ex.submit(decide, cfg, game, pid, backend, pool, history,
                                   memory[pid], stats, mock, sessions.get(pid), extras[pid])
                    for pid in sorted(llm_pids)}
            # 并行时 decide 里已经各自 select_actions 过；mock / 退回程序 AI 的那几席在这里补上
            for pid, f in futs.items():
                picks, reason, redraws = f.result()
                sel = game.selections[pid]
                if not sel.picks:
                    game.select_actions(pid, picks)
                game.lock_action(pid)
                cards = [p.get("action") for p in picks]
                memory[pid].append(f"第 {rnd} 轮：出了 {[CARD_NAMES.get(c, c) for c in cards]}。{reason}")
                d = {"round": rnd, "pid": pid, "picks": picks, "reason": reason, "redraws": redraws}
                if mixed:
                    d["controller"] = "llm"
                d.update(extras[pid])
                decisions.append(d)
            game.force_lock_all()
            game.reveal_event()
            outcome = game.resolve()
            history.append(round_summary(cfg, game, outcome))
            tele.append(T.round_record(game, outcome, controller))
            if not game.is_over:
                if mixed:
                    T.reseed_round(game, gs)
                game.advance_round()

        fb_futs = {}
        learning = NOTEBOOK is not None
        if sessions and (want_feedback or learning):
            # 会话模式：整局都在它的上下文里，只要告诉它结局。
            # 笔记本模式每局都要心得；收反馈的局心得和反馈一起交
            tpl = (FEEDBACK_AND_LESSONS_IN_SESSION if learning else FEEDBACK_IN_SESSION) \
                if want_feedback else LESSONS_IN_SESSION
            fb_futs = {pid: ex.submit(sessions[pid].ask, tpl.format(
                           reason=game.game_over_reason, last=history[-1] if history else ""))
                       for pid in sorted(sessions)}
        elif want_feedback:
            fb_futs = {pid: ex.submit(backend.ask, feedback_prompt(cfg, game, pid, history, memory[pid]))
                       for pid in sorted(llm_pids)}
        feedback = {}
        for pid, f in fb_futs.items():
            try:
                feedback[pid] = extract_json(f.result())
            except Exception as exc:  # noqa: BLE001 反馈拿不到不影响对局数据
                feedback[pid] = {"error": str(exc)}
    lessons: list[str] = []
    if NOTEBOOK is not None:
        for pid, fb in feedback.items():
            o = game.players[pid]
            tag = f"[{(cfg.origin(o.origin.value) or {}).get('name', '?')}{'·冠军' if pid in game.winners else ''}] "
            lessons += [tag + str(x) for x in (fb.get("lessons") or [])[:3]]
        if not want_feedback:  # 只要了心得的局，不当反馈统计
            feedback = {}
        if mock:
            lessons = lessons or [f"[mock] 第 {g} 局的心得"]
        winner_txt = "、".join(f"{game.players[w].name}（{(cfg.origin(game.players[w].origin.value) or {}).get('name')}）"
                              for w in game.winners)
        try:
            NOTEBOOK.merge(g, f"{game.round_number} 轮结束，{winner_txt}获胜。{game.game_over_reason}",
                           lessons, cfg)
        except Exception as exc:  # noqa: BLE001 笔记本更新失败不影响对局数据
            stats["notebook_errors"] += 1
            lessons.append(f"（笔记本更新失败：{exc}）")
    for s in sessions.values():
        backend.calls += s.calls
        backend.seconds += s.seconds
        backend.retries += s.retries
        backend.restarts += s.restarts
        s.close()
    stats["gateway_retries"] += backend.retries
    stats["session_restarts"] += backend.restarts

    summ = T.game_summary(game, controller)
    place = {p["pid"]: p["place"] for p in summ["players"]}
    players = []
    for p in game.ordered_players():
        players.append({
            "pid": p.id, "name": p.name, "origin": p.origin.value if p.origin else None,
            "rank": p.rank, "money": p.money, "merit": p.merit,
            "winner": p.id in game.winners, "feedback": feedback.get(p.id),
            "controller": controller[p.id], "place": place[p.id], "warnings": p.warnings,
        })
    result = {
        "schema": 2, "game": g, "seed": seed, "model": model, "variant": variant, "rounds": game.round_number,
        "president": any(p.rank >= cfg.president_rank for p in game.players.values()),
        "end": summ["end"], "winners": summ["winners"], "final_standing": summ["final_standing"],
        "llm_seats": len(llm_pids), "crowd": crowd if mixed else None, "focal": focal,
        "rules_fp": T.rules_fingerprint(cfg),
        "reason": game.game_over_reason, "players": players, "history": history,
        "decisions": decisions, "telemetry": tele, "stats": dict(stats),
        "notebook_version": notebook_version, "lessons": lessons,
        "llm_calls": backend.calls, "llm_seconds": round(backend.seconds, 1),
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out_dir / f"game_{g:03d}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    if mixed:
        for k in range(int(opts.get("control", 0))):
            write_control(out_dir, cfg, g, focal["base_seed"], crowd, k)
    return result


def write_control(out_dir: Path, cfg: Config, g: int, base_seed: int, crowd: str, k: int) -> dict[str, Any]:
    """Python 对照局：和第 g 局大模型混坐局同身份、同座位、同种子，大模型的座位换成学习型 AI。
    k = 0 用同一套陪练种子（逐局配对），k > 0 只换陪练的随机数（多几份样本）。"""
    import probe
    import search_ai

    r = probe.play_probe_game(cfg, g, base_seed, "learned", crowd, search_ai.SearchConfig(budget=0),
                              want_telemetry=True, crowd_tag="" if k == 0 else str(k))
    r.update({"schema": 2, "control": k, "rules_fp": T.rules_fingerprint(cfg)})
    (out_dir / f"control_{g:03d}_{k}.json").write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    return r


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def _origin_in(text: str, origin_name: dict[str, str]) -> str:
    """反馈里写的身份（"红二代「硬保」""富二代（老刘）"……）归到标准身份名；认不出就原样。"""
    best, pos = text, 10 ** 9
    for name in origin_name.values():
        for key in {name, name.split("·")[-1]}:
            i = text.find(key)
            if i != -1 and i < pos:
                best, pos = name, i
    return best


def report(out_dir: Path, summarize: bool, model: str, backend_name: str) -> str:
    games = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("game_*.json"))]
    if not games:
        return "没有对局数据"
    variant = games[0].get("variant", "base")
    cfg = variant_cfg(variant)
    redraw_by: Counter = Counter()
    atk_recv: Counter = Counter()
    rep_recv: Counter = Counter()
    n = len(games)
    wins: Counter = Counter()
    seats: Counter = Counter()
    rank_sum: Counter = Counter()
    votes_strong: Counter = Counter()
    votes_weak: Counter = Counter()
    fun_by_origin: dict[str, list[int]] = defaultdict(list)
    fun_by_result: dict[str, list[int]] = defaultdict(list)
    cards: Counter = Counter()
    stats: Counter = Counter()
    texts: list[str] = []
    origin_name = {o["id"]: o["name"] for o in cfg.origin_definitions}
    # 混坐局（1 个大模型 + 5 个 Python）：身份战绩、出牌习惯只统计大模型的座位
    mixed = any(gm.get("llm_seats", N) < N for gm in games)
    for gm in games:
        stats.update(gm.get("stats", {}))
        winners = [p for p in gm["players"] if p["winner"]]
        for p in gm["players"]:
            if mixed and p.get("controller") != "llm":
                continue
            o = origin_name.get(p["origin"], p["origin"])
            seats[o] += 1
            rank_sum[o] += p["rank"]
            if p["winner"]:
                wins[o] += 1.0 / max(1, len(winners))
            fb = p.get("feedback") or {}
            if "strongest" in fb:
                votes_strong[_origin_in(str(fb["strongest"]), origin_name)] += 1
                votes_weak[_origin_in(str(fb["weakest"]), origin_name)] += 1
                try:
                    fun = int(fb.get("fun"))
                    fun_by_origin[o].append(fun)
                    fun_by_result["冠军" if p["winner"] else "其他人"].append(fun)
                except (TypeError, ValueError):
                    pass
                texts.append(
                    f"[{o}{'·冠军' if p['winner'] else ''}] 最强 {fb.get('strongest')}（{fb.get('strongest_why', '')}）；"
                    f"最弱 {fb.get('weakest')}（{fb.get('weakest_why', '')}）；不公平：{fb.get('unfair', '')}；"
                    f"体验 {fb.get('fun')}/10：{fb.get('experience', '')}；建议：{fb.get('suggestion', '')}"
                )
        origin_of = {p["pid"]: origin_name.get(p["origin"], p["origin"]) for p in gm["players"]}
        for d in gm["decisions"]:
            if mixed and d.get("controller", "llm") != "llm":
                continue
            redraw_by[origin_of.get(d["pid"], "?")] += d.get("redraws", 0)
            for p in d["picks"]:
                cards[p.get("action")] += 1
                t = p.get("target")
                if t in origin_of and p.get("action") in ("ATTACK", "REPORT"):
                    (atk_recv if p["action"] == "ATTACK" else rep_recv)[origin_of[t]] += 1

    # 拦人：攻击打在省级身上的比例、第一个到省级的人夺冠率（从每轮"结束后：…"的官职里推）
    top_name = cfg.rank_name(cfg.president_rank - 1)
    atk_total = atk_top = first_n = first_won = 0
    for gm in games:
        names = {p["pid"]: p["name"] for p in gm["players"]}
        rank_after: dict[int, dict[str, str]] = {}
        for line in gm["history"]:
            m = re.match(r"第 (\d+) 轮.*结束后：(.*)$", line)
            if m:
                rank_after[int(m.group(1))] = dict(
                    x.split(" ", 1) for x in m.group(2).split("、") if " " in x)
        for d in gm["decisions"]:
            prev = rank_after.get(d["round"] - 1, {})
            for p in d["picks"]:
                if p.get("action") == "ATTACK" and p.get("target") in names:
                    atk_total += 1
                    atk_top += prev.get(names[p["target"]]) == top_name
        for r in sorted(rank_after):
            tops = [nm for nm, rk in rank_after[r].items() if rk in (top_name, cfg.rank_name(cfg.president_rank))]
            if tops:
                first_n += 1
                first_won += any(p["winner"] and p["name"] == tops[0] for p in gm["players"])
                break

    lines = [f"# 大模型对局小样本报告（规则变体 {variant}，{n} 局，{model}，{backend_name}）", ""]
    lines.append(f"- 主席率 {100 * sum(g['president'] for g in games) / n:.0f}%，"
                 f"平均 {sum(g['rounds'] for g in games) / n:.1f} 轮结束")
    total_dec = stats["llm_ok"] + stats["llm_fallback"]
    if total_dec:
        lines.append(f"- 模型出牌 {stats['llm_ok']} 次，重试 {stats['llm_retry']} 次，"
                     f"退回程序 AI {stats['llm_fallback']} 次（{100 * stats['llm_fallback'] / total_dec:.1f}%）")
    lines.append(f"- 拦人：攻击里打在{top_name}身上的 {100 * atk_top / max(1, atk_total):.0f}%；"
                 f"第一个到{top_name}的人最终夺冠 {100 * first_won / max(1, first_n):.0f}%"
                 f"（公平线约 17%，越高说明领先者越没被拦住）")
    calls = sum(g["llm_calls"] for g in games)
    secs = sum(g["llm_seconds"] for g in games)
    lines.append(f"- 模型调用 {calls} 次，平均每次 {secs / max(1, calls):.1f} 秒")
    ref_path = RUN_DIR / "python_reference.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else {}
    ref_win = ref.get(variant, {}).get("win", {})
    ta, tr = sum(atk_recv.values()) or 1, sum(rep_recv.values()) or 1
    if mixed:
        lines += [""] + mixed_origin_table(games, out_dir, cfg, fun_by_origin, votes_strong, votes_weak)
    else:
        lines += _symmetric_origin_table(n, seats, wins, ref_win, rank_sum, atk_recv, rep_recv, ta, tr,
                                         fun_by_origin, votes_strong, votes_weak)
    lines += ["", "## 出牌习惯（" + ("大模型座位" if mixed else "所有玩家") + "打出的牌）", ""]
    tot = sum(cards.values()) or 1
    lines.append("、".join(f"{CARD_NAMES.get(c, c)} {100 * v / tot:.0f}%" for c, v in cards.most_common()))
    if stats["redraws"]:
        lines += ["", f"换牌：共 {stats['redraws']} 次、花了 {stats['redraw_spent']} 金钱；每个身份平均每局换牌 "
                  + "、".join(f"{o} {redraw_by[o] / max(1, seats[o]):.1f} 次" for o in sorted(seats))]
    lines += ["", "## 体验打分", ""]
    for k, v in fun_by_result.items():
        lines.append(f"- {k}：平均 {sum(v) / len(v):.1f} / 10（{len(v)} 份）")
    nb = out_dir / "notebook.md"
    if nb.exists():
        lines += ["", "## 学习曲线（跨局笔记本：按局号每 25 局一段）", "",
                  "| 局号 | 平均局长 | 攻击打在省级 | 第一个到省级的人夺冠 | 换牌/局 | 体验 | 胜率最高的身份 |",
                  "|---|---|---|---|---|---|---|"]
        for lo in range(0, max(g["game"] for g in games) + 1, 25):
            seg = [g for g in games if lo <= g["game"] < lo + 25]
            if not seg:
                continue
            a_t = a_n = f_n = f_w = rd = 0
            seg_fun: list[int] = []
            seg_win: Counter = Counter()
            for gm in seg:
                names = {p["pid"]: p["name"] for p in gm["players"]}
                ra: dict[int, dict[str, str]] = {}
                for line in gm["history"]:
                    m = re.match(r"第 (\d+) 轮.*结束后：(.*)$", line)
                    if m:
                        ra[int(m.group(1))] = dict(x.split(" ", 1) for x in m.group(2).split("、") if " " in x)
                for d in gm["decisions"]:
                    rd += d.get("redraws", 0)
                    for p in d["picks"]:
                        if p.get("action") == "ATTACK" and p.get("target") in names:
                            a_n += 1
                            a_t += ra.get(d["round"] - 1, {}).get(names[p["target"]]) == top_name
                for r in sorted(ra):
                    tops = [nm for nm, rk in ra[r].items() if rk in (top_name, cfg.rank_name(cfg.president_rank))]
                    if tops:
                        f_n += 1
                        f_w += any(p["winner"] and p["name"] == tops[0] for p in gm["players"])
                        break
                w = [p for p in gm["players"] if p["winner"]]
                for p in w:
                    seg_win[origin_name.get(p["origin"], p["origin"])] += 1 / len(w)
                for p in gm["players"]:
                    try:
                        seg_fun.append(int((p.get("feedback") or {}).get("fun")))
                    except (TypeError, ValueError):
                        pass
            best = "、".join(f"{o} {100 * v / len(seg):.0f}%" for o, v in seg_win.most_common(2))
            lines.append(f"| {lo}~{lo + len(seg) - 1} | {sum(g['rounds'] for g in seg) / len(seg):.1f} | "
                         f"{100 * a_t / max(1, a_n):.0f}% | {100 * f_w / max(1, f_n):.0f}% | {rd / len(seg):.1f} | "
                         f"{(sum(seg_fun) / len(seg_fun)) if seg_fun else 0:.1f} | {best} |")
        lines += ["", "## 最终笔记本", "", nb.read_text(encoding="utf-8")]
    lines += ["", "## 投票", "",
              "最强：" + "、".join(f"{k} {v}" for k, v in votes_strong.most_common()),
              "", "最弱：" + "、".join(f"{k} {v}" for k, v in votes_weak.most_common())]
    if summarize and texts and backend_name != "mock":
        ask = ClaudeCLI(model, timeout=900).ask
        task = ("请你作为游戏设计顾问，用中文归纳：1) 大家公认哪个身份强、哪个弱，理由是什么；"
                "2) 反复被提到的不公平规则（大约多少份提到）；3) 体验上的共同痛点和爽点；"
                "4) 最值得改的 3 条建议（按优先级）。要具体、引用反馈里的说法，别空泛。直接输出 Markdown 正文。")
        try:
            # 反馈太多时分批归纳（每批 100 份），再把各批的归纳合并成一份
            chunks = [texts[i:i + 100] for i in range(0, len(texts), 100)]
            parts = [ask(rules_text(cfg) + f"\n\n====================\n下面是一批赛后反馈（{len(c)} 份）。"
                         + task + "\n\n" + "\n".join(c)) for c in chunks]
            final = parts[0] if len(parts) == 1 else ask(
                rules_text(cfg) + f"\n\n====================\n下面是 {len(parts)} 份分批归纳，"
                f"合起来覆盖 {n} 局、{len(texts)} 份赛后反馈。请合并成一份总归纳，" + task
                + "\n\n" + "\n\n---\n\n".join(parts))
            lines += ["", "## 反馈归纳（由模型整理）", "", final]
        except Exception as exc:  # noqa: BLE001
            lines += ["", f"（归纳失败：{exc}）"]
    lines += ["", "## 全部反馈原文", ""] + [f"- {t}" for t in texts]
    text = "\n".join(lines)
    # 带模型归纳的完整报告写 report.md；快速报告（阶段报告 / 预览）写 report_quick.md，
    # 免得覆盖掉之前那份花了几十次调用归纳出来的
    name = "report.md" if (summarize and backend_name != "mock") else "report_quick.md"
    (out_dir / name).write_text(text, encoding="utf-8")
    return text


def _symmetric_origin_table(n, seats, wins, ref_win, rank_sum, atk_recv, rep_recv, ta, tr,
                            fun_by_origin, votes_strong, votes_weak) -> list[str]:
    """全大模型桌：六个座位都算。"""
    lines = ["", f"## 各身份战绩（每个身份约 {n} 局）", "",
             "| 身份 | 胜率 | ±95% | Python AI 同规则 | 平均终局官职 | 挨攻击份额 | 被举报份额 | 玩家打分 | 被评为最强 | 被评为最弱 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for o in sorted(seats, key=lambda k: -wins[k]):
        fun = fun_by_origin.get(o, [])
        p = wins[o] / seats[o]
        half = 196 * (p * (1 - p) / seats[o]) ** 0.5
        lines.append(f"| {o} | {100 * p:.1f}% | ±{half:.1f} | "
                     + (f"{ref_win[o]:.1f}%" if o in ref_win else "—")
                     + f" | {rank_sum[o] / seats[o]:.2f} | {100 * atk_recv[o] / ta:.0f}% | {100 * rep_recv[o] / tr:.0f}% | "
                     f"{(sum(fun) / len(fun)) if fun else 0:.1f} | {votes_strong.get(o, 0)} | {votes_weak.get(o, 0)} |")
    return lines


def load_controls(out_dir: Path) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in sorted(out_dir.glob("control_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        out[r["g"]].append(r)
    return out


def mixed_origin_table(games, out_dir: Path, cfg: Config, fun_by_origin, votes_strong, votes_weak) -> list[str]:
    """混坐局：大模型坐的那个身份 vs 同一局同一座位换成学习型 AI（对照局，逐局配对）。"""
    import balance_stats as bs

    origin_name = {o["id"]: o["name"] for o in cfg.origin_definitions}
    controls = load_controls(out_dir)
    rows: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for gm in games:
        f = gm.get("focal") or {}
        pid = f.get("pid")
        me = next((p for p in gm["players"] if p["pid"] == pid), None)
        if me is None:
            continue
        o = origin_name.get(me["origin"], me["origin"])
        nw = max(1, len(gm.get("winners") or [p for p in gm["players"] if p["winner"]]))
        share = (1.0 / nw) if me["winner"] else 0.0
        r = rows[o]
        r["share"].append(share)
        r["place"].append(me.get("place", 0))
        r["rank"].append(me["rank"])
        r["rounds"].append(gm["rounds"])
        atk = rep = 0
        for t in gm.get("telemetry") or []:
            for pl in t["players"]:
                if pl["pid"] == pid:
                    atk += pl["attacker_count"]
                    rep += pl["report_count_players"]
        r["atk"].append(atk / max(1, gm["rounds"]))
        r["rep"].append(rep / max(1, gm["rounds"]))
        cs = controls.get(gm["game"], [])
        if cs:
            r["ctrl"].append(sum(c["focal_share"] for c in cs) / len(cs))
            c0 = next((c for c in cs if c.get("control") == 0), None)
            if c0 is not None:
                r["paired_llm"].append(share)
                r["paired_ctrl"].append(c0["focal_share"])
                r["paired_place_llm"].append(me.get("place", 0))
                r["paired_place_ctrl"].append(c0["place"])
    n = sum(len(r["share"]) for r in rows.values())
    lines = [f"## 各身份战绩：1 个大模型 + 5 个 Python 陪练（{n} 局，只统计大模型的座位）", "",
             "对照 = 同一局、同一座位换成学习型 AI（同种子），差值逐局配对；公平线 16.67%。", "",
             "| 身份 | 局数 | 大模型胜率 | 对照（学习型）胜率 | 差值 | 大模型平均名次 | 对照平均名次 | 终局官职 | 挨攻击/轮 | 被举报/轮 | 玩家打分 | 被评为最强 | 被评为最弱 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for o in sorted(rows, key=lambda k: -sum(rows[k]["share"]) / len(rows[k]["share"])):
        r = rows[o]
        m, se = bs.mean_se(r["share"])
        cm, cse = bs.mean_se(r["ctrl"]) if r["ctrl"] else (float("nan"), float("nan"))
        d, dse = bs.paired_diff(r["paired_llm"], r["paired_ctrl"]) if r["paired_llm"] else (float("nan"), float("nan"))
        pl, _ = bs.mean_se(r["paired_place_llm"]) if r["paired_place_llm"] else bs.mean_se(r["place"])
        pc, _ = bs.mean_se(r["paired_place_ctrl"]) if r["paired_place_ctrl"] else (float("nan"), 0)
        fun = fun_by_origin.get(o, [])
        dtxt = "—" if d != d else (f"{100 * d:+.1f}" if dse != dse else f"{100 * d:+.1f} ±{100 * bs.Z * dse:.1f}")
        lines.append(
            f"| {o} | {len(r['share'])} | {bs.fmt_pct(m, se)} | {bs.fmt_pct(cm, cse)} | {dtxt} | {pl:.2f} | "
            + ("—" if pc != pc else f"{pc:.2f}")
            + f" | {sum(r['rank']) / len(r['rank']):.2f} | {sum(r['atk']) / len(r['atk']):.2f} | "
            f"{sum(r['rep']) / len(r['rep']):.2f} | {f'{sum(fun) / len(fun):.1f}' if fun else '—'} | "
            f"{votes_strong.get(o, 0)} | {votes_weak.get(o, 0)} |")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--parallel", type=int, default=2, help="同时跑几局（每局 6 个玩家并行思考）")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--backend", choices=["session", "claude", "mock"], default="session",
                    help="session = 每个玩家一个常驻会话（快，推荐）；claude = 每次决策单独调用一次")
    ap.add_argument("--extra", default='--tools ""',
                    help="session 模式额外传给 claude 的参数（默认关掉工具）；不支持就传空字符串")
    ap.add_argument("--selftest", action="store_true", help="测一下哪几种调用方式能用、各要几秒")
    ap.add_argument("--variant", default="base",
                    help="sweep.py 里的规则变体（noreward / off_notip / off34 / peasant_old，可用 + 组合）")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--report", default="", help="只对这个目录出报告")
    ap.add_argument("--no-summary", action="store_true", help="报告里不请模型归纳反馈")
    ap.add_argument("--resume", default="", help="接着这个目录跑：已经跑完的局跳过")
    ap.add_argument("--feedback-every", type=int, default=1,
                    help="每 N 局收一次赛后反馈（局号能被 N 整除的那些局；1 = 每局都收）")
    ap.add_argument("--notebook", action="store_true",
                    help="跨局学习：六个座位共用一本心得笔记本，每局开局读最新版、打完把心得合并进去")
    ap.add_argument("--seed-notebook", default=str(Path(__file__).resolve().parent / "llm_notebook_seed.md"),
                    help="笔记本的初始版（入门套路）；传空字符串就从空白开始")
    ap.add_argument("--llm-seats", type=int, default=N, choices=[1, N],
                    help="大模型坐几个座位：6 = 全大模型桌（默认）；1 = 混坐（1 个大模型 + 5 个 Python 陪练，"
                         "大模型坐的身份按局号轮换，每个身份局数一样）")
    ap.add_argument("--crowd", default="learned", choices=["learned", "smart"],
                    help="混坐时陪练用的 AI")
    ap.add_argument("--control", type=int, default=1,
                    help="混坐时每局附带几份 Python 对照局（同种子，大模型的座位换成学习型 AI；几乎不花时间）")
    ap.add_argument("--min-free-gb", type=float, default=3.0,
                    help="开新一局前系统可用内存低于这么多 GB 就先等（避免几十个 claude 进程把机器拖死）")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest(args.model)
    if args.report:
        print(report(Path(args.report), not args.no_summary, args.model, args.backend))
        return 0
    variant_cfg(args.variant)  # 变体名写错就当场报
    if args.resume:
        out_dir = Path(args.resume)
        done_games = {int(p.stem.split("_")[1]) for p in out_dir.glob("game_*.json")}
        print(f"接着跑：{out_dir}（已完成 {len(done_games)} 局）", flush=True)
    else:
        out_dir = RUN_DIR / (time.strftime("%Y%m%d-%H%M%S") + f"-{args.variant.replace('+', '_')}")
        out_dir.mkdir(parents=True, exist_ok=True)
        done_games = set()
        print(f"输出目录：{out_dir}", flush=True)
    import shlex

    global NOTEBOOK
    if args.notebook:
        NOTEBOOK = Notebook(out_dir, args.model, args.backend, args.seed_notebook)
        print(f"跨局笔记本：{NOTEBOOK.path}（当前第 {NOTEBOOK.version} 版）", flush=True)
    extra = shlex.split(args.extra)
    todo = [g for g in range(args.games) if g not in done_games]
    opts = ({"llm_seats": args.llm_seats, "crowd": args.crowd, "control": args.control,
             "base_seed": args.seed} if args.llm_seats < N else {})
    jobs = [(g, args.seed * 1000 + g, args.model, args.backend, out_dir, extra, args.variant,
             g % max(1, args.feedback_every) == 0, args.min_free_gb, opts)
            for g in todo]
    log = open(out_dir / "progress.log", "a", encoding="utf-8")
    t_start = time.time()
    done = 0
    totals: Counter = Counter()
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        for res in ex.map(lambda j: _safe(_wait_for_memory_then_play, j), jobs):
            done += 1
            elapsed = time.time() - t_start
            rate = done / max(1e-9, elapsed / 60)
            eta = (len(todo) - done) / max(1e-9, rate)
            finished = len(done_games) + done
            if "error" in res:
                totals["game_errors"] += 1
                line = f"[{finished}/{args.games}] 第 {res['game']} 局出错：{res['error'][:300]}"
            else:
                st = res["stats"]
                for k in ("llm_fallback", "llm_retry", "gateway_retries", "session_restarts"):
                    totals[k] += st.get(k, 0)
                w = [p for p in res["players"] if p["winner"]]
                line = (f"[{finished}/{args.games}] 第 {res['game']} 局：{res['rounds']} 轮，冠军 "
                        + "、".join(f"{p['name']}({p['origin']})" for p in w)
                        + f"，用时 {res['wall_seconds']:.0f} 秒")
            line += (f"  | 每分钟 {rate:.2f} 局，预计还要 {eta / 60:.1f} 小时"
                     f" | 退回程序AI {totals['llm_fallback']}、网关重试 {totals['gateway_retries']}、"
                     f"会话重开 {totals['session_restarts']}、整局出错 {totals['game_errors']}"
                     f" | claude 进程内存 {_claude_rss_gb():.1f} GB、系统可用 {_free_gb():.1f} GB")
            print(line, flush=True)
            log.write(time.strftime("%H:%M:%S ") + line + "\n")
            log.flush()
            if finished % 100 == 0:
                report(out_dir, False, args.model, args.backend)  # 阶段报告：不调模型，秒出
    log.close()
    print("\n" + report(out_dir, not args.no_summary, args.model, args.backend))
    return 0


def _free_gb() -> float:
    """macOS 可用内存（空闲 + 可回收），GB。拿不到就返回一个很大的数（不拦）。"""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        page = int(re.search(r"page size of (\d+)", out).group(1))
        pages = sum(int(m) for k, m in re.findall(r"Pages (free|inactive|speculative|purgeable):\s+(\d+)", out))
        return pages * page / 1024 ** 3
    except Exception:  # noqa: BLE001
        return 999.0


def _claude_rss_gb() -> float:
    try:
        out = subprocess.run(["ps", "-axo", "rss=,comm="], capture_output=True, text=True, timeout=5).stdout
        return sum(int(l.split()[0]) for l in out.splitlines() if "claude" in l) / 1024 ** 2
    except Exception:  # noqa: BLE001
        return 0.0


def _wait_for_memory_then_play(job: tuple) -> dict[str, Any]:
    min_free = job[8] if len(job) > 8 else 0.0
    while min_free and _free_gb() < min_free:
        time.sleep(30)
    return play_one(job)


def selftest(model: str) -> int:
    """依次试几种调用方式，打印能不能用、每次几秒。"""
    cfg = DEFAULT_CONFIG
    sysp = session_system_prompt(cfg, "测试员", "贫农")
    q1 = "【测试】请只输出 JSON：{\"picks\": [], \"reason\": \"测试\"}"
    q2 = "【测试】再来一次，只输出 JSON：{\"picks\": [], \"reason\": \"还记得你是谁吗\"}"
    print(f"模型：{model}")
    try:
        t = time.time()
        ClaudeCLI(model).ask(q1)
        print(f"  [单次调用]           能用，{time.time() - t:.1f} 秒/次")
    except Exception as exc:  # noqa: BLE001
        print(f"  [单次调用]           不能用：{exc}")
    for label, extra in (("常驻会话 + 关工具", ["--tools", ""]), ("常驻会话", [])):
        try:
            s = ClaudeSession(model, sysp, extra, timeout=180)
            t = time.time()
            s.ask(q1)
            t1 = time.time() - t
            t = time.time()
            ans = s.ask(q2)
            t2 = time.time() - t
            s.close()
            print(f"  [{label:<12}] 能用，第一次 {t1:.1f} 秒、第二次 {t2:.1f} 秒  回答：{ans[:60]!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{label:<12}] 不能用：{str(exc)[:200]}")
    print("\n常驻会话 + 关工具能用就直接跑：  python3 llm_play.py --games 10 --parallel 3")
    print("只有常驻会话能用：                python3 llm_play.py --games 10 --parallel 3 --extra ''")
    print("都不能用就退回单次调用：          python3 llm_play.py --games 10 --backend claude")
    return 0


def _safe(fn, job):
    try:
        return fn(job)
    except Exception as exc:  # noqa: BLE001 一局出错不影响其他局
        return {"game": job[0], "error": f"{exc}\n{traceback.format_exc()[-800:]}"}


if __name__ == "__main__":
    raise SystemExit(main())
