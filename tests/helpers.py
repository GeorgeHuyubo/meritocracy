"""测试用的确定性随机源。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



def index_for(distribution: Sequence[tuple[int, int]], value: int) -> int:
    """算出让 weighted_choice 抽到 `value` 所需的 randrange 返回值。"""
    upto = 0
    for v, w in distribution:
        if v == value:
            return upto
        upto += w
    raise ValueError(f"{value} not in distribution")


# 测试用的固定牌面：8~12，期望 10。真实配置的牌面会随平衡调整而变，
# 但这些用例测的是"机制"而不是"牌面数值"，所以钉一份 fixture 让脚本化的骰点稳定。
# 真实配置的期望值另有用例专门守（TestCardDistributions）。
CLASSIC_CARDS: list[tuple[int, int]] = [(8, 1), (9, 2), (10, 2), (11, 2), (12, 1)]


def work_roll(value: int, cfg=None) -> int:
    return index_for(CLASSIC_CARDS if cfg is None else cfg.work_card_distribution, value)


def corrupt_roll(value: int, cfg=None) -> int:
    return index_for(CLASSIC_CARDS if cfg is None else cfg.corrupt_card_distribution, value)


class ScriptedRng:
    """按脚本返回 randrange 结果，脚本用完后固定返回 0。

    只实现 game/rules 用到的接口：randrange / choice。
    """

    def __init__(self, script: Sequence[int] | None = None) -> None:
        self.script = list(script or [])
        self.calls = 0

    def randrange(self, n: int) -> int:
        self.calls += 1
        value = self.script.pop(0) if self.script else 0
        return value % n

    def choice(self, seq):
        return seq[self.randrange(len(seq))]
