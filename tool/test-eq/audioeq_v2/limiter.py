"""
输出限幅器模块（look-ahead-free 峰值限幅器）

用于兜住实时输入中不可预测的瞬态（EQ放大低/高频瞬态后可能短时削波）。
不是为了"变响"，而是防止削波失真被听成噪声。

设计：
- ceiling（上限）：默认 -2 dBFS
- attack：快速跟随峰值上升（1-3ms）
- release：缓慢释放（60-120ms）
- 块间状态保持，无边界不连续
"""

import numpy as np


class StreamingLimiter:
    """流式峰值限幅器 - 块间状态保持"""

    def __init__(self, sr, ceiling_db=-2.0, attack_ms=2.0, release_ms=90.0):
        """
        参数：
            sr: int，采样率
            ceiling_db: float，输出上限 dBFS（默认-2）
            attack_ms: float，攻击时间ms（默认2，建议1~3）
            release_ms: float，释放时间ms（默认90，建议60~120）
        """
        self.sr = sr
        self.ceiling_db = ceiling_db
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self._recompute()
        self.reset()

    def _recompute(self):
        self.ceiling = 10 ** (self.ceiling_db / 20.0)
        self.attack_coeff = 1.0 - np.exp(-1.0 / (self.sr * self.attack_ms / 1000.0))
        self.release_coeff = 1.0 - np.exp(-1.0 / (self.sr * self.release_ms / 1000.0))

    def reset(self):
        """重置状态（每声道当前增益）"""
        self.gains = {}

    def update_params(self, ceiling_db=None, attack_ms=None, release_ms=None):
        """运行时更新参数"""
        if ceiling_db is not None:
            self.ceiling_db = ceiling_db
        if attack_ms is not None:
            self.attack_ms = attack_ms
        if release_ms is not None:
            self.release_ms = release_ms
        self._recompute()

    def process(self, x, channel_idx=0):
        """
        限幅处理。当瞬时幅度超过 ceiling 时，用平滑增益压回，
        attack 快速介入、release 缓慢恢复，避免削波与泵浦。

        参数：
            x: np.ndarray，输入音频块（float32）
            channel_idx: int，声道索引

        返回：
            np.ndarray，限幅后音频块（float32）
        """
        x = np.asarray(x, dtype=np.float32)
        n = len(x)

        if channel_idx not in self.gains:
            self.gains[channel_idx] = 1.0

        gain = self.gains[channel_idx]
        ceiling = self.ceiling
        out = np.empty(n, dtype=np.float32)

        for i in range(n):
            peak = abs(x[i])
            # 目标增益：峰值超过上限时需要衰减
            if peak * gain > ceiling:
                target = ceiling / (peak + 1e-12)
            else:
                target = 1.0

            # attack快速降增益，release缓慢升增益
            if target < gain:
                gain += self.attack_coeff * (target - gain)
            else:
                gain += self.release_coeff * (target - gain)

            out[i] = x[i] * gain

        self.gains[channel_idx] = gain
        # 硬兜底，防止极端瞬态漏过
        np.clip(out, -ceiling, ceiling, out=out)
        return out
