"""
噪声门压缩器模块（温和下行扩展器版本）

将传统硬噪声门改造为温和的 downward expander，专为实时麦克风场景优化，
避免小声说话、环境声、尾音、呼吸声在阈值附近反复触发导致的
“沙沙声一阵一阵”“尾音发毛”问题。

核心设计：
- downward expander：低于阈值只做温和衰减，不完全静音
- hold（保持时间）：门打开后保持一段时间，避免频繁开关
- hysteresis（迟滞）：双阈值，避免阈值附近抖动
- max_attenuation（最大衰减）：背景噪声不会突然消失又出现
- 平滑包络 follower：RMS 或平滑 abs，不逐样本硬判断
- 增益平滑：门开合用 attack/release 平滑，消除爆音
"""

import numpy as np


class StreamingNoiseGateCompressor:
    """流式温和下行扩展器 - 块间状态保持，消除边界不连续

    兼容旧接口（threshold_db/ratio/attack_ms/release_ms/makeup_gain_db），
    并新增 hold/hysteresis/max_attenuation/envelope_mode 等参数。
    """

    def __init__(self, sr, threshold_db=-50, ratio=2.0,
                 attack_ms=15, release_ms=250, makeup_gain_db=0.0,
                 hold_ms=80, max_attenuation_db=15.0, hysteresis_db=3.0,
                 envelope_mode='rms', env_attack_ms=None, env_release_ms=None):
        """
        初始化温和下行扩展器。

        参数：
            sr: int，采样率
            threshold_db: float，扩展阈值（建议 -55~-48dB，默认-50）
            ratio: float，扩展比（建议 1.5~2.2，默认2.0；越大衰减越强）
            attack_ms: float，增益恢复(开门)时间ms（建议10~20，默认15）
            release_ms: float，增益衰减(关门)时间ms（建议180~300，默认250）
            makeup_gain_db: float，补偿增益dB（默认0）
            hold_ms: float，保持时间ms（建议60~100，默认80）
            max_attenuation_db: float，最大衰减量dB（建议12~18，默认15）
            hysteresis_db: float，迟滞量dB（默认3）
            envelope_mode: str，包络检测方式 'rms' 或 'abs'（默认'rms'）
            env_attack_ms: float/None，包络跟踪上升时间ms（默认取较快值）
            env_release_ms: float/None，包络跟踪下降时间ms（默认取较慢值）
        """
        self.sr = sr
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.makeup_gain_db = makeup_gain_db
        self.hold_ms = hold_ms
        self.max_attenuation_db = max_attenuation_db
        self.hysteresis_db = hysteresis_db
        self.envelope_mode = envelope_mode

        # 包络跟踪时间常数（独立于增益平滑，默认较快上升、较慢下降）
        self.env_attack_ms = env_attack_ms if env_attack_ms is not None else 5.0
        self.env_release_ms = env_release_ms if env_release_ms is not None else 50.0

        self._recompute_coeffs()
        self.reset()

    def _recompute_coeffs(self):
        """根据当前时间常数重算平滑系数"""
        sr = self.sr
        # 增益平滑：开门(attack)快，关门(release)慢
        self.gain_attack_coeff = 1.0 - np.exp(-1.0 / (sr * self.attack_ms / 1000.0))
        self.gain_release_coeff = 1.0 - np.exp(-1.0 / (sr * self.release_ms / 1000.0))
        # 包络跟踪
        self.env_attack_coeff = 1.0 - np.exp(-1.0 / (sr * self.env_attack_ms / 1000.0))
        self.env_release_coeff = 1.0 - np.exp(-1.0 / (sr * self.env_release_ms / 1000.0))
        # hold 采样数
        self.hold_samples = int(sr * self.hold_ms / 1000.0)

    def reset(self):
        """重置所有声道状态"""
        # 每声道独立状态：包络、门开合、hold计数、当前增益dB
        self.states = {}

    def _get_state(self, channel_idx):
        if channel_idx not in self.states:
            self.states[channel_idx] = {
                'env': 0.0,          # 包络（rms模式下为均方值，abs模式下为幅度）
                'gate_open': True,   # 门是否打开
                'hold_counter': 0,   # 剩余hold采样数
                'gain_db': 0.0,      # 当前增益(dB)
            }
        return self.states[channel_idx]

    def update_params(self, threshold_db=None, ratio=None, attack_ms=None,
                      release_ms=None, makeup_gain_db=None,
                      hold_ms=None, max_attenuation_db=None, hysteresis_db=None):
        """
        运行时更新参数。

        参数：
            threshold_db/ratio/makeup_gain_db: 同构造函数
            attack_ms/release_ms: 增益开合平滑时间（会重算系数）
            hold_ms/max_attenuation_db/hysteresis_db: 扩展器专有参数
        """
        if threshold_db is not None:
            self.threshold_db = threshold_db
        if ratio is not None:
            self.ratio = ratio
        if makeup_gain_db is not None:
            self.makeup_gain_db = makeup_gain_db
        if max_attenuation_db is not None:
            self.max_attenuation_db = max_attenuation_db
        if hysteresis_db is not None:
            self.hysteresis_db = hysteresis_db

        recompute = False
        if attack_ms is not None:
            self.attack_ms = attack_ms
            recompute = True
        if release_ms is not None:
            self.release_ms = release_ms
            recompute = True
        if hold_ms is not None:
            self.hold_ms = hold_ms
            recompute = True
        if recompute:
            self._recompute_coeffs()

    def process(self, x, channel_idx=0):
        """
        处理音频块。

        参数：
            x: np.ndarray，输入音频块（float32）
            channel_idx: int，声道索引（默认0）

        返回：
            np.ndarray，处理后音频块（float32）
        """
        x = np.asarray(x, dtype=np.float32)
        n = len(x)

        st = self._get_state(channel_idx)
        env = st['env']
        gate_open = st['gate_open']
        hold_counter = st['hold_counter']
        gain_db = st['gain_db']

        threshold_db = self.threshold_db
        close_thr = threshold_db - self.hysteresis_db
        ratio = self.ratio
        max_atten = self.max_attenuation_db
        is_rms = (self.envelope_mode == 'rms')

        gains = np.empty(n, dtype=np.float32)

        for i in range(n):
            xi = x[i]
            # 1. 包络跟踪（平滑，避免逐样本硬判断）
            inst = xi * xi if is_rms else abs(xi)
            if inst > env:
                env += self.env_attack_coeff * (inst - env)
            else:
                env += self.env_release_coeff * (inst - env)

            level = np.sqrt(env) if is_rms else env
            level_db = 20.0 * np.log10(level + 1e-10)

            # 2. 门状态机（迟滞 + hold）
            if level_db > threshold_db:
                gate_open = True
                hold_counter = self.hold_samples
            elif level_db < close_thr:
                if hold_counter > 0:
                    hold_counter -= 1
                    gate_open = True
                else:
                    gate_open = False
            else:
                # 迟滞区间：保持先前状态，仍消耗hold
                if hold_counter > 0:
                    hold_counter -= 1

            # 3. 目标增益（downward expander，温和衰减）
            if gate_open:
                target_gain_db = 0.0
            else:
                below = threshold_db - level_db  # 低于阈值的量（正）
                target_gain_db = -min(below * (ratio - 1.0), max_atten)

            # 4. 增益平滑（关门用release慢速淡出，开门用attack快速淡入）
            if target_gain_db < gain_db:
                gain_db += self.gain_release_coeff * (target_gain_db - gain_db)
            else:
                gain_db += self.gain_attack_coeff * (target_gain_db - gain_db)

            gains[i] = 10.0 ** (gain_db / 20.0)

        # 保存状态
        st['env'] = env
        st['gate_open'] = gate_open
        st['hold_counter'] = hold_counter
        st['gain_db'] = gain_db

        makeup_linear = 10.0 ** (self.makeup_gain_db / 20.0)
        return (x * gains * makeup_linear).astype(np.float32)
