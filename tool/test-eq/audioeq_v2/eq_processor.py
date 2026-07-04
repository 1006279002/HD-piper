"""
流式EQ处理器模块

提供实时音频EQ处理功能：
- StreamingEQProcessor: 流式FIR EQ处理器（含crossfade切换）
- StreamingEQWithMasterGain: 组合处理器（EQ + 噪声门 + 高中低频调节 + 主音量）
"""

import numpy as np
from scipy import signal

from .fir_design import design_fir_filter, apply_gain_limits
from .noise_gate import StreamingNoiseGateCompressor
from .limiter import StreamingLimiter
from .presets import FREQ_BANDS


class StreamingEQProcessor:
    """流式FIR EQ处理器 - 支持强度档和crossfade切换"""

    CROSSFADE_MS = 30  # crossfade时长（毫秒）

    def __init__(self, fs, gains_db, num_taps=257, strength=1.0,
                 use_minimum_phase=True, limit_8000=5.0, limit_low_freq=0.0,
                 max_gain=15.0, normalize_broadband=True):
        """
        初始化流式EQ处理器。

        参数：
            fs: int，采样率
            gains_db: list，7个频点的dB增益值
            num_taps: int，FIR阶数（默认257）
            strength: float，强度系数
            use_minimum_phase: bool，是否使用minimum-phase
            limit_8000: float，8kHz上限
            limit_low_freq: float，低频上限
            max_gain: float，单频段最大增益（默认15dB）
            normalize_broadband: bool，是否归一化宽带增益（默认True）。
                听力补偿场景应设为False，避免补偿增益被抵消。
        """
        self.fs = fs
        self.num_taps = num_taps
        self.strength = strength
        self.use_minimum_phase = use_minimum_phase
        self.gains_db = gains_db
        self.limit_8000 = limit_8000
        self.limit_low_freq = limit_low_freq
        self.max_gain = max_gain
        self.normalize_broadband = normalize_broadband

        # 当前滤波器
        self.current_taps = design_fir_filter(
            fs, gains_db, FREQ_BANDS, num_taps, strength, use_minimum_phase,
            max_gain=max_gain, limit_8000=limit_8000, limit_low_freq=limit_low_freq,
            normalize_broadband=normalize_broadband
        )
        self.current_states = {}

        # crossfade状态
        self.old_taps = None
        self.old_states = {}
        self.crossfade_total = 0
        self.crossfade_remaining = 0

        self.reset()

    def reset(self):
        """重置所有状态"""
        self.current_states = {}
        self.old_taps = None
        self.old_states = {}
        self.crossfade_total = 0
        self.crossfade_remaining = 0

    def get_effective_gains(self):
        """获取当前实际生效的增益值（应用强度和限制后）"""
        gains = np.array(self.gains_db, dtype=float) * self.strength
        gains = apply_gain_limits(gains, FREQ_BANDS, limit_8000=self.limit_8000,
                                   limit_low_freq=self.limit_low_freq)
        return gains

    def switch_filter(self, new_gains_db, new_strength=None,
                       limit_8000=None, limit_low_freq=None):
        """
        切换滤波器，启动crossfade消除click/pop。

        参数：
            new_gains_db: list，新增益值
            new_strength: float/None，新强度
            limit_8000: float/None，新8kHz上限
            limit_low_freq: float/None，新低频上限
        """
        if new_strength is None:
            new_strength = self.strength
        if limit_8000 is not None:
            self.limit_8000 = limit_8000
        if limit_low_freq is not None:
            self.limit_low_freq = limit_low_freq

        self.gains_db = new_gains_db
        self.strength = new_strength

        # 保存旧滤波器
        self.old_taps = self.current_taps.copy()
        self.old_states = {k: v.copy() for k, v in self.current_states.items()}

        # 创建新滤波器
        self.current_taps = design_fir_filter(
            self.fs, new_gains_db, FREQ_BANDS, self.num_taps, new_strength,
            self.use_minimum_phase, max_gain=self.max_gain,
            limit_8000=self.limit_8000, limit_low_freq=self.limit_low_freq,
            normalize_broadband=self.normalize_broadband
        )

        # 新滤波器状态从旧滤波器状态继承（近似）
        new_state_len = len(self.current_taps) - 1
        self.current_states = {}
        for k in self.old_states:
            old_len = len(self.old_states[k])
            if old_len >= new_state_len:
                self.current_states[k] = self.old_states[k][-new_state_len:].copy()
            else:
                padded = np.zeros(new_state_len, dtype=np.float32)
                padded[-old_len:] = self.old_states[k]
                self.current_states[k] = padded

        # 启动crossfade
        self.crossfade_total = int(self.fs * self.CROSSFADE_MS / 1000)
        self.crossfade_remaining = self.crossfade_total

    def process(self, x, channel_idx=0):
        """
        处理音频块。

        参数：
            x: np.ndarray，输入音频块
            channel_idx: int，声道索引

        返回：
            np.ndarray，EQ处理后的音频块
        """
        x = np.asarray(x, dtype=np.float32)
        n = len(x)

        # 确保状态存在
        if channel_idx not in self.current_states:
            self.current_states[channel_idx] = np.zeros(len(self.current_taps) - 1, dtype=np.float32)

        # 当前滤波器处理
        zi = self.current_states[channel_idx]
        y_new, zf = signal.lfilter(
            self.current_taps, np.array([1.0], dtype=np.float32), x, zi=zi
        )
        self.current_states[channel_idx] = zf.astype(np.float32)

        # crossfade处理
        if self.crossfade_remaining > 0 and self.old_taps is not None:
            if channel_idx not in self.old_states:
                self.old_states[channel_idx] = np.zeros(len(self.old_taps) - 1, dtype=np.float32)

            zi_old = self.old_states[channel_idx]
            y_old, zf_old = signal.lfilter(
                self.old_taps, np.array([1.0], dtype=np.float32), x, zi=zi_old
            )
            self.old_states[channel_idx] = zf_old.astype(np.float32)

            # 计算crossfade权重
            fade_in_block = min(n, self.crossfade_remaining)
            progress_start = 1.0 - self.crossfade_remaining / self.crossfade_total
            progress_end = 1.0 - (self.crossfade_remaining - fade_in_block) / self.crossfade_total
            alpha = np.linspace(progress_start, progress_end, fade_in_block)

            y = y_new.copy()
            y[:fade_in_block] = y_old[:fade_in_block] * (1 - alpha) + y_new[:fade_in_block] * alpha

            self.crossfade_remaining -= fade_in_block
            if self.crossfade_remaining <= 0:
                self.old_taps = None
                self.old_states = {}
                self.crossfade_total = 0
        else:
            y = y_new

        return y.astype(np.float32)


class StreamingEQWithMasterGain:
    """组合处理器：FIR EQ + 噪声门压缩器 + 高中低频调节 + 主音量控制"""

    def __init__(self, fs, gains_db, num_taps=257, strength=1.0,
                 use_minimum_phase=True, limit_8000=5.0, limit_low_freq=0.0,
                 master_gain_db=0.0,
                 low_cut_db=0.0, mid_gain_db=0.0, high_gain_db=0.0,
                 ng_threshold_db=-50, ng_ratio=4.0, ng_attack_ms=5,
                 ng_release_ms=100, ng_makeup_gain_db=0.0,
                 max_gain=15.0, normalize_broadband=True,
                 limiter_enabled=True, limiter_ceiling_db=-2.0,
                 limiter_attack_ms=2.0, limiter_release_ms=90.0,
                 dc_blocker_hz=25.0, highpass_hz=80.0, highpass_order=2,
                 highpass_enabled=True):
        """
        初始化组合处理器。

        参数：
            fs: int，采样率
            gains_db: list，EQ增益值
            num_taps: int，FIR阶数
            strength: float，强度系数
            use_minimum_phase: bool
            limit_8000: float
            limit_low_freq: float
            master_gain_db: float，主音量增益dB
            low_cut_db: float，低频调节dB（0-300Hz）
            mid_gain_db: float，中频调节dB（300-3000Hz）
            high_gain_db: float，高频调节dB（3000Hz+）
            ng_threshold_db: float，噪声门阈值
            ng_ratio: float，压缩比
            ng_attack_ms: float，攻击时间
            ng_release_ms: float，释放时间
            ng_makeup_gain_db: float，补偿增益
            max_gain: float，单频段最大增益（默认15dB）
            normalize_broadband: bool，是否归一化宽带增益（默认True，
                听力补偿场景应设为False）
        """
        self.eq_proc = StreamingEQProcessor(
            fs, gains_db, num_taps, strength, use_minimum_phase,
            limit_8000, limit_low_freq,
            max_gain=max_gain, normalize_broadband=normalize_broadband
        )

        self.ng_proc = StreamingNoiseGateCompressor(
            sr=fs, threshold_db=ng_threshold_db, ratio=ng_ratio,
            attack_ms=ng_attack_ms, release_ms=ng_release_ms,
            makeup_gain_db=ng_makeup_gain_db
        )

        # 输出限幅器（兜住EQ放大后的瞬态削波）
        self.limiter_enabled = limiter_enabled
        self.limiter = StreamingLimiter(
            fs, ceiling_db=limiter_ceiling_db,
            attack_ms=limiter_attack_ms, release_ms=limiter_release_ms
        )

        self.master_gain = 10 ** (master_gain_db / 20)
        self.master_gain_db = master_gain_db
        self.fs = fs

        # 增益 ramp 目标值（用于平滑跳变，消除 click/pop）
        self.master_gain_target = self.master_gain
        self.low_cut_coeff_target = 10 ** (low_cut_db / 20)
        self.mid_gain_coeff_target = 10 ** (mid_gain_db / 20)
        self.high_gain_coeff_target = 10 ** (high_gain_db / 20)

        # 高中低频调节
        self.low_cut_db = low_cut_db
        self.mid_gain_db = mid_gain_db
        self.high_gain_db = high_gain_db

        self.low_cut_coeff = 10 ** (low_cut_db / 20)
        self.mid_gain_coeff = 10 ** (mid_gain_db / 20)
        self.high_gain_coeff = 10 ** (high_gain_db / 20)

        self.low_filter_b, self.low_filter_a = signal.butter(1, 300, btype='low', fs=fs)
        self.mid_filter_b, self.mid_filter_a = signal.butter(1, [300, 3000], btype='band', fs=fs)
        self.high_filter_b, self.high_filter_a = signal.butter(1, 3000, btype='high', fs=fs)

        self.low_state = np.zeros(len(self.low_filter_b) - 1, dtype=np.float32)
        self.mid_state = np.zeros(len(self.mid_filter_b) - 1, dtype=np.float32)
        self.high_state = np.zeros(len(self.high_filter_b) - 1, dtype=np.float32)

        # ===== 输入端低频清理链：DC blocker + 高通 =====
        # DC blocker：去除直流/超低频（20-30Hz），恒定开启（SOS数值稳定）
        self.dc_blocker_hz = dc_blocker_hz
        self.dc_sos = signal.butter(2, dc_blocker_hz, btype='high', fs=fs, output='sos')
        self.dc_state = {}

        # 语音高通：默认80Hz/2阶，抑制低频哄叫/风噪/轰鸣，可开关+调参
        # 使用SOS形式，4阶高通在float32下也数值稳定（ba直接lfilter会发散）
        self.highpass_enabled = highpass_enabled
        self.highpass_hz = highpass_hz
        self.highpass_order = highpass_order
        self._build_highpass()
        self.hp_state = {}

    def _build_highpass(self):
        """构建/重建高通滤波器（SOS二阶节，数值稳定）"""
        self.hp_sos = signal.butter(
            self.highpass_order, self.highpass_hz, btype='high', fs=self.fs, output='sos'
        )

    def reset(self):
        """重置所有状态"""
        self.eq_proc.reset()
        self.ng_proc.reset()
        self.limiter.reset()
        self.dc_state = {}
        self.hp_state = {}
        self.low_state = np.zeros(len(self.low_filter_b) - 1, dtype=np.float32)
        self.mid_state = np.zeros(len(self.mid_filter_b) - 1, dtype=np.float32)
        self.high_state = np.zeros(len(self.high_filter_b) - 1, dtype=np.float32)

    def update_highpass(self, highpass_hz=None, highpass_order=None, enabled=None):
        """更新高通滤波器参数（截止频率/阶数/开关）"""
        rebuild = False
        if highpass_hz is not None:
            self.highpass_hz = highpass_hz
            rebuild = True
        if highpass_order is not None:
            self.highpass_order = highpass_order
            rebuild = True
        if enabled is not None:
            self.highpass_enabled = bool(enabled)
        if rebuild:
            self._build_highpass()
            self.hp_state = {}

    def switch_eq(self, new_gains_db, new_strength=None, limit_8000=None, limit_low_freq=None):
        """切换EQ模板/强度档（带crossfade）"""
        self.eq_proc.switch_filter(new_gains_db, new_strength, limit_8000, limit_low_freq)

    def update_master_gain(self, gain_db):
        """更新主音量增益（块级ramp平滑，避免click/pop）"""
        self.master_gain_db = gain_db
        self.master_gain_target = 10 ** (gain_db / 20)

    def update_low_cut(self, db):
        """更新低频增益（ramp平滑）"""
        self.low_cut_db = db
        self.low_cut_coeff_target = 10 ** (db / 20)

    def update_mid_gain(self, db):
        """更新中频增益（ramp平滑）"""
        self.mid_gain_db = db
        self.mid_gain_coeff_target = 10 ** (db / 20)

    def update_high_gain(self, db):
        """更新高频增益（ramp平滑）"""
        self.high_gain_db = db
        self.high_gain_coeff_target = 10 ** (db / 20)

    def update_ng_params(self, threshold_db=None, ratio=None, attack_ms=None,
                         release_ms=None, makeup_gain_db=None):
        """更新噪声门参数"""
        self.ng_proc.update_params(threshold_db, ratio, attack_ms, release_ms, makeup_gain_db)

    def set_limiter_enabled(self, enabled):
        """启用/禁用输出限幅器"""
        self.limiter_enabled = bool(enabled)

    def update_limiter_params(self, ceiling_db=None, attack_ms=None, release_ms=None):
        """更新限幅器参数"""
        self.limiter.update_params(ceiling_db, attack_ms, release_ms)

    def process(self, x, channel_idx=0):
        """
        处理音频块。

        处理流程：
        0. DC blocker + 高通（输入端低频清理）
        1. FIR EQ处理
        2. 噪声门处理
        3. 高中低频调节
        4. 主音量控制
        5. 输出限幅

        参数：
            x: np.ndarray，输入音频块
            channel_idx: int，声道索引

        返回：
            np.ndarray，处理后音频块
        """
        x = np.asarray(x, dtype=np.float32)

        # 0. DC blocker（恒定）+ 高通（可开关），输入端清理低频哄叫/直流
        if channel_idx not in self.dc_state:
            self.dc_state[channel_idx] = signal.sosfilt_zi(self.dc_sos).astype(np.float32) * 0.0
        x, self.dc_state[channel_idx] = signal.sosfilt(
            self.dc_sos, x, zi=self.dc_state[channel_idx]
        )
        if self.highpass_enabled:
            if channel_idx not in self.hp_state:
                self.hp_state[channel_idx] = signal.sosfilt_zi(self.hp_sos).astype(np.float32) * 0.0
            elif self.hp_state[channel_idx].shape[0] != self.hp_sos.shape[0]:
                # 阶数变更导致SOS段数变化，重置该声道高通状态
                self.hp_state[channel_idx] = signal.sosfilt_zi(self.hp_sos).astype(np.float32) * 0.0
            x, self.hp_state[channel_idx] = signal.sosfilt(
                self.hp_sos, x, zi=self.hp_state[channel_idx]
            )
        x = x.astype(np.float32)

        # 1. FIR EQ处理
        eq_output = self.eq_proc.process(x, channel_idx)

        # 2. 噪声门处理
        ng_output = self.ng_proc.process(eq_output, channel_idx)

        # 3. 高中低频调节
        low_band, self.low_state = signal.lfilter(
            self.low_filter_b.astype(np.float32),
            self.low_filter_a.astype(np.float32),
            ng_output, zi=self.low_state
        )

        mid_band, self.mid_state = signal.lfilter(
            self.mid_filter_b.astype(np.float32),
            self.mid_filter_a.astype(np.float32),
            ng_output, zi=self.mid_state
        )

        high_band, self.high_state = signal.lfilter(
            self.high_filter_b.astype(np.float32),
            self.high_filter_a.astype(np.float32),
            ng_output, zi=self.high_state
        )

        n = len(x)

        # 生成从当前值到目标值的样本级 ramp（消除跳变 click/pop）
        low_ramp = np.linspace(self.low_cut_coeff, self.low_cut_coeff_target, n, dtype=np.float32)
        mid_ramp = np.linspace(self.mid_gain_coeff, self.mid_gain_coeff_target, n, dtype=np.float32)
        high_ramp = np.linspace(self.high_gain_coeff, self.high_gain_coeff_target, n, dtype=np.float32)
        master_ramp = np.linspace(self.master_gain, self.master_gain_target, n, dtype=np.float32)

        # 应用增益并合并
        output = (low_band * low_ramp +
                  mid_band * mid_ramp +
                  high_band * high_ramp)

        # 主音量（同样 ramp）
        output = output * master_ramp

        # 更新当前值为目标值（ramp已在本块内完成过渡）
        self.low_cut_coeff = self.low_cut_coeff_target
        self.mid_gain_coeff = self.mid_gain_coeff_target
        self.high_gain_coeff = self.high_gain_coeff_target
        self.master_gain = self.master_gain_target

        # 5. 输出限幅（兜住EQ放大后的不可预测瞬态，防止削波）
        if self.limiter_enabled:
            output = self.limiter.process(output, channel_idx)

        return output.astype(np.float32)
