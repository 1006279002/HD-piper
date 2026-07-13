"""
FIR滤波器设计模块

提供基于频率采样法的FIR滤波器设计，包含：
- 增益限制（最大增益、相邻差值、高频保护）
- PCHIP插值（log频率轴，平滑曲线）
- minimum-phase转换（降低群延迟）
- 宽带增益归一化
"""

import numpy as np
from scipy import signal
from scipy.interpolate import PchipInterpolator


def apply_gain_limits(gains_db, freq_points, max_gain=15.0, max_adjacent_delta=6.0,
                      limit_8000=None, limit_low_freq=None):
    """
    对dB增益值应用限制规则。

    参数：
        gains_db: list/array，各频点的dB增益值
        freq_points: list，频点Hz值（用于识别特定频率）
        max_gain: float，最大允许增益（默认15dB）
        max_adjacent_delta: float，相邻频点最大差值（默认6dB）
        limit_8000: float/None，8kHz频点的上限
        limit_low_freq: float/None，最低频点（250Hz）的上限

    返回：
        np.ndarray，限制后的dB增益值
    """
    gains = np.array(gains_db, dtype=float)

    # 1. 限制最大/最小增益
    gains = np.clip(gains, -20.0, max_gain)

    # 2. 特定频率限制
    for i, f in enumerate(freq_points):
        if f == 8000 and limit_8000 is not None:
            gains[i] = min(gains[i], limit_8000)
        if f == 250 and limit_low_freq is not None:
            gains[i] = min(gains[i], limit_low_freq)

    # 3. 限制相邻频点差值（从低频到高频方向）
    for i in range(1, len(gains)):
        delta = gains[i] - gains[i - 1]
        if abs(delta) > max_adjacent_delta:
            gains[i] = gains[i - 1] + np.sign(delta) * max_adjacent_delta

    return gains


def design_fir_filter(sample_rate, gains_db, freq_points, num_taps=257,
                      strength=1.0, use_minimum_phase=True,
                      max_gain=15.0, max_adjacent_delta=6.0,
                      limit_8000=5.0, limit_low_freq=0.0,
                      normalize_broadband=True):
    """
    基于频率采样法设计FIR滤波器。

    设计流程：
    1. 应用强度系数
    2. 应用增益限制（最大增益、相邻差值、8kHz保护）
    3. PCHIP插值（log频率轴，平滑曲线）
    4. firwin2设计linear-phase FIR
    5. 转换为minimum-phase（降低延迟，可选）
    6. 归一化宽带增益（保持语音频段平均响度，可选）

    参数：
        sample_rate: int，采样率（如44100或48000）
        gains_db: list，各频点的dB增益值
        freq_points: list，频点Hz值
        num_taps: int，FIR阶数（默认257，建议65-2049）
        strength: float，强度系数（0.6/1.0/1.25）
        use_minimum_phase: bool，是否转换为minimum-phase
        max_gain: float，最大允许增益（默认15dB）
        max_adjacent_delta: float，相邻频点最大差值
        limit_8000: float，8kHz上限
        limit_low_freq: float，低频上限
        normalize_broadband: bool，是否归一化宽带增益（默认True，保持
            语音频段平均响度一致，适合实时EQ切换）。听力补偿场景应设为
            False，否则补偿增益会被平均响度归一化抵消。

    返回：
        np.ndarray，FIR滤波器系数（float32）
    """
    # 1. 应用强度
    gains = np.array(gains_db, dtype=float) * strength

    # 2. 应用增益限制
    gains = apply_gain_limits(gains, freq_points, max_gain, max_adjacent_delta,
                               limit_8000, limit_low_freq)

    # 3. PCHIP插值（log频率轴，避免过窄尖峰）
    nyquist = sample_rate // 2
    freq_arr = np.array(freq_points, dtype=float)
    log_freqs = np.log10(freq_arr)
    pchip = PchipInterpolator(log_freqs, gains)

    # 生成密集频率点
    dense_freqs = np.logspace(np.log10(1), np.log10(nyquist), 256)
    dense_freqs[0] = 0.0
    dense_freqs[-1] = nyquist

    # 分段插值：PCHIP只在[250,8000]内有效，外部平推避免外推NaN
    f_low, f_high = freq_arr[0], freq_arr[-1]
    dense_gains_db = np.empty(len(dense_freqs), dtype=float)
    for i, f in enumerate(dense_freqs):
        if f <= 0:
            dense_gains_db[i] = gains[0]
        elif f < f_low:
            dense_gains_db[i] = gains[0]
        elif f > f_high:
            dense_gains_db[i] = gains[-1]
        else:
            dense_gains_db[i] = float(pchip(np.log10(f)))
    dense_gains_db = np.clip(dense_gains_db, -20.0, max_gain)

    # 4. 转换为线性幅度
    gains_linear = 10 ** (dense_gains_db / 20)

    # 5. 设计FIR
    if num_taps % 2 == 0:
        num_taps += 1

    taps = signal.firwin2(num_taps, dense_freqs, gains_linear, fs=sample_rate,
                         window=('kaiser', 8.6))

    # 6. 转换为minimum-phase（降低群延迟）
    if use_minimum_phase:
        linear_taps = taps.copy()
        mp_taps = None
        for method in ('homomorphic', 'hilbert'):
            try:
                # SciPy's historical default is half=True, which creates a
                # half-length filter with approximately sqrt(|H|). In dB this
                # halves the requested EQ curve. For template training targets
                # we need the designed magnitude response, so require half=False.
                candidate = signal.minimum_phase(
                    linear_taps, method=method, half=False
                )
                if candidate is not None and not np.any(np.isnan(candidate)):
                    mp_taps = candidate
                    break
            except TypeError:
                # Older SciPy versions do not expose half=False. Falling back
                # to their default would silently weaken the EQ curve, so keep
                # the linear-phase filter instead.
                continue
            except Exception:
                continue
        if mp_taps is not None:
            taps = mp_taps
        else:
            taps = linear_taps  # 回退到linear-phase

    # 7. 归一化宽带增益（可选，听力补偿场景应关闭）
    if normalize_broadband:
        w, h = signal.freqz(taps, worN=512, fs=sample_rate)
        mask = (w >= 250) & (w <= 8000)
        if np.any(mask):
            avg_gain = np.mean(np.abs(h[mask]))
            if avg_gain > 1e-6:
                taps = taps / avg_gain

    # 最终安全检查
    if np.any(np.isnan(taps)) or np.any(np.isinf(taps)):
        taps = np.zeros(num_taps, dtype=np.float32)
        taps[0] = 1.0

    return taps.astype(np.float32)
