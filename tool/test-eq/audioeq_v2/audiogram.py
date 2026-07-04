"""
听力图谱解析与增益计算模块

支持从用户听力图谱（6频点）计算个体化EQ增益，映射到新的8频点
控制轴（125/250/500/1000/2000/3000/4000/8000 Hz）。

计算方法参考 step_spectral_eq.py 的半增益规则 + 语音重要性加权，
但频率点和权重按新EQ频率轴调整：
    频点:   125   250   500   1k    2k    3k    4k    8k
    权重:   0.3   0.5   0.8   1.0   1.2   1.2   1.2   0.8

125Hz 权重最低（0.3），言语增强通常不需要补偿低频哄叫段。

此模块为新增内容，不修改现有模块，不影响 eq_preset_processor_v2.py。
"""

import numpy as np

from .presets import FREQ_BANDS

# 8频点对应的语音重要性权重（125Hz低频权重最低0.3，3k按1.2处理）
SPEECH_WEIGHTS = np.array([0.3, 0.5, 0.8, 1.0, 1.2, 1.2, 1.2, 0.8])



def parse_audiogram(s):
    """
    解析听力图谱字符串。

    格式："250:65,500:70,1000:70,2000:65,4000:75,8000:90"

    参数：
        s: str，听力图谱字符串（频率:听力损失dB，逗号分隔）

    返回：
        dict，{频率(float): 听力损失dB(float)}

    异常：
        ValueError，无法解析时抛出
    """
    try:
        result = {}
        for item in s.split(","):
            item = item.strip()
            if ':' in item:
                f, hl = item.split(":")
                result[float(f.strip())] = float(hl.strip())
        if not result:
            raise ValueError("空的听力图谱")
        return result
    except ValueError:
        raise
    except Exception:
        raise ValueError(f"无法解析听力图谱输入: {s}")


def interpolate_hearing_loss(audiogram, freq_points=None):
    """
    将听力图谱插值/外推到目标频率点（log频率轴）。

    参数：
        audiogram: dict，{频率: 听力损失dB}
        freq_points: list，目标频率点（默认使用FREQ_BANDS的7频点）

    返回：
        np.ndarray，各目标频点的听力损失dB
    """
    if freq_points is None:
        freq_points = FREQ_BANDS

    keys = sorted(audiogram.keys())
    hl_values = np.array([audiogram[k] for k in keys])
    log_keys = np.log10(np.array(keys))

    hearing_loss = []
    for freq in freq_points:
        if freq in audiogram:
            hearing_loss.append(audiogram[freq])
        elif freq < keys[0]:
            hearing_loss.append(audiogram[keys[0]])
        elif freq > keys[-1]:
            hearing_loss.append(audiogram[keys[-1]])
        else:
            interpolated = float(np.interp(np.log10(freq), log_keys, hl_values))
            hearing_loss.append(interpolated)

    return np.array(hearing_loss, dtype=float)


def calculate_eq_gains_from_audiogram(audiogram, freq_points=None,
                                       alpha=0.5, max_gain_db=15.0):
    """
    根据听力图谱计算个体化EQ增益（半增益规则 + 语音重要性加权）。

    参数：
        audiogram: dict，{频率: 听力损失dB}
        freq_points: list，目标频率点（默认FREQ_BANDS的7频点）
        alpha: float，半增益因子（默认0.5，建议0.3~0.6）
        max_gain_db: float，最大增益限制（默认15dB，与EQ模块一致）

    返回：
        np.ndarray，7个频点的EQ增益（dB）
    """
    if freq_points is None:
        freq_points = FREQ_BANDS

    hearing_loss_db = interpolate_hearing_loss(audiogram, freq_points)

    # 权重：仅当频率点与新7频点一致时使用语音权重，否则用全1
    if len(freq_points) == len(FREQ_BANDS) and \
            np.allclose(np.array(freq_points, dtype=float), np.array(FREQ_BANDS, dtype=float)):
        weights = SPEECH_WEIGHTS
    else:
        weights = np.ones(len(freq_points))

    base_gains = alpha * hearing_loss_db
    weighted_gains = base_gains * weights
    clipped_gains = np.clip(weighted_gains, 0, max_gain_db)

    return clipped_gains


def build_audiogram_template(audiogram, alpha=0.5, max_gain_db=15.0):
    """
    从听力图谱构建可供 process_audio_array 使用的模板字典。

    与预设模板结构兼容（含gains/limit_8000/limit_low），但不额外限制
    8kHz和低频（设为None），以保留听力补偿所需的增益。

    参数：
        audiogram: dict，听力图谱
        alpha: float，半增益因子
        max_gain_db: float，最大增益限制

    返回：
        dict，模板字典
    """
    gains = calculate_eq_gains_from_audiogram(audiogram, FREQ_BANDS, alpha, max_gain_db)
    return {
        'id': 0,
        'name': '自定义听力图谱',
        'category': '听力补偿',
        'gains': [round(float(g), 1) for g in gains],
        'desc': '根据听力图谱按半增益规则+语音权重计算的个体化增益',
        'limit_8000': None,
        'limit_low': None,
    }
