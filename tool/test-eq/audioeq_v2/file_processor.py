"""
音频文件离线处理模块

复用流式EQ处理器对整个音频文件进行离线处理，行为与实时版本一致
（分块处理 + 块间状态保持），支持多声道、任意采样率。

此模块为新增内容，不修改现有模块，不影响 eq_preset_processor_v2.py。
"""

import numpy as np

from .eq_processor import StreamingEQWithMasterGain
from .presets import EQ_TEMPLATES, STRENGTH_LEVELS, FREQ_BANDS


def safe_limiter(x, peak=0.95):
    """宽带限幅器，防止削波"""
    m = np.max(np.abs(x)) + 1e-9
    if m > peak:
        x = x * (peak / m)
    return x.astype(np.float32)


def get_template(template_ref):
    """
    按 id（int）或名称（str）查找模板。

    参数：
        template_ref: int/str，模板id或名称

    返回：
        dict，模板字典

    异常：
        ValueError，未找到时抛出
    """
    for tpl in EQ_TEMPLATES:
        if isinstance(template_ref, int) and tpl['id'] == template_ref:
            return tpl
        if isinstance(template_ref, str):
            if tpl['name'] == template_ref or str(tpl['id']) == template_ref:
                return tpl
    raise ValueError(f"未找到模板: {template_ref}")


def list_templates():
    """返回所有模板的简要信息列表（用于命令行展示）"""
    return [
        {
            'id': tpl['id'],
            'name': tpl['name'],
            'category': tpl['category'],
            'gains': tpl['gains'],
            'desc': tpl['desc'],
        }
        for tpl in EQ_TEMPLATES
    ]


def process_audio_array(audio, sample_rate, template, strength=1.0,
                        num_taps=257, use_minimum_phase=True,
                        master_gain_db=0.0,
                        low_cut_db=0.0, mid_gain_db=0.0, high_gain_db=0.0,
                        ng_threshold_db=-50, ng_ratio=4.0, ng_attack_ms=5,
                        ng_release_ms=100, ng_makeup_gain_db=0.0,
                        blocksize=16000, apply_limiter=True,
                        max_gain=15.0, normalize_broadband=True):
    """
    对音频数组进行离线EQ处理。

    参数：
        audio: np.ndarray，形状 (frames,) 或 (frames, channels)
        sample_rate: int，采样率
        template: dict，EQ模板（含gains/limit_8000/limit_low）
        strength: float，强度系数
        num_taps: int，FIR阶数
        use_minimum_phase: bool，是否使用minimum-phase FIR
        master_gain_db: float，主音量增益
        low_cut_db/mid_gain_db/high_gain_db: float，高中低频调节
        ng_*: 噪声门参数
        blocksize: int，分块处理大小
        apply_limiter: bool，是否应用输出限幅
        max_gain: float，单频段最大增益（默认15dB）
        normalize_broadband: bool，是否归一化宽带增益（默认True，
            听力补偿场景应设为False）

    返回：
        np.ndarray，处理后的音频（与输入形状一致，float32）
    """
    audio = np.asarray(audio, dtype=np.float32)

    # 统一为 (frames, channels)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
        squeeze_output = True
    else:
        squeeze_output = False

    frames, channels = audio.shape

    # 每个声道独立处理器（保证声道间状态互不干扰）
    processors = []
    for _ in range(channels):
        proc = StreamingEQWithMasterGain(
            sample_rate, template['gains'], num_taps,
            strength=strength, use_minimum_phase=use_minimum_phase,
            limit_8000=template.get('limit_8000', 5.0),
            limit_low_freq=template.get('limit_low', 0.0),
            master_gain_db=master_gain_db,
            low_cut_db=low_cut_db, mid_gain_db=mid_gain_db, high_gain_db=high_gain_db,
            ng_threshold_db=ng_threshold_db, ng_ratio=ng_ratio,
            ng_attack_ms=ng_attack_ms, ng_release_ms=ng_release_ms,
            ng_makeup_gain_db=ng_makeup_gain_db,
            max_gain=max_gain, normalize_broadband=normalize_broadband
        )
        processors.append(proc)

    output = np.zeros_like(audio)

    # 分块处理
    for start in range(0, frames, blocksize):
        end = min(start + blocksize, frames)
        for ch in range(channels):
            block = audio[start:end, ch]
            out_block = processors[ch].process(block, channel_idx=0)
            output[start:end, ch] = out_block

    # 输出限幅（对整段统一处理，避免逐块限幅导致音量跳变）
    if apply_limiter:
        output = safe_limiter(output)

    if squeeze_output:
        output = output[:, 0]

    return output.astype(np.float32)
