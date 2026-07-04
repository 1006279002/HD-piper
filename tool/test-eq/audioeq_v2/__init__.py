"""
audioeq_v2 - 实时流式音频EQ处理库

核心模块：
- fir_design: FIR滤波器设计（PCHIP插值、增益限制、minimum-phase）
- eq_processor: 流式EQ处理器（含crossfade切换）
- noise_gate: 流式噪声门/压缩器
- presets: 言语增强预设模板

用法示例：
    from audioeq_v2 import StreamingEQWithMasterGain, EQ_TEMPLATES, STRENGTH_LEVELS

    proc = StreamingEQWithMasterGain(
        fs=44100,
        gains_db=EQ_TEMPLATES[0]['gains'],
        num_taps=257,
        strength=1.0,
        master_gain_db=0.0
    )
    output = proc.process(input_data)
"""

from .fir_design import design_fir_filter, apply_gain_limits
from .noise_gate import StreamingNoiseGateCompressor
from .limiter import StreamingLimiter
from .eq_processor import StreamingEQProcessor, StreamingEQWithMasterGain
from .presets import EQ_TEMPLATES, STRENGTH_LEVELS, FREQ_BANDS, BAND_LABELS
from .file_processor import (
    process_audio_array, get_template, list_templates, safe_limiter
)
from .audiogram import (
    parse_audiogram, calculate_eq_gains_from_audiogram,
    interpolate_hearing_loss, build_audiogram_template, SPEECH_WEIGHTS
)

__all__ = [
    'design_fir_filter',
    'apply_gain_limits',
    'StreamingNoiseGateCompressor',
    'StreamingLimiter',
    'StreamingEQProcessor',
    'StreamingEQWithMasterGain',
    'EQ_TEMPLATES',
    'STRENGTH_LEVELS',
    'FREQ_BANDS',
    'BAND_LABELS',
    'process_audio_array',
    'get_template',
    'list_templates',
    'safe_limiter',
    'parse_audiogram',
    'calculate_eq_gains_from_audiogram',
    'interpolate_hearing_loss',
    'build_audiogram_template',
    'SPEECH_WEIGHTS',
]
