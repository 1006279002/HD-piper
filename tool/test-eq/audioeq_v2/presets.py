"""
预设模板数据模块

定义言语增强的EQ模板、频段控制点和强度档位。

频段设计（8个控制点）：
- 125Hz: 低频哄叫/轰鸣控制点（仅内部实时低频控制，前台不展示）
- 250Hz: 低频噪声、厚度控制
- 500Hz: 元音、人声主体、响度基础
- 1000Hz: 语音可懂度基础
- 2000Hz: 言语清晰度核心
- 3000Hz: 辅音清晰度、语音存在感
- 4000Hz: 高频辅音、噪声性听损notch
- 8000Hz: 齿音、细节、刺耳感

强度档位：
- 轻度: 增益 × 0.6
- 标准: 增益 × 1.0
- 加强: 增益 × 1.25（限制最大+15dB）
"""

# 频段定义（8个控制点，含125Hz低频控制点）
FREQ_BANDS = [125, 250, 500, 1000, 2000, 3000, 4000, 8000]
BAND_LABELS = ["125Hz", "250Hz", "500Hz", "1kHz", "2kHz", "3kHz", "4kHz", "8kHz"]

# 前台隐藏的频段索引（125Hz仅作内部实时低频控制，不在前台展示）
HIDDEN_BAND_INDICES = [0]
# 前台可见的频段索引
VISIBLE_BAND_INDICES = [
    i for i in range(len(FREQ_BANDS)) if i not in HIDDEN_BAND_INDICES
]

# 强度档位
STRENGTH_LEVELS = {"轻度": 0.6, "标准": 1.0, "加强": 1.25}

# TTS训练用预设模板（6个）
# gains 顺序对应 FREQ_BANDS: [125, 250, 500, 1k, 2k, 3k, 4k, 8k]
# highpass_hz: 建议的输入端高通截止频率（Hz）
EQ_TEMPLATES = [
    {
        "id": 1,
        "name": "TTS 标准清晰",
        "category": "TTS模板",
        "shape": "常频缓降型",
        "gains": [-1, 0, 1, 3, 5, 5, 4, 1],
        "desc": "常频缓降型，作为温和清晰度增强模板",
        "limit_8000": 1,
        "limit_low": 0,
        "highpass_hz": 80,
    },
    {
        "id": 2,
        "name": "TTS 陡降补偿",
        "category": "TTS模板",
        "shape": "常频陡降型",
        "gains": [-1, 0, 1, 4, 7, 8, 7, 2],
        "desc": "常频陡降型，增强2k-4kHz辅音清晰度",
        "limit_8000": 2,
        "limit_low": 0,
        "highpass_hz": 80,
    },
    {
        "id": 3,
        "name": "TTS 平坦增强",
        "category": "TTS模板",
        "shape": "常频平坦型",
        "gains": [0, 1, 2, 4, 5, 5, 4, 1],
        "desc": "常频平坦型，整体轻中度增强",
        "limit_8000": 1,
        "limit_low": 1,
        "highpass_hz": 80,
    },
    {
        "id": 4,
        "name": "TTS 4000 Hz 切迹补偿",
        "category": "TTS模板",
        "shape": "常频4000 Hz切迹型",
        "gains": [-1, 0, 1, 3, 5, 7, 8, 2],
        "desc": "常频4000 Hz切迹型，突出4kHz附近补偿",
        "limit_8000": 2,
        "limit_low": 0,
        "highpass_hz": 80,
    },
    {
        "id": 5,
        "name": "TTS 低频保留",
        "category": "TTS模板",
        "shape": "低频下降型",
        "gains": [1, 2, 2, 3, 4, 4, 3, 1],
        "desc": "低频保留，整体增强更平缓",
        "limit_8000": 1,
        "limit_low": 2,
        "highpass_hz": 80,
    },
    {
        "id": 6,
        "name": "TTS 中频补偿",
        "category": "TTS模板",
        "shape": "U型/中频凹陷表现",
        "gains": [0, 1, 4, 6, 6, 4, 3, 0],
        "desc": "U型/中频凹陷表现，突出500Hz-2kHz补偿",
        "limit_8000": 0,
        "limit_low": 1,
        "highpass_hz": 80,
    },
]
