# audioeq_v2 — 实时流式言语增强 EQ 处理库

面向**实时助听 / 通话增强 / 媒体对白**场景的流式音频处理库。核心是一条低延迟处理链，把输入音频经过低频清理、EQ 频谱塑形、温和降噪、动态保护后输出，专为提升言语可懂度设计。

- 8 个内部频率控制点（125 / 250 / 500 / 1k / 2k / 3k / 4k / 8k Hz），其中 125Hz 仅作内部低频控制，前台不展示
- 6 个 TTS 训练预设模板 + 3 档强度
- 支持自定义听力图谱按半增益规则计算个体化增益
- 完整低频清理链（DC blocker + 可调高通）、温和下行扩展器（downward expander）、输出限幅器
- 全链路块间状态连续、参数变更平滑（ramp / crossfade），无 click/pop

---

## 目录

1. [安装依赖](#安装依赖)
2. [快速开始](#快速开始)
3. [处理链路总览](#处理链路总览)
4. [模块划分与职责](#模块划分与职责)
5. [模块详解与主要函数](#模块详解与主要函数)
6. [预设模板与频段](#预设模板与频段)
7. [听力图谱模式](#听力图谱模式)
8. [命令行文件处理](#命令行文件处理)
9. [GUI 主程序](#gui-主程序)
10. [性能与延迟参考](#性能与延迟参考)
11. [设计要点与常见坑](#设计要点与常见坑)

---

## 安装依赖

```bash
pip install numpy scipy soundcard
```

- `numpy` / `scipy`：核心 DSP
- `soundcard`：实时设备采集/播放（仅 GUI 实时模式需要）
- 文件离线处理需要 `soundfile`：`pip install soundfile`

---

## 快速开始

### 作为库使用（最简单）

```python
from audioeq_v2 import StreamingEQWithMasterGain, EQ_TEMPLATES

# 用第 1 个模板（TTS 标准清晰），标准强度构建处理器
proc = StreamingEQWithMasterGain(
    fs=44100,
    gains_db=EQ_TEMPLATES[0]['gains'],   # 8 个频点 dB 增益
    num_taps=257,
    strength=1.0,
    master_gain_db=0.0,
)

# 逐块处理（numpy float32，单声道；多声道用不同 channel_idx）
output = proc.process(input_block, channel_idx=0)
```

### 离线处理整段音频数组

```python
from audioeq_v2 import process_audio_array, get_template
import soundfile as sf

audio, sr = sf.read('in.wav')
tpl = get_template('TTS 标准清晰')       # 按名称或 id 取模板
out = process_audio_array(audio, sr, tpl, strength=1.0, num_taps=513)
sf.write('out.wav', out, sr)
```

---

## 处理链路总览

推荐（也是 `StreamingEQWithMasterGain.process` 的实际顺序）：

```
输入
  → DC blocker（25Hz，恒开，去直流/超低频）
  → 高通 High-pass（默认80Hz/2阶，可开关/调参，抑制哄叫风噪轰鸣）
  → FIR 模板 EQ（8频点 PCHIP 插值 + minimum-phase，支持 crossfade 切换）
  → 温和降噪 Downward Expander（噪声门改进版）
  → 高中低频微调（low/mid/high，块级 ramp）
  → 主音量（块级 ramp）
  → 输出限幅 Limiter（默认 -2 dBFS，attack/release 平滑）
输出
```

关键设计原则：

- **状态跨块连续**：FIR 用 `lfilter` 的 `zi/zf`；高通/DC 用 `sosfilt` 状态；扩展器保存包络；limiter 保存增益。都不在每块重置。
- **参数变更平滑**：增益类走块级 `linspace` ramp；EQ 模板/强度切换走双滤波器输出级 crossfade。
- **低频优先用滤波器而非 EQ**：低频哄叫用稳定的高通/DC blocker 处理，比靠 FIR 模板压低频更直接可靠。

---

## 模块划分与职责

```
audioeq_v2/
├── __init__.py        # 包入口，统一导出公共 API
├── presets.py         # 频段定义、强度档、6 个 TTS 预设模板数据
├── fir_design.py      # FIR 滤波器设计 + 增益限制规则（纯函数）
├── noise_gate.py      # 温和下行扩展器（噪声门改进版）
├── limiter.py         # 输出峰值限幅器
├── audiogram.py       # 听力图谱解析 + 个体化增益计算
├── eq_processor.py    # 流式 EQ 处理器 + 组合处理器（核心编排）
└── file_processor.py  # 离线文件/数组处理 + 模板查询工具
```

| 模块 | 类型 | 职责一句话 |
|---|---|---|
| `presets` | 数据 | 定义频点、强度档、模板增益表与每模板建议高通 |
| `fir_design` | 无状态函数 | 把 dB 增益表转成一组 FIR 系数（含限制/插值/最小相位） |
| `noise_gate` | 有状态类 | 温和抑制低于阈值的背景噪声，不硬性静音 |
| `limiter` | 有状态类 | 兜住 EQ 放大后的瞬态，防削波 |
| `audiogram` | 函数+常量 | 从听力图谱算出个体化 EQ 增益并封装成模板 |
| `eq_processor` | 有状态类 | 串起整条链路，管理所有块间状态与实时参数 |
| `file_processor` | 函数 | 面向整段音频的离线处理封装 |

---

## 模块详解与主要函数

### 1. `presets.py` — 频段与模板数据

**导出常量**

| 名称 | 值 / 说明 |
|---|---|
| `FREQ_BANDS` | `[125, 250, 500, 1000, 2000, 3000, 4000, 8000]`，8 个控制点 |
| `BAND_LABELS` | `["125Hz","250Hz","500Hz","1kHz","2kHz","3kHz","4kHz","8kHz"]` |
| `HIDDEN_BAND_INDICES` | `[0]`，前台隐藏的频段索引（125Hz 仅内部低频控制） |
| `VISIBLE_BAND_INDICES` | 前台展示的频段索引（除 125Hz 外） |
| `STRENGTH_LEVELS` | `{'轻度':0.6, '标准':1.0, '加强':1.25}` |
| `EQ_TEMPLATES` | 6 个 TTS 模板字典列表 |

**模板字典字段**

```python
{
  'id': 1, 'name': 'TTS 标准清晰', 'category': 'TTS模板',
  'shape': '常频缓降型',
  'gains': [-1, 0, 1, 3, 5, 5, 4, 1],   # 对应 FREQ_BANDS 的 8 个 dB 值
  'desc': '……',
  'limit_8000': 1,     # 8kHz 增益上限；当前模板设为 8kHz 目标值
  'limit_low': 0,      # 250Hz 增益上限；当前模板设为 250Hz 目标值
  'highpass_hz': 80,   # 该模板建议的输入端高通截止频率
}
```

### 2. `fir_design.py` — FIR 滤波器设计（无状态）

> **`apply_gain_limits(gains_db, freq_points, max_gain=15.0, max_adjacent_delta=6.0, limit_8000=None, limit_low_freq=None)`**
> 对 dB 增益数组施加安全限制。规则：① clip 到 `[-20, max_gain]`；② 8kHz / 250Hz 分别不超过 `limit_8000` / `limit_low_freq`；③ 相邻频点差值不超过 `max_adjacent_delta`（避免曲线过陡）。返回限制后的 dB 数组。

> **`design_fir_filter(sample_rate, gains_db, freq_points, num_taps=257, strength=1.0, use_minimum_phase=True, max_gain=15.0, max_adjacent_delta=6.0, limit_8000=5.0, limit_low_freq=0.0, normalize_broadband=True)`**
> 把 dB 增益表设计成 FIR 系数（`np.ndarray float32`）。流程：
> 1. `gains × strength`（应用强度档）
> 2. `apply_gain_limits`（安全限制）
> 3. **PCHIP 对数频率插值**：只在 `[freq[0], freq[-1]]` 内插值，低于/高于端点平推，避免外推 NaN；低频负增益因此表现为平滑倾斜而非尖锐 notch
> 4. `firwin2` + Kaiser 窗设计 linear-phase FIR
> 5. 可选转 **minimum-phase**（`homomorphic`→`hilbert` 依次尝试，都失败则回退 linear-phase，降低群延迟）
> 6. 可选**宽带增益归一化**（250–8000Hz 平均增益归一）
> 7. NaN/Inf 安全兜底 → 回退单位脉冲（直通）
>
> **重要**：`normalize_broadband` 用于实时 EQ 保持响度一致；**听力补偿场景应设为 `False`**，否则补偿增益会被平均响度归一化抵消。

### 3. `noise_gate.py` — 温和下行扩展器 `StreamingNoiseGateCompressor`

传统硬噪声门改造为 **downward expander**，专治实时麦克风"沙沙声一阵一阵""尾音发毛"。

> **`__init__(sr, threshold_db=-50, ratio=2.0, attack_ms=15, release_ms=250, makeup_gain_db=0.0, hold_ms=80, max_attenuation_db=15.0, hysteresis_db=3.0, envelope_mode='rms', env_attack_ms=None, env_release_ms=None)`**

| 参数 | 建议 | 作用 |
|---|---|---|
| `threshold_db` | -55~-48 | 扩展阈值 |
| `ratio` | 1.5~2.2 | 扩展比，越大衰减越强（≠传统门的4:1） |
| `attack_ms` | 10~20 | 开门（增益恢复）平滑 |
| `release_ms` | 180~300 | 关门（增益衰减）平滑 |
| `hold_ms` | 60~100 | 保持时间，避免门频繁开关 |
| `max_attenuation_db` | 12~18 | 最大衰减量，背景不会突然消失 |
| `hysteresis_db` | 3 | 迟滞，避免阈值附近抖动 |
| `envelope_mode` | 'rms' | 平滑包络，非逐样本硬判断 |

- **`reset()`**：清空各声道包络/门状态。
- **`update_params(threshold_db=None, ratio=None, attack_ms=None, release_ms=None, makeup_gain_db=None, hold_ms=None, max_attenuation_db=None, hysteresis_db=None)`**：运行时改参数（时间常数类会重算系数）。
- **`process(x, channel_idx=0)`**：RMS 包络跟踪 → 门状态机（迟滞+hold）→ 温和衰减（受 max_attenuation 封顶）→ 增益平滑。块间状态连续。

### 4. `limiter.py` — 输出限幅器 `StreamingLimiter`

> **`__init__(sr, ceiling_db=-2.0, attack_ms=2.0, release_ms=90.0)`**
> 兜住 EQ 放大后不可预测的瞬态削波（不是为了变响）。ceiling 默认 -2 dBFS，attack 1~3ms，release 60~120ms。

- **`reset()`** / **`update_params(ceiling_db=None, attack_ms=None, release_ms=None)`**
- **`process(x, channel_idx=0)`**：峰值超 ceiling 时用平滑增益压回，attack 快介入、release 慢恢复，末尾硬 clip 兜底。块间保持增益状态。

### 5. `audiogram.py` — 听力图谱增益计算

> **常量 `SPEECH_WEIGHTS`** = `[0.3, 0.5, 0.8, 1.0, 1.2, 1.2, 1.2, 0.8]`（对应 8 频点的语音重要性权重，125Hz 最低 0.3）

> **`parse_audiogram(s)`**：解析 `"250:65,500:70,1000:70,2000:65,4000:75,8000:90"` 格式为 `{freq: hl_db}` 字典。

> **`interpolate_hearing_loss(audiogram, freq_points=None)`**：把听力图谱按 **log 频率轴**插值/外推到 8 个目标频点。

> **`calculate_eq_gains_from_audiogram(audiogram, freq_points=None, alpha=0.5, max_gain_db=15.0)`**：半增益规则 + 语音权重 → `增益 = clip(听损 × alpha × 权重, 0, max_gain)`，返回 8 个频点 dB 增益。

> **`build_audiogram_template(audiogram, alpha=0.5, max_gain_db=15.0)`**：把听力图谱封装成与预设模板同结构的字典（`limit_8000`/`limit_low` 设为 None，保留补偿量），可直接喂给处理器。

### 6. `eq_processor.py` — 核心处理器

#### `StreamingEQProcessor` — 纯 FIR EQ（含 crossfade 切换）

> **`__init__(fs, gains_db, num_taps=257, strength=1.0, use_minimum_phase=True, limit_8000=5.0, limit_low_freq=0.0, max_gain=15.0, normalize_broadband=True)`**

- **`process(x, channel_idx=0)`**：`lfilter` 卷积 + 保存 `zf`；若处于 crossfade 期间，同时用旧/新 FIR 各自状态处理并线性混合。
- **`switch_filter(new_gains_db, new_strength=None, limit_8000=None, limit_low_freq=None)`**：重建新 FIR，保留旧 FIR，启动 30ms **输出级 crossfade**（新滤波器状态从旧状态继承，避免归零爆音）。
- **`get_effective_gains()`**：返回应用强度+限制后的实际 dB 增益。
- **`reset()`**：清空滤波器与 crossfade 状态。

#### `StreamingEQWithMasterGain` — 组合处理器（对外主入口）

> **`__init__(fs, gains_db, num_taps=257, strength=1.0, use_minimum_phase=True, limit_8000=5.0, limit_low_freq=0.0, master_gain_db=0.0, low_cut_db=0.0, mid_gain_db=0.0, high_gain_db=0.0, ng_threshold_db=-50, ng_ratio=4.0, ng_attack_ms=5, ng_release_ms=100, ng_makeup_gain_db=0.0, max_gain=15.0, normalize_broadband=True, limiter_enabled=True, limiter_ceiling_db=-2.0, limiter_attack_ms=2.0, limiter_release_ms=90.0, dc_blocker_hz=25.0, highpass_hz=80.0, highpass_order=2, highpass_enabled=True)`**

- **`process(x, channel_idx=0)`**：执行完整链路（DC blocker → 高通 → FIR EQ → 扩展器 → 高中低频 → 主音量 → limiter）。低频/主音量增益走块级 ramp。
- **`reset()`**：重置所有子模块与滤波器状态。

**实时可调方法（无需重启）：**

| 方法 | 说明 |
|---|---|
| `switch_eq(new_gains_db, new_strength=None, limit_8000=None, limit_low_freq=None)` | 切换模板/强度（crossfade） |
| `update_master_gain(gain_db)` | 主音量（ramp 平滑） |
| `update_low_cut(db)` / `update_mid_gain(db)` / `update_high_gain(db)` | 高中低频微调（ramp） |
| `update_ng_params(...)` | 噪声门/扩展器参数 |
| `set_limiter_enabled(enabled)` / `update_limiter_params(ceiling_db, attack_ms, release_ms)` | 限幅器开关与参数 |
| `update_highpass(highpass_hz=None, highpass_order=None, enabled=None)` | 高通截止/阶数/开关（SOS 重建） |

> **数值稳定性**：DC blocker 与高通均使用 **SOS 二阶节 + `sosfilt`**。早期用 `(b,a)+lfilter` 的 4 阶高通在 float32 下会发散产生 NaN，改 SOS 后已解决。

### 7. `file_processor.py` — 离线处理

> **`process_audio_array(audio, sample_rate, template, strength=1.0, num_taps=257, use_minimum_phase=True, master_gain_db=0.0, low_cut_db=0.0, mid_gain_db=0.0, high_gain_db=0.0, ng_threshold_db=-50, ng_ratio=4.0, ng_attack_ms=5, ng_release_ms=100, ng_makeup_gain_db=0.0, blocksize=16000, apply_limiter=True, max_gain=15.0, normalize_broadband=True)`**
> 对整段音频（`(frames,)` 或 `(frames, channels)`）离线处理。每声道独立处理器，分块跑以复现实时行为，末尾统一限幅。返回同形状 float32。

> **`get_template(template_ref)`**：按 `id`(int) 或 `name`/`id`字符串查模板，未找到抛 `ValueError`。
> **`list_templates()`**：返回模板简要信息列表（供 CLI 展示）。
> **`safe_limiter(x, peak=0.95)`**：无状态峰值归一化（简单兜底，区别于 `StreamingLimiter`）。

---

## 预设模板与频段

**频段控制点作用：**

| 控制点 | 作用 | 前台展示 |
|---|---|---|
| 125 Hz | 低频哄叫/轰鸣控制（多为负值平滑倾斜） | 否（内部） |
| 250 Hz | 低频噪声、厚度 | 是 |
| 500 Hz | 元音、人声主体、响度基础 | 是 |
| 1000 Hz | 语音可懂度基础 | 是 |
| 2000 Hz | 言语清晰度核心 | 是 |
| 3000 Hz | 辅音清晰度、语音存在感 | 是 |
| 4000 Hz | 高频辅音、噪声性听损 notch | 是 |
| 8000 Hz | 齿音、细节、刺耳感 | 是 |

**6 个 TTS 预设模板（dB，顺序 125/250/500/1k/2k/3k/4k/8k）：**

| ID | 模板 | 主要覆盖形态 | 125 | 250 | 500 | 1k | 2k | 3k | 4k | 8k | limit_low | limit_8000 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | TTS 标准清晰 | 常频缓降型 | -1 | 0 | +1 | +3 | +5 | +5 | +4 | +1 | 0 | +1 |
| 2 | TTS 陡降补偿 | 常频陡降型 | -1 | 0 | +1 | +4 | +7 | +8 | +7 | +2 | 0 | +2 |
| 3 | TTS 平坦增强 | 常频平坦型 | 0 | +1 | +2 | +4 | +5 | +5 | +4 | +1 | +1 | +1 |
| 4 | TTS 4000 Hz 切迹补偿 | 常频 4000 Hz 切迹型 | -1 | 0 | +1 | +3 | +5 | +7 | +8 | +2 | 0 | +2 |
| 5 | TTS 低频保留 | 低频下降型 | +1 | +2 | +2 | +3 | +4 | +4 | +3 | +1 | +2 | +1 |
| 6 | TTS 中频补偿 | U 型 / 中频凹陷表现 | 0 | +1 | +4 | +6 | +6 | +4 | +3 | 0 | +1 | 0 |

`limit_low` 和 `limit_8000` 是 `fir_design.apply_gain_limits` 里的安全上限，分别作用在 250Hz 与 8kHz。当前 TTS 模板把这两个上限设为模板表中对应频点的目标值，这样在 `--strength 加强` 时不会额外抬高 250Hz 或 8kHz 端点。

**3 档强度**：轻度 ×0.6 / 标准 ×1.0 / 加强 ×1.25（限制最大 +15dB）。

**自定义模板/增益**（提供 8 个频点 dB 值）：

```python
from audioeq_v2 import design_fir_filter, FREQ_BANDS

my_gains = [-3, 0, 2, 4, 6, 8, 6, 3]   # 对应 125/250/500/1k/2k/3k/4k/8k
taps = design_fir_filter(44100, my_gains, FREQ_BANDS, num_taps=257, strength=1.0)
```

---

## 听力图谱模式

不用模板，直接按用户听力图谱计算个体化增益：

```python
from audioeq_v2 import parse_audiogram, build_audiogram_template, process_audio_array
import soundfile as sf

ag = parse_audiogram("250:65,500:70,1000:70,2000:65,4000:75,8000:90")
tpl = build_audiogram_template(ag, alpha=0.5, max_gain_db=15.0)

audio, sr = sf.read('in.wav')
# 听力补偿务必关闭宽带归一化，否则补偿量被抵消；建议高阶 + linear-phase
out = process_audio_array(audio, sr, tpl, num_taps=1025,
                          use_minimum_phase=False, normalize_broadband=False)
sf.write('out.wav', out, sr)
```

- `alpha`：半增益因子（0.3~0.6），越小补偿越温和。听损较重时用小 alpha 避免全部顶到上限。
- `max_gain_db`：单频段上限，听力补偿可放宽到 25~30。

---

## 命令行文件处理

脚本 `audio_file_eq_processor.py`（位于项目根，非包内）：

```bash
# 列出模板
python audio_file_eq_processor.py --list

# 用模板（id 或名称），加强档，主音量 +3dB
python audio_file_eq_processor.py -i in.wav -o out.wav -t "TTS 陡降补偿" --strength 加强 --master-gain 3

# 听力图谱模式（自动套用 1025阶 + linear-phase + 关归一化）
python audio_file_eq_processor.py -i in.wav -o out.wav \
    --audiogram "250:65,500:70,1000:70,2000:65,4000:75,8000:90" --alpha 0.4
```

常用参数：`-t/--template`、`--audiogram`（与 -t 互斥）、`--alpha`、`--max-gain`、`--strength`、`--num-taps`、`--minimum-phase`/`--linear-phase`、`--normalize`/`--no-normalize`、`--master-gain`、`--low-gain`/`--mid-gain`/`--high-gain`、噪声门 `--ng-*`、`--blocksize`、`--no-limiter`。

---

## GUI 主程序

`eq_preset_processor_v2.py`（项目根）提供 Tkinter 实时界面：

```bash
python eq_preset_processor_v2.py
```

功能：设备选择/刷新、模板+强度选择（运行时可切换，自动 crossfade）、听力图谱模式、FIR 阶数/相位/宽带归一化/最大增益、主音量、高中低频微调、噪声门 5 参数、**输出限幅器开关+ceiling**、**低频高通开关+截止+阶数**、运行状态与 RTF 统计。125Hz 频段在增益展示区被隐藏（仅内部生效）。

---

## 性能与延迟参考

| 采样率 | FIR 阶数 | linear-phase 群延迟 | 适用场景 |
|---|---:|---:|---|
| 48kHz | 257 | ~2.7ms | 实时助听/通话 |
| 48kHz | 513 | ~5.3ms | 高精度处理 |
| 48kHz | 1025 | ~10.7ms | 离线/听力补偿 |

minimum-phase 可进一步降低有效延迟，实时场景推荐开启；离线/听力补偿追求相位线性时用 linear-phase。

---

## 设计要点与常见坑

1. **块间状态必须连续**：所有滤波/包络/增益状态仅首次初始化，之后保存续用。切勿在每块 `process` 里把状态归零，否则块边界会有伪影/抽吸声。
2. **参数变更要平滑**：增益走块级 ramp，EQ 切换走输出级 crossfade。直接替换系数或跳变会 click/pop。
3. **低频哄叫优先用高通**：`highpass` / `dc_blocker` 比靠 FIR 模板压低频更稳；4 阶高通务必用 SOS（本库已用 `sosfilt`）。
4. **听力补偿关归一化**：`normalize_broadband=False`，否则补偿增益被平均响度抵消。
5. **不要二次小分块**：callback 给 512/1024 就整块处理，别再切 128/256，减少边界暴露与状态更新开销。
6. **输出务必过 limiter**：EQ 放大瞬态可能短时削波，limiter 兜底（默认 -2 dBFS）。

---

## 文件清单

| 文件 | 说明 |
|---|---|
| `__init__.py` | 包入口，统一导出公共 API |
| `presets.py` | 频段/强度档定义、6 个 TTS 预设模板 |
| `fir_design.py` | `design_fir_filter`、`apply_gain_limits` |
| `noise_gate.py` | `StreamingNoiseGateCompressor`（温和扩展器） |
| `limiter.py` | `StreamingLimiter`（输出限幅器） |
| `audiogram.py` | 听力图谱解析与增益计算 |
| `eq_processor.py` | `StreamingEQProcessor`、`StreamingEQWithMasterGain` |
| `file_processor.py` | `process_audio_array`、`get_template`、`list_templates` |
| `README.md` | 本设计文档 |
