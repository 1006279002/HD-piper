"""
实时流式EQ处理器 - 预设模板版 V2（GUI主程序）

使用 audioeq_v2 模块提供核心处理功能。
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import soundcard as sc
import threading
import time
import statistics

from audioeq_v2 import (
    StreamingEQWithMasterGain,
    EQ_TEMPLATES, STRENGTH_LEVELS,
    FREQ_BANDS, BAND_LABELS,
    parse_audiogram, build_audiogram_template
)
from audioeq_v2.presets import VISIBLE_BAND_INDICES


def safe_limiter(x, peak=0.95):
    """宽带限幅器，防止削波"""
    m = np.max(np.abs(x)) + 1e-9
    if m > peak:
        x = x * (peak / m)
    return x.astype(np.float32)


def apply_gain_limits(gains_db, freq_points, max_gain=15.0, max_adjacent_delta=6.0,
                      limit_8000=None, limit_low_freq=None):
    """应用增益限制规则（用于GUI显示）"""
    from audioeq_v2.fir_design import apply_gain_limits as _apply
    return _apply(gains_db, freq_points, max_gain, max_adjacent_delta,
                  limit_8000, limit_low_freq)


class EQPresetAppV2:
    def __init__(self, root):
        self.root = root
        self.root.title("实时流式EQ处理器 V2 - 言语增强版")
        self.root.geometry("1150x820")
        self.root.resizable(True, True)

        self.is_running = False
        self.processor = None
        self.audio_thread = None
        self.input_device = None
        self.output_device = None
        self.blocksize = 16000
        self.num_taps = 257
        self.use_minimum_phase = True
        self.max_gain = 15.0
        self.normalize_broadband = True
        self.eq_source = 'template'  # 'template' 或 'audiogram'
        self.selected_template = EQ_TEMPLATES[0]
        self.active_template = EQ_TEMPLATES[0]  # 实际用于处理的模板（模板或听力图谱生成）
        self.strength_name = '标准'
        self.strength = STRENGTH_LEVELS[self.strength_name]
        self.master_gain_db = 0.0
        self.low_cut_db = 0.0
        self.mid_gain_db = 0.0
        self.high_gain_db = 0.0

        self.input_devices = sc.all_microphones(include_loopback=True)
        self.output_devices = sc.all_speakers()

        self.process_times = []
        self.rtf_values = []
        self.block_count = 0
        self.start_time = None

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # ===== 左列：设备配置 =====
        device_frame = ttk.LabelFrame(left_frame, text="设备配置（重启生效）", padding="5")
        device_frame.pack(fill=tk.X, pady=5)

        ttk.Label(device_frame, text="输入设备:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.input_combo = ttk.Combobox(device_frame, width=30, state='readonly')
        self.input_combo['values'] = [f"{i}: {dev.name}" for i, dev in enumerate(self.input_devices)]
        self.input_combo.current(3 if len(self.input_devices) > 3 else 0)
        self.input_combo.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(device_frame, text="输出设备:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.output_combo = ttk.Combobox(device_frame, width=30, state='readonly')
        self.output_combo['values'] = [f"{i}: {dev.name}" for i, dev in enumerate(self.output_devices)]
        self.output_combo.current(0)
        self.output_combo.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(device_frame, text="块大小:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.blocksize_var = tk.StringVar(value="16000")
        self.blocksize_entry = ttk.Entry(device_frame, textvariable=self.blocksize_var, width=12)
        self.blocksize_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(device_frame, text="FIR阶数:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.num_taps_var = tk.StringVar(value="257")
        self.num_taps_combo = ttk.Combobox(device_frame, textvariable=self.num_taps_var,
                                           values=["65", "129", "257", "513", "1025", "2049"], width=12)
        self.num_taps_combo.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)

        self.min_phase_var = tk.BooleanVar(value=True)
        self.min_phase_check = ttk.Checkbutton(device_frame, text="使用minimum-phase FIR（降低延迟）",
                                               variable=self.min_phase_var)
        self.min_phase_check.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=2)

        self.normalize_var = tk.BooleanVar(value=True)
        self.normalize_check = ttk.Checkbutton(device_frame, text="宽带增益归一化（听力补偿建议关闭）",
                                               variable=self.normalize_var)
        self.normalize_check.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=2)

        ttk.Label(device_frame, text="最大增益(dB):").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.max_gain_var = tk.StringVar(value="15")
        self.max_gain_entry = ttk.Entry(device_frame, textvariable=self.max_gain_var, width=12)
        self.max_gain_entry.grid(row=6, column=1, sticky=tk.W, padx=5, pady=2)

        self.refresh_btn = ttk.Button(device_frame, text="刷新设备", command=self.refresh_devices, width=10)
        self.refresh_btn.grid(row=7, column=1, sticky=tk.E, pady=5)

        # ===== 左列：EQ来源（模板 / 听力图谱，重启生效） =====
        source_frame = ttk.LabelFrame(left_frame, text="EQ来源（重启生效）", padding="5")
        source_frame.pack(fill=tk.X, pady=5)

        self.eq_source_var = tk.StringVar(value='template')
        ttk.Radiobutton(source_frame, text="预设模板", variable=self.eq_source_var,
                        value='template', command=self.on_eq_source_change).grid(
                        row=0, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(source_frame, text="自定义听力图谱", variable=self.eq_source_var,
                        value='audiogram', command=self.on_eq_source_change).grid(
                        row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(source_frame, text="听力图谱:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.audiogram_var = tk.StringVar(value="250:65,500:70,1000:70,2000:65,4000:75,8000:90")
        self.audiogram_entry = ttk.Entry(source_frame, textvariable=self.audiogram_var, width=32)
        self.audiogram_entry.grid(row=1, column=1, padx=5, pady=2)

        ag_param_frame = ttk.Frame(source_frame)
        ag_param_frame.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Label(ag_param_frame, text="alpha(半增益):").pack(side=tk.LEFT)
        self.alpha_var = tk.StringVar(value="0.5")
        self.alpha_entry = ttk.Entry(ag_param_frame, textvariable=self.alpha_var, width=6)
        self.alpha_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(ag_param_frame, text="（听力图谱模式生效）").pack(side=tk.LEFT)

        # ===== 左列：模板选择 + 强度档 =====
        template_frame = ttk.LabelFrame(left_frame, text="EQ模板 + 强度档（运行时可切换）", padding="5")
        template_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        strength_frame = ttk.Frame(template_frame)
        strength_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(strength_frame, text="强度档:").pack(side=tk.LEFT)
        self.strength_combo = ttk.Combobox(strength_frame, textvariable=tk.StringVar(value=self.strength_name),
                                           values=list(STRENGTH_LEVELS.keys()),
                                           state='readonly', width=8)
        self.strength_combo.pack(side=tk.LEFT, padx=5)
        self.strength_combo.bind('<<ComboboxSelected>>', self.on_strength_change)

        template_list_frame = ttk.Frame(template_frame)
        template_list_frame.pack(fill=tk.BOTH, expand=True)

        self.template_listbox = tk.Listbox(template_list_frame, width=40, height=10)
        self.template_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(template_list_frame, orient=tk.VERTICAL, command=self.template_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.template_listbox.configure(yscrollcommand=scrollbar.set)

        for template in EQ_TEMPLATES:
            self.template_listbox.insert(tk.END, f"[{template['category']}] {template['name']}")

        self.template_listbox.bind('<<ListboxSelect>>', self.on_template_select)
        self.template_listbox.selection_set(0)

        # ===== 左列：控制按钮 =====
        control_frame = ttk.Frame(left_frame, padding="5")
        control_frame.pack(fill=tk.X, pady=5)

        self.start_btn = ttk.Button(control_frame, text="启动", command=self.start_processing, width=12)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(control_frame, text="停止", command=self.stop_processing,
                                   width=12, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = ttk.Button(control_frame, text="重置参数", command=self.reset_params, width=12)
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        # ===== 右列：模板详情 =====
        detail_frame = ttk.LabelFrame(right_frame, text="模板详情", padding="5")
        detail_frame.pack(fill=tk.X, pady=5)

        self.template_name_label = ttk.Label(detail_frame, text="名称: " + self.selected_template['name'],
                                             font=('Arial', 12, 'bold'))
        self.template_name_label.pack(anchor=tk.W, pady=2)

        self.template_desc_label = ttk.Label(detail_frame, text="说明: " + self.selected_template['desc'],
                                             wraplength=400, justify=tk.LEFT)
        self.template_desc_label.pack(anchor=tk.W, pady=2)

        gains_frame = ttk.Frame(detail_frame)
        gains_frame.pack(fill=tk.X, pady=5)
        ttk.Label(gains_frame, text="频段增益（应用强度档后，125Hz为内部低频控制不展示）:").pack(anchor=tk.W)

        gains_display_frame = ttk.Frame(gains_frame)
        gains_display_frame.pack(fill=tk.X)

        # 只展示可见频段（隐藏125Hz内部低频控制点），gain_labels 与 VISIBLE_BAND_INDICES 一一对应
        self.gain_labels = []
        for col, band_idx in enumerate(VISIBLE_BAND_INDICES):
            frame = ttk.Frame(gains_display_frame)
            frame.grid(row=0, column=col, padx=3, pady=2)
            label = ttk.Label(frame, text=BAND_LABELS[band_idx], width=6)
            label.pack()
            gain_label = ttk.Label(frame, text="+0dB", width=6, font=('Arial', 10, 'bold'))
            gain_label.pack()
            self.gain_labels.append(gain_label)

        self.update_gain_display()

        # ===== 右列：主音量控制 =====
        volume_frame = ttk.LabelFrame(right_frame, text="主音量控制（实时调整）", padding="5")
        volume_frame.pack(fill=tk.X, pady=5)

        ttk.Label(volume_frame, text="主增益(dB):").pack(anchor=tk.W)
        self.master_gain_var = tk.DoubleVar(value=0.0)
        self.master_gain_scale = ttk.Scale(volume_frame, from_=-20, to=50,
                                           variable=self.master_gain_var, orient=tk.HORIZONTAL,
                                           length=400, command=self.on_master_gain_change)
        self.master_gain_scale.pack(fill=tk.X, pady=5)

        gain_frame = ttk.Frame(volume_frame)
        gain_frame.pack(fill=tk.X)
        ttk.Label(gain_frame, text="-20dB").pack(side=tk.LEFT)
        self.master_gain_label = ttk.Label(gain_frame, text="0.0 dB", width=12, font=('Arial', 12, 'bold'))
        self.master_gain_label.pack(side=tk.LEFT, padx=20)
        ttk.Label(gain_frame, text="+50dB").pack(side=tk.RIGHT)

        self.gain_warning_label = ttk.Label(volume_frame, text="", foreground='red', font=('Arial', 10))
        self.gain_warning_label.pack(anchor=tk.W)

        # ===== 右列：高中低频调节 =====
        tone_frame = ttk.LabelFrame(right_frame, text="高中低频调节（实时调整）", padding="5")
        tone_frame.pack(fill=tk.X, pady=5)

        low_frame = ttk.Frame(tone_frame)
        low_frame.pack(fill=tk.X, pady=3)
        ttk.Label(low_frame, text="低频(0-300Hz):").pack(side=tk.LEFT)
        self.low_cut_var = tk.DoubleVar(value=0.0)
        self.low_cut_scale = ttk.Scale(low_frame, from_=-50, to=50,
                                       variable=self.low_cut_var, orient=tk.HORIZONTAL,
                                       length=300, command=self.on_low_cut_change)
        self.low_cut_scale.pack(side=tk.LEFT, padx=5)
        self.low_cut_label = ttk.Label(low_frame, text="0.0 dB", width=10, font=('Arial', 10, 'bold'))
        self.low_cut_label.pack(side=tk.LEFT)

        mid_frame = ttk.Frame(tone_frame)
        mid_frame.pack(fill=tk.X, pady=3)
        ttk.Label(mid_frame, text="中频(300-3000Hz):").pack(side=tk.LEFT)
        self.mid_gain_var = tk.DoubleVar(value=0.0)
        self.mid_gain_scale = ttk.Scale(mid_frame, from_=-50, to=50,
                                        variable=self.mid_gain_var, orient=tk.HORIZONTAL,
                                        length=300, command=self.on_mid_gain_change)
        self.mid_gain_scale.pack(side=tk.LEFT, padx=5)
        self.mid_gain_label = ttk.Label(mid_frame, text="0.0 dB", width=10, font=('Arial', 10, 'bold'))
        self.mid_gain_label.pack(side=tk.LEFT)

        high_frame = ttk.Frame(tone_frame)
        high_frame.pack(fill=tk.X, pady=3)
        ttk.Label(high_frame, text="高频(3000Hz+):").pack(side=tk.LEFT)
        self.high_gain_var = tk.DoubleVar(value=0.0)
        self.high_gain_scale = ttk.Scale(high_frame, from_=-50, to=50,
                                         variable=self.high_gain_var, orient=tk.HORIZONTAL,
                                         length=300, command=self.on_high_gain_change)
        self.high_gain_scale.pack(side=tk.LEFT, padx=5)
        self.high_gain_label = ttk.Label(high_frame, text="0.0 dB", width=10, font=('Arial', 10, 'bold'))
        self.high_gain_label.pack(side=tk.LEFT)

        # ===== 右列：噪声门压缩器 =====
        ng_frame = ttk.LabelFrame(right_frame, text="噪声门压缩器（实时调整）", padding="5")
        ng_frame.pack(fill=tk.X, pady=5)

        ng_threshold_frame = ttk.Frame(ng_frame)
        ng_threshold_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ng_threshold_frame, text="阈值(dB):").pack(side=tk.LEFT)
        self.ng_threshold_var = tk.DoubleVar(value=-50.0)
        self.ng_threshold_scale = ttk.Scale(ng_threshold_frame, from_=-80, to=-20,
                                            variable=self.ng_threshold_var, orient=tk.HORIZONTAL,
                                            length=300, command=self.on_ng_threshold_change)
        self.ng_threshold_scale.pack(side=tk.LEFT, padx=5)
        self.ng_threshold_label = ttk.Label(ng_threshold_frame, text="-50.0 dB", width=10, font=('Arial', 10, 'bold'))
        self.ng_threshold_label.pack(side=tk.LEFT)

        ng_ratio_frame = ttk.Frame(ng_frame)
        ng_ratio_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ng_ratio_frame, text="压缩比:").pack(side=tk.LEFT)
        self.ng_ratio_var = tk.DoubleVar(value=4.0)
        self.ng_ratio_scale = ttk.Scale(ng_ratio_frame, from_=1.0, to=10.0,
                                        variable=self.ng_ratio_var, orient=tk.HORIZONTAL,
                                        length=300, command=self.on_ng_ratio_change)
        self.ng_ratio_scale.pack(side=tk.LEFT, padx=5)
        self.ng_ratio_label = ttk.Label(ng_ratio_frame, text="4.0:1", width=10, font=('Arial', 10, 'bold'))
        self.ng_ratio_label.pack(side=tk.LEFT)

        ng_attack_frame = ttk.Frame(ng_frame)
        ng_attack_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ng_attack_frame, text="攻击(ms):").pack(side=tk.LEFT)
        self.ng_attack_var = tk.DoubleVar(value=5.0)
        self.ng_attack_scale = ttk.Scale(ng_attack_frame, from_=1, to=50,
                                         variable=self.ng_attack_var, orient=tk.HORIZONTAL,
                                         length=300, command=self.on_ng_attack_change)
        self.ng_attack_scale.pack(side=tk.LEFT, padx=5)
        self.ng_attack_label = ttk.Label(ng_attack_frame, text="5 ms", width=10, font=('Arial', 10, 'bold'))
        self.ng_attack_label.pack(side=tk.LEFT)

        ng_release_frame = ttk.Frame(ng_frame)
        ng_release_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ng_release_frame, text="释放(ms):").pack(side=tk.LEFT)
        self.ng_release_var = tk.DoubleVar(value=100.0)
        self.ng_release_scale = ttk.Scale(ng_release_frame, from_=10, to=500,
                                          variable=self.ng_release_var, orient=tk.HORIZONTAL,
                                          length=300, command=self.on_ng_release_change)
        self.ng_release_scale.pack(side=tk.LEFT, padx=5)
        self.ng_release_label = ttk.Label(ng_release_frame, text="100 ms", width=10, font=('Arial', 10, 'bold'))
        self.ng_release_label.pack(side=tk.LEFT)

        ng_makeup_frame = ttk.Frame(ng_frame)
        ng_makeup_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ng_makeup_frame, text="补偿(dB):").pack(side=tk.LEFT)
        self.ng_makeup_var = tk.DoubleVar(value=0.0)
        self.ng_makeup_scale = ttk.Scale(ng_makeup_frame, from_=0, to=12,
                                         variable=self.ng_makeup_var, orient=tk.HORIZONTAL,
                                         length=300, command=self.on_ng_makeup_change)
        self.ng_makeup_scale.pack(side=tk.LEFT, padx=5)
        self.ng_makeup_label = ttk.Label(ng_makeup_frame, text="0.0 dB", width=10, font=('Arial', 10, 'bold'))
        self.ng_makeup_label.pack(side=tk.LEFT)

        # ===== 右列：输出限幅器（实时调整） =====
        limiter_frame = ttk.LabelFrame(right_frame, text="输出限幅器（实时调整）", padding="5")
        limiter_frame.pack(fill=tk.X, pady=5)

        lim_switch_frame = ttk.Frame(limiter_frame)
        lim_switch_frame.pack(fill=tk.X, pady=2)
        self.limiter_enabled_var = tk.BooleanVar(value=True)
        self.limiter_check = ttk.Checkbutton(lim_switch_frame, text="启用限幅器（兜住瞬态削波）",
                                             variable=self.limiter_enabled_var,
                                             command=self.on_limiter_toggle)
        self.limiter_check.pack(side=tk.LEFT)

        lim_ceiling_frame = ttk.Frame(limiter_frame)
        lim_ceiling_frame.pack(fill=tk.X, pady=2)
        ttk.Label(lim_ceiling_frame, text="上限(dBFS):").pack(side=tk.LEFT)
        self.limiter_ceiling_var = tk.DoubleVar(value=-2.0)
        self.limiter_ceiling_scale = ttk.Scale(lim_ceiling_frame, from_=-12, to=0,
                                               variable=self.limiter_ceiling_var, orient=tk.HORIZONTAL,
                                               length=300, command=self.on_limiter_ceiling_change)
        self.limiter_ceiling_scale.pack(side=tk.LEFT, padx=5)
        self.limiter_ceiling_label = ttk.Label(lim_ceiling_frame, text="-2.0 dBFS", width=10, font=('Arial', 10, 'bold'))
        self.limiter_ceiling_label.pack(side=tk.LEFT)

        # ===== 右列：低频高通（实时调整） =====
        hp_frame = ttk.LabelFrame(right_frame, text="低频高通（抑制哄叫/风噪/轰鸣，实时调整）", padding="5")
        hp_frame.pack(fill=tk.X, pady=5)

        hp_switch_frame = ttk.Frame(hp_frame)
        hp_switch_frame.pack(fill=tk.X, pady=2)
        self.highpass_enabled_var = tk.BooleanVar(value=True)
        self.highpass_check = ttk.Checkbutton(hp_switch_frame, text="启用高通（默认开启）",
                                              variable=self.highpass_enabled_var,
                                              command=self.on_highpass_toggle)
        self.highpass_check.pack(side=tk.LEFT)
        ttk.Label(hp_switch_frame, text="阶数:").pack(side=tk.LEFT, padx=(15, 2))
        self.highpass_order_var = tk.StringVar(value="2")
        self.highpass_order_combo = ttk.Combobox(hp_switch_frame, textvariable=self.highpass_order_var,
                                                 values=["2", "4"], width=4, state='readonly')
        self.highpass_order_combo.pack(side=tk.LEFT)
        self.highpass_order_combo.bind('<<ComboboxSelected>>', self.on_highpass_change)

        hp_freq_frame = ttk.Frame(hp_frame)
        hp_freq_frame.pack(fill=tk.X, pady=2)
        ttk.Label(hp_freq_frame, text="截止(Hz):").pack(side=tk.LEFT)
        self.highpass_hz_var = tk.DoubleVar(value=80.0)
        self.highpass_hz_scale = ttk.Scale(hp_freq_frame, from_=40, to=150,
                                           variable=self.highpass_hz_var, orient=tk.HORIZONTAL,
                                           length=280, command=self.on_highpass_change)
        self.highpass_hz_scale.pack(side=tk.LEFT, padx=5)
        self.highpass_hz_label = ttk.Label(hp_freq_frame, text="80 Hz", width=8, font=('Arial', 10, 'bold'))
        self.highpass_hz_label.pack(side=tk.LEFT)

        # ===== 右列：运行状态 =====
        status_frame = ttk.LabelFrame(right_frame, text="运行状态", padding="5")
        status_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.status_text = tk.Text(status_frame, height=8, width=50, state=tk.DISABLED)
        self.status_text.pack(fill=tk.BOTH, expand=True)

        self.log_status("就绪。选择EQ模板和强度档后点击\"启动\"开始处理。")
        self.log_status(f"检测到 {len(self.input_devices)} 个输入设备，{len(self.output_devices)} 个输出设备。")
        self.log_status("V2改进：7频点PCHIP插值 + minimum-phase FIR + crossfade切换")
        self.log_status("新增：听力图谱补偿 / 宽带归一化开关 / 最大增益可调")

        # 默认模板模式：禁用听力图谱输入框
        self.audiogram_entry.config(state=tk.DISABLED)
        self.alpha_entry.config(state=tk.DISABLED)

    def update_gain_display(self):
        if self.eq_source == 'audiogram':
            # 听力图谱模式：展示按图谱计算的增益
            try:
                self.max_gain = float(self.max_gain_var.get())
                tpl = self.build_active_template()
                gains = np.array(tpl['gains'], dtype=float)
                self.template_name_label.config(text="名称: 自定义听力图谱")
                self.template_desc_label.config(text="说明: " + tpl['desc'])
            except Exception:
                gains = np.zeros(len(BAND_LABELS))
                self.template_name_label.config(text="名称: 自定义听力图谱")
                self.template_desc_label.config(text="说明: 听力图谱格式有误，请检查输入")
        else:
            self.template_name_label.config(text="名称: " + self.selected_template['name'])
            self.template_desc_label.config(text="说明: " + self.selected_template['desc'])
            gains = np.array(self.selected_template['gains'], dtype=float) * self.strength
            gains = apply_gain_limits(gains, FREQ_BANDS,
                                       limit_8000=self.selected_template['limit_8000'],
                                       limit_low_freq=self.selected_template['limit_low'])
        for col, band_idx in enumerate(VISIBLE_BAND_INDICES):
            self.gain_labels[col].config(text=f"{gains[band_idx]:+.1f}dB")
            color = 'green' if gains[band_idx] > 0 else ('red' if gains[band_idx] < 0 else 'black')
            self.gain_labels[col].config(foreground=color)

    def on_template_select(self, event):
        if self.eq_source == 'audiogram':
            return
        selection = self.template_listbox.curselection()
        if selection:
            index = selection[0]
            self.selected_template = EQ_TEMPLATES[index]
            self.template_name_label.config(text="名称: " + self.selected_template['name'])
            self.template_desc_label.config(text="说明: " + self.selected_template['desc'])
            # 自动同步高通截止频率到模板建议值
            self.highpass_hz_var.set(self.selected_template.get('highpass_hz', 80))
            self.highpass_hz_label.config(text=f"{self.selected_template.get('highpass_hz', 80):.0f} Hz")
            self.update_gain_display()
            if self.is_running and self.processor:
                self.processor.switch_eq(
                    self.selected_template['gains'],
                    new_strength=self.strength,
                    limit_8000=self.selected_template['limit_8000'],
                    limit_low_freq=self.selected_template['limit_low']
                )
                self.log_status(f"切换模板: {self.selected_template['name']}（crossfade中）")

    def on_strength_change(self, event):
        if self.eq_source == 'audiogram':
            return
        self.strength_name = self.strength_combo.get()
        self.strength = STRENGTH_LEVELS[self.strength_name]
        self.update_gain_display()
        if self.is_running and self.processor:
            self.processor.switch_eq(
                self.selected_template['gains'],
                new_strength=self.strength,
                limit_8000=self.selected_template['limit_8000'],
                limit_low_freq=self.selected_template['limit_low']
            )
            self.log_status(f"切换强度档: {self.strength_name}（crossfade中）")

    def on_eq_source_change(self):
        """切换EQ来源（模板/听力图谱），仅更新界面显示，重启生效"""
        self.eq_source = self.eq_source_var.get()
        if self.eq_source == 'audiogram':
            # 听力补偿推荐默认值
            self.normalize_var.set(False)
            self.min_phase_var.set(False)
            self.num_taps_var.set("1025")
            self.template_listbox.config(state=tk.DISABLED)
            self.strength_combo.config(state=tk.DISABLED)
            self.audiogram_entry.config(state=tk.NORMAL)
            self.alpha_entry.config(state=tk.NORMAL)
            self.log_status("EQ来源切换为：自定义听力图谱（已套用补偿推荐参数，重启生效）")
        else:
            self.normalize_var.set(True)
            self.min_phase_var.set(True)
            self.num_taps_var.set("257")
            self.template_listbox.config(state=tk.NORMAL)
            self.strength_combo.config(state='readonly')
            self.audiogram_entry.config(state=tk.DISABLED)
            self.alpha_entry.config(state=tk.DISABLED)
            self.log_status("EQ来源切换为：预设模板（重启生效）")
        self.update_gain_display()

    def build_active_template(self):
        """根据当前EQ来源构建实际用于处理的模板字典"""
        if self.eq_source == 'audiogram':
            ag = parse_audiogram(self.audiogram_var.get())
            alpha = float(self.alpha_var.get())
            return build_audiogram_template(ag, alpha=alpha, max_gain_db=self.max_gain)
        return self.selected_template

    def on_master_gain_change(self, val):
        gain = float(val)
        self.master_gain_label.config(text=f"{gain:.1f} dB")
        if gain > 30:
            self.gain_warning_label.config(text="⚠️ 高增益警告：音量超过30dB可能导致音频失真")
        else:
            self.gain_warning_label.config(text="")
        if self.processor:
            self.processor.update_master_gain(gain)

    def on_low_cut_change(self, val):
        gain = float(val)
        self.low_cut_label.config(text=f"{gain:.1f} dB")
        if self.processor:
            self.processor.update_low_cut(gain)

    def on_mid_gain_change(self, val):
        gain = float(val)
        self.mid_gain_label.config(text=f"{gain:.1f} dB")
        if self.processor:
            self.processor.update_mid_gain(gain)

    def on_high_gain_change(self, val):
        gain = float(val)
        self.high_gain_label.config(text=f"{gain:.1f} dB")
        if self.processor:
            self.processor.update_high_gain(gain)

    def on_ng_threshold_change(self, val):
        threshold = float(val)
        self.ng_threshold_label.config(text=f"{threshold:.1f} dB")
        if self.processor:
            self.processor.update_ng_params(threshold_db=threshold)

    def on_ng_ratio_change(self, val):
        ratio = float(val)
        self.ng_ratio_label.config(text=f"{ratio:.1f}:1")
        if self.processor:
            self.processor.update_ng_params(ratio=ratio)

    def on_ng_attack_change(self, val):
        attack = float(val)
        self.ng_attack_label.config(text=f"{attack:.0f} ms")
        if self.processor:
            self.processor.update_ng_params(attack_ms=attack)

    def on_ng_release_change(self, val):
        release = float(val)
        self.ng_release_label.config(text=f"{release:.0f} ms")
        if self.processor:
            self.processor.update_ng_params(release_ms=release)

    def on_ng_makeup_change(self, val):
        makeup = float(val)
        self.ng_makeup_label.config(text=f"{makeup:.1f} dB")
        if self.processor:
            self.processor.update_ng_params(makeup_gain_db=makeup)

    def on_limiter_toggle(self):
        enabled = self.limiter_enabled_var.get()
        if self.processor:
            self.processor.set_limiter_enabled(enabled)
        self.log_status(f"限幅器已{'启用' if enabled else '禁用'}")

    def on_limiter_ceiling_change(self, val):
        ceiling = float(val)
        self.limiter_ceiling_label.config(text=f"{ceiling:.1f} dBFS")
        if self.processor:
            self.processor.update_limiter_params(ceiling_db=ceiling)

    def on_highpass_toggle(self):
        enabled = self.highpass_enabled_var.get()
        if self.processor:
            self.processor.update_highpass(enabled=enabled)
        self.log_status(f"低频高通已{'启用' if enabled else '禁用'}")

    def on_highpass_change(self, *args):
        hz = float(self.highpass_hz_var.get())
        order = int(self.highpass_order_var.get())
        self.highpass_hz_label.config(text=f"{hz:.0f} Hz")
        if self.processor:
            self.processor.update_highpass(highpass_hz=hz, highpass_order=order)

    def refresh_devices(self):
        self.input_devices = sc.all_microphones(include_loopback=True)
        self.output_devices = sc.all_speakers()
        old_input = self.input_combo.get() if self.input_combo['values'] else ""
        old_output = self.output_combo.get() if self.output_combo['values'] else ""
        self.input_combo['values'] = [f"{i}: {dev.name}" for i, dev in enumerate(self.input_devices)]
        self.output_combo['values'] = [f"{i}: {dev.name}" for i, dev in enumerate(self.output_devices)]
        if old_input in self.input_combo['values']:
            self.input_combo.set(old_input)
        else:
            self.input_combo.current(0)
        if old_output in self.output_combo['values']:
            self.output_combo.set(old_output)
        else:
            self.output_combo.current(0)
        self.log_status(f"设备列表已刷新：{len(self.input_devices)} 个输入设备，{len(self.output_devices)} 个输出设备")

    def log_status(self, msg):
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, msg + "\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)

    def reset_params(self):
        self.master_gain_var.set(0.0)
        self.master_gain_label.config(text="0.0 dB")
        self.gain_warning_label.config(text="")
        self.low_cut_var.set(0.0)
        self.low_cut_label.config(text="0.0 dB")
        self.mid_gain_var.set(0.0)
        self.mid_gain_label.config(text="0.0 dB")
        self.high_gain_var.set(0.0)
        self.high_gain_label.config(text="0.0 dB")
        self.ng_threshold_var.set(-50.0)
        self.ng_threshold_label.config(text="-50.0 dB")
        self.ng_ratio_var.set(4.0)
        self.ng_ratio_label.config(text="4.0:1")
        self.ng_attack_var.set(5.0)
        self.ng_attack_label.config(text="5 ms")
        self.ng_release_var.set(100.0)
        self.ng_release_label.config(text="100 ms")
        self.ng_makeup_var.set(0.0)
        self.ng_makeup_label.config(text="0.0 dB")
        self.limiter_enabled_var.set(True)
        self.limiter_ceiling_var.set(-2.0)
        self.limiter_ceiling_label.config(text="-2.0 dBFS")
        self.highpass_enabled_var.set(True)
        self.highpass_hz_var.set(80.0)
        self.highpass_hz_label.config(text="80 Hz")
        self.highpass_order_var.set("2")
        if self.processor:
            self.processor.update_master_gain(0.0)
            self.processor.update_low_cut(0.0)
            self.processor.update_mid_gain(0.0)
            self.processor.update_high_gain(0.0)
            self.processor.update_ng_params(
                threshold_db=-50.0, ratio=4.0, attack_ms=5.0,
                release_ms=100.0, makeup_gain_db=0.0
            )
            self.processor.set_limiter_enabled(True)
            self.processor.update_limiter_params(ceiling_db=-2.0)
            self.processor.update_highpass(highpass_hz=80.0, highpass_order=2, enabled=True)
        self.log_status("参数已重置为默认值")

    def start_processing(self):
        try:
            input_idx = int(self.input_combo.get().split(":")[0])
            output_idx = int(self.output_combo.get().split(":")[0])
            self.blocksize = int(self.blocksize_var.get())
            self.num_taps = int(self.num_taps_var.get())
            self.use_minimum_phase = self.min_phase_var.get()
            self.normalize_broadband = self.normalize_var.get()
            self.max_gain = float(self.max_gain_var.get())
            self.eq_source = self.eq_source_var.get()
            self.master_gain_db = self.master_gain_var.get()
            self.low_cut_db = self.low_cut_var.get()
            self.mid_gain_db = self.mid_gain_var.get()
            self.high_gain_db = self.high_gain_var.get()
            ng_threshold = self.ng_threshold_var.get()
            ng_ratio = self.ng_ratio_var.get()
            ng_attack = self.ng_attack_var.get()
            ng_release = self.ng_release_var.get()
            ng_makeup = self.ng_makeup_var.get()
            limiter_enabled = self.limiter_enabled_var.get()
            limiter_ceiling = self.limiter_ceiling_var.get()
            hp_enabled = self.highpass_enabled_var.get()
            hp_hz = self.highpass_hz_var.get()
            hp_order = int(self.highpass_order_var.get())

            # 构建实际用于处理的模板（模板 或 听力图谱生成）
            try:
                self.active_template = self.build_active_template()
            except Exception as e:
                messagebox.showerror("错误", f"听力图谱解析失败: {e}")
                return
            # 听力图谱模式强度固定为1.0（增益已由图谱计算）
            eq_strength = 1.0 if self.eq_source == 'audiogram' else self.strength

            self.input_device = self.input_devices[input_idx]
            self.output_device = self.output_devices[output_idx]

            self.log_status(f"输入设备: {self.input_device.name}")
            self.log_status(f"输出设备: {self.output_device.name}")
            self.log_status(f"块大小: {self.blocksize}, FIR阶数: {self.num_taps}, "
                          f"minimum-phase: {self.use_minimum_phase}, "
                          f"归一化: {self.normalize_broadband}, max_gain: {self.max_gain}dB")
            if self.eq_source == 'audiogram':
                self.log_status(f"EQ来源: 自定义听力图谱, alpha={self.alpha_var.get()}")
                self.log_status(f"听力图谱: {self.audiogram_var.get()}")
            else:
                self.log_status(f"EQ来源: 模板 [{self.selected_template['category']}] "
                              f"{self.selected_template['name']}, 强度: {self.strength_name}")

            gains_str = ", ".join(f"{band}: {g:+.1f}dB"
                                 for band, g in zip(BAND_LABELS,
                                                    np.array(self.active_template['gains']) * eq_strength))
            self.log_status(f"生效增益: {gains_str}")

            self.processor = StreamingEQWithMasterGain(
                44100, self.active_template['gains'], self.num_taps,
                strength=eq_strength, use_minimum_phase=self.use_minimum_phase,
                limit_8000=self.active_template.get('limit_8000'),
                limit_low_freq=self.active_template.get('limit_low'),
                master_gain_db=self.master_gain_db,
                low_cut_db=self.low_cut_db, mid_gain_db=self.mid_gain_db,
                high_gain_db=self.high_gain_db,
                ng_threshold_db=ng_threshold, ng_ratio=ng_ratio,
                ng_attack_ms=ng_attack, ng_release_ms=ng_release,
                ng_makeup_gain_db=ng_makeup,
                max_gain=self.max_gain, normalize_broadband=self.normalize_broadband,
                limiter_enabled=limiter_enabled, limiter_ceiling_db=limiter_ceiling,
                highpass_enabled=hp_enabled, highpass_hz=hp_hz, highpass_order=hp_order
            )

            self.process_times = []
            self.rtf_values = []
            self.block_count = 0
            self.start_time = time.time()

            self.is_running = True
            self.audio_thread = threading.Thread(target=self.audio_loop, daemon=True)
            self.audio_thread.start()

            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.disable_device_controls()

            self.log_status("=" * 50)
            self.log_status("处理已启动。运行时可切换模板/强度档（自动crossfade）")

        except Exception as e:
            messagebox.showerror("错误", f"启动失败: {e}")

    def stop_processing(self):
        self.is_running = False
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)
        if self.process_times:
            self.log_status("\n" + "=" * 50)
            self.log_status("处理统计")
            self.log_status("=" * 50)
            total_time = time.time() - self.start_time
            avg_process_time = statistics.mean(self.process_times)
            min_process_time = min(self.process_times)
            max_process_time = max(self.process_times)
            avg_rtf = statistics.mean(self.rtf_values)
            self.log_status(f"总运行时间: {total_time:.2f} 秒")
            self.log_status(f"处理块数: {self.block_count}")
            self.log_status(f"处理时间 - 平均: {avg_process_time:.2f}ms, 最小: {min_process_time:.2f}ms, 最大: {max_process_time:.2f}ms")
            self.log_status(f"RTF - 平均: {avg_rtf:.3f} ({'充足' if avg_rtf < 1.0 else '紧张'})")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.enable_device_controls()
        self.log_status("\n处理已停止")

    def audio_loop(self):
        try:
            with self.input_device.recorder(samplerate=44100, blocksize=self.blocksize) as mic, \
                 self.output_device.player(samplerate=44100, blocksize=self.blocksize) as spk:
                block_duration_ms = self.blocksize / 44100 * 1000
                while self.is_running:
                    audio_data = mic.record(numframes=self.blocksize)
                    channel_data = audio_data[:, 0]
                    process_start = time.time()
                    output = self.processor.process(channel_data, channel_idx=0)
                    output = safe_limiter(output)
                    process_end = time.time()
                    process_time_ms = (process_end - process_start) * 1000
                    rtf = process_time_ms / block_duration_ms
                    if self.block_count > 5:
                        self.process_times.append(process_time_ms)
                        self.rtf_values.append(rtf)
                    self.block_count += 1
                    if self.block_count % 20 == 0:
                        avg_rtf = statistics.mean(self.rtf_values[-10:]) if self.rtf_values else 0
                        avg_time = statistics.mean(self.process_times[-10:]) if self.process_times else 0
                        self.root.after(0, lambda: self.log_status(
                            f"RTF: {avg_rtf:.3f} | 处理: {avg_time:.2f}ms | 块数: {self.block_count}"))
                    output_data = np.zeros((self.blocksize, 2))
                    output_data[:, 0] = output
                    output_data[:, 1] = output
                    spk.play(output_data)
        except Exception as e:
            self.root.after(0, lambda: self.log_status(f"处理错误: {e}"))
            self.root.after(0, self.stop_processing)

    def disable_device_controls(self):
        self.input_combo.config(state=tk.DISABLED)
        self.output_combo.config(state=tk.DISABLED)
        self.blocksize_entry.config(state=tk.DISABLED)
        self.num_taps_combo.config(state=tk.DISABLED)
        self.min_phase_check.config(state=tk.DISABLED)
        self.normalize_check.config(state=tk.DISABLED)
        self.max_gain_entry.config(state=tk.DISABLED)
        self.audiogram_entry.config(state=tk.DISABLED)
        self.alpha_entry.config(state=tk.DISABLED)

    def enable_device_controls(self):
        self.input_combo.config(state='readonly')
        self.output_combo.config(state='readonly')
        self.blocksize_entry.config(state=tk.NORMAL)
        self.num_taps_combo.config(state='readonly')
        self.min_phase_check.config(state=tk.NORMAL)
        self.normalize_check.config(state=tk.NORMAL)
        self.max_gain_entry.config(state=tk.NORMAL)
        # 听力图谱输入框仅在听力图谱模式可用
        if self.eq_source_var.get() == 'audiogram':
            self.audiogram_entry.config(state=tk.NORMAL)
            self.alpha_entry.config(state=tk.NORMAL)


def main():
    root = tk.Tk()
    app = EQPresetAppV2(root)
    root.mainloop()


if __name__ == "__main__":
    main()
