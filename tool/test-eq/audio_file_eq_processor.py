"""
音频文件EQ处理器（命令行版，无UI）

使用 audioeq_v2 模块对音频文件进行离线EQ处理，处理流程与实时版本一致：
    FIR EQ -> 噪声门 -> 高中低频调节 -> 主音量 -> 限幅

用法示例：
    # 列出所有模板
    python audio_file_eq_processor.py --list

    # 用模板id=4（会议发言）处理，标准强度，minimum-phase
    python audio_file_eq_processor.py -i input.wav -o output.wav -t 4

    # 用模板名称，加强档，linear-phase，主音量+6dB
    python audio_file_eq_processor.py -i in.wav -o out.wav -t 会议发言 \\
        --strength 加强 --linear-phase --master-gain 6

    # 自定义FIR阶数、块大小、噪声门参数
    python audio_file_eq_processor.py -i in.wav -o out.wav -t 2 \\
        --num-taps 513 --blocksize 32000 --ng-threshold -45 --ng-ratio 5
"""

import argparse
import sys
import time

import numpy as np
import soundfile as sf

from audioeq_v2 import (
    process_audio_array,
    get_template,
    list_templates,
    STRENGTH_LEVELS,
    parse_audiogram,
    build_audiogram_template,
    FREQ_BANDS,
    BAND_LABELS,
)


def print_templates():
    print("可用EQ模板：")
    print("-" * 70)
    for tpl in list_templates():
        gains_str = " ".join(f"{g:+d}" for g in tpl["gains"])
        print(f"  [{tpl['id']:>2}] {tpl['name']:<8} ({tpl['category']})")
        print(f"       频段增益(125/250/500/1k/2k/3k/4k/8k): {gains_str}")
        print(f"       {tpl['desc']}")
    print("-" * 70)
    print(f"强度档: {', '.join(STRENGTH_LEVELS.keys())}")


def main():
    parser = argparse.ArgumentParser(
        description="音频文件EQ处理器（基于 audioeq_v2 模块）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-i", "--input", help="输入音频文件路径")
    parser.add_argument("-o", "--output", help="输出音频文件路径")
    parser.add_argument(
        "-t",
        "--template",
        default=None,
        help="EQ模板id或名称（默认1，标准增强；与--audiogram互斥）",
    )
    parser.add_argument(
        "--audiogram",
        default=None,
        help='自定义听力图谱，格式如 "250:65,500:70,1000:70,2000:65,4000:75,8000:90"，'
        "按半增益规则+语音权重计算增益（与-t互斥）",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="半增益因子（听力图谱模式，默认0.5，建议0.3~0.6）",
    )
    parser.add_argument(
        "--max-gain",
        type=float,
        default=15.0,
        help="单频段最大增益限制dB（默认15；听力补偿可提高到25~30）",
    )
    parser.add_argument(
        "--strength",
        default="标准",
        choices=list(STRENGTH_LEVELS.keys()),
        help="强度档（默认标准）",
    )
    parser.add_argument("--list", action="store_true", help="列出所有模板后退出")

    # FIR参数
    parser.add_argument(
        "--num-taps",
        type=int,
        default=None,
        help="FIR阶数（模板模式默认257；听力图谱模式默认1025）",
    )
    phase_group = parser.add_mutually_exclusive_group()
    phase_group.add_argument(
        "--minimum-phase",
        dest="minimum_phase",
        action="store_true",
        default=None,
        help="使用minimum-phase FIR（低延迟；模板模式默认）",
    )
    phase_group.add_argument(
        "--linear-phase",
        dest="minimum_phase",
        action="store_false",
        help="使用linear-phase FIR（听力图谱模式默认）",
    )
    parser.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        default=None,
        help="启用宽带增益归一化（模板模式默认开启）",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="关闭宽带增益归一化（听力补偿模式默认关闭，保留补偿量）",
    )

    # 音量与音色
    parser.add_argument(
        "--master-gain", type=float, default=0.0, help="主音量增益dB（默认0）"
    )
    parser.add_argument(
        "--low-gain", type=float, default=0.0, help="低频调节dB，0-300Hz（默认0）"
    )
    parser.add_argument(
        "--mid-gain", type=float, default=0.0, help="中频调节dB，300-3000Hz（默认0）"
    )
    parser.add_argument(
        "--high-gain", type=float, default=0.0, help="高频调节dB，3000Hz+（默认0）"
    )

    # 噪声门参数
    parser.add_argument(
        "--ng-threshold", type=float, default=-50.0, help="噪声门阈值dB（默认-50）"
    )
    parser.add_argument("--ng-ratio", type=float, default=4.0, help="压缩比（默认4.0）")
    parser.add_argument(
        "--ng-attack", type=float, default=5.0, help="攻击时间ms（默认5）"
    )
    parser.add_argument(
        "--ng-release", type=float, default=100.0, help="释放时间ms（默认100）"
    )
    parser.add_argument(
        "--ng-makeup", type=float, default=0.0, help="补偿增益dB（默认0）"
    )

    # 处理选项
    parser.add_argument(
        "--blocksize", type=int, default=16000, help="分块处理大小（默认16000）"
    )
    parser.add_argument("--no-limiter", action="store_true", help="禁用输出限幅")

    args = parser.parse_args()

    if args.list:
        print_templates()
        return

    if not args.input or not args.output:
        parser.error("需要指定 -i/--input 和 -o/--output（或使用 --list 查看模板）")

    # 校验模板与听力图谱互斥
    if args.template is not None and args.audiogram is not None:
        parser.error("-t/--template 与 --audiogram 不能同时使用")

    # 确定EQ来源：听力图谱 或 预设模板
    audiogram_mode = args.audiogram is not None
    if audiogram_mode:
        try:
            ag = parse_audiogram(args.audiogram)
            template = build_audiogram_template(
                ag, alpha=args.alpha, max_gain_db=args.max_gain
            )
        except ValueError as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # 未指定则默认模板1
        template_ref = args.template if args.template is not None else "1"
        try:
            template = get_template(template_ref)
        except ValueError as e:
            print(f"错误: {e}", file=sys.stderr)
            print("使用 --list 查看可用模板", file=sys.stderr)
            sys.exit(1)

    strength = STRENGTH_LEVELS[args.strength]

    # 按模式解析未显式指定的默认值
    if args.num_taps is not None:
        num_taps = args.num_taps
    else:
        num_taps = 1025 if audiogram_mode else 257

    if args.minimum_phase is not None:
        use_minimum_phase = args.minimum_phase
    else:
        # 听力图谱模式默认linear-phase（形状更准），模板模式默认minimum-phase（低延迟）
        use_minimum_phase = not audiogram_mode

    if args.normalize is not None:
        normalize_broadband = args.normalize
    else:
        # 听力补偿模式默认关闭归一化（保留补偿量），模板模式默认开启
        normalize_broadband = not audiogram_mode

    # 读取音频
    try:
        audio, sample_rate = sf.read(args.input)
    except Exception as e:
        print(f"读取音频失败: {e}", file=sys.stderr)
        sys.exit(1)

    audio = np.asarray(audio, dtype=np.float32)
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    duration = len(audio) / sample_rate

    phase_mode = "minimum-phase" if use_minimum_phase else "linear-phase"
    print(f"输入文件: {args.input}")
    print(f"  采样率: {sample_rate} Hz, 声道: {channels}, 时长: {duration:.2f} 秒")
    if audiogram_mode:
        print(
            f"EQ来源: 自定义听力图谱 (alpha={args.alpha}, max_gain={args.max_gain}dB)"
        )
        print(f"  听力图谱: {args.audiogram}")
        gains_str = ", ".join(
            f"{b}:{g:+.1f}" for b, g in zip(BAND_LABELS, template["gains"])
        )
        print(f"  计算增益(125/250/500/1k/2k/3k/4k/8k): {gains_str}")
    else:
        print(f"EQ模板: [{template['id']}] {template['name']} ({template['category']})")
    print(f"强度档: {args.strength} (×{strength})")
    print(
        f"FIR: {num_taps} taps, {phase_mode}, "
        f"宽带归一化: {'开' if normalize_broadband else '关'}, max_gain: {args.max_gain}dB"
    )
    print(
        f"主音量: {args.master_gain:+.1f}dB | "
        f"高中低频: 低{args.low_gain:+.1f} 中{args.mid_gain:+.1f} 高{args.high_gain:+.1f}dB"
    )
    print(
        f"噪声门: 阈值{args.ng_threshold:.1f}dB, 压缩比{args.ng_ratio:.1f}:1, "
        f"攻击{args.ng_attack:.0f}ms, 释放{args.ng_release:.0f}ms, 补偿{args.ng_makeup:.1f}dB"
    )
    print("处理中...")

    t0 = time.time()
    output = process_audio_array(
        audio,
        sample_rate,
        template,
        strength=strength,
        num_taps=num_taps,
        use_minimum_phase=use_minimum_phase,
        master_gain_db=args.master_gain,
        low_cut_db=args.low_gain,
        mid_gain_db=args.mid_gain,
        high_gain_db=args.high_gain,
        ng_threshold_db=args.ng_threshold,
        ng_ratio=args.ng_ratio,
        ng_attack_ms=args.ng_attack,
        ng_release_ms=args.ng_release,
        ng_makeup_gain_db=args.ng_makeup,
        blocksize=args.blocksize,
        apply_limiter=not args.no_limiter,
        max_gain=args.max_gain,
        normalize_broadband=normalize_broadband,
    )
    elapsed = time.time() - t0

    # 写出音频
    try:
        sf.write(args.output, output, sample_rate)
    except Exception as e:
        print(f"写入音频失败: {e}", file=sys.stderr)
        sys.exit(1)

    rtf = elapsed / duration if duration > 0 else 0
    print(f"完成！耗时 {elapsed:.2f} 秒 (RTF={rtf:.3f})")
    print(f"输出文件: {args.output}")


if __name__ == "__main__":
    main()
