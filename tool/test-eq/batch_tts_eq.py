#!/usr/bin/env python3
"""Batch-generate EQ_1..EQ_6 template folders from an EQ_0 WAV folder.

EQ_0 is clean/original audio. EQ_1..EQ_6 use the six TTS templates in
audioeq_v2.presets.
"""

import argparse
import os
import runpy
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List


PRESETS = runpy.run_path(str(Path(__file__).parent / "audioeq_v2" / "presets.py"))
STRENGTH_LEVELS = PRESETS["STRENGTH_LEVELS"]
TTS_TEMPLATES = PRESETS["EQ_TEMPLATES"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch process EQ_0 WAV files into EQ_1..EQ_6 template folders."
    )
    parser.add_argument(
        "--input-dir",
        "--input_dir",
        type=Path,
        required=True,
        help="Input clean WAV folder, usually EQ_0.",
    )
    parser.add_argument(
        "--output-root",
        "--output_root",
        type=Path,
        required=True,
        help="Output root that will contain EQ_1..EQ_6.",
    )
    parser.add_argument(
        "--template-max-gain",
        "--template_max_gain",
        type=float,
        default=15.0,
        help="Max band gain for EQ_1..EQ_6 template mode (default: 15.0).",
    )
    parser.add_argument(
        "--strength",
        default="标准",
        choices=list(STRENGTH_LEVELS.keys()),
        help="Template strength for EQ_1..EQ_6 (default: 标准).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Parallel worker processes (default: min(8, cpu_count)).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively process WAV files and preserve relative paths.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N WAV files for testing; 0 means all.",
    )
    parser.add_argument(
        "--num-taps-template",
        "--num_taps_template",
        type=int,
        default=257,
        help="FIR taps for EQ_1..EQ_6 templates (default: 257).",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=16000,
        help="Offline processing block size (default: 16000).",
    )
    parser.add_argument(
        "--no-limiter",
        action="store_true",
        help="Disable final limiter.",
    )
    return parser.parse_args()


def collect_wavs(input_dir: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.wav" if recursive else "*.wav"
    return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def process_one(task: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np
    import soundfile as sf
    from audioeq_v2 import process_audio_array

    input_path = Path(task["input_path"])
    output_path = Path(task["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not task["force"]:
        return {
            "status": "skipped",
            "input": str(input_path),
            "output": str(output_path),
        }

    audio, sample_rate = sf.read(input_path)
    audio = np.asarray(audio, dtype=np.float32)

    template = task["template"]
    output = process_audio_array(
        audio,
        sample_rate,
        template,
        strength=task["strength"],
        num_taps=task["num_taps"],
        use_minimum_phase=True,
        blocksize=task["blocksize"],
        apply_limiter=task["apply_limiter"],
        max_gain=task["max_gain"],
        normalize_broadband=False,
    )

    sf.write(output_path, output, sample_rate)
    return {"status": "processed", "input": str(input_path), "output": str(output_path)}


def build_tasks(
    args: argparse.Namespace,
    wavs: List[Path],
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    strength = STRENGTH_LEVELS[args.strength]

    for wav_path in wavs:
        rel_path = wav_path.relative_to(args.input_dir)

        for offset, template in enumerate(TTS_TEMPLATES, start=1):
            tasks.append(
                {
                    "input_path": str(wav_path),
                    "output_path": str(args.output_root / f"EQ_{offset}" / rel_path),
                    "mode": "template",
                    "template": template,
                    "strength": strength,
                    "max_gain": args.template_max_gain,
                    "num_taps": args.num_taps_template,
                    "blocksize": args.blocksize,
                    "apply_limiter": not args.no_limiter,
                    "force": args.force,
                }
            )

    return tasks


def print_counts(done: int, total: int, counts: Dict[str, int]) -> None:
    summary = " ".join(f"{key}={counts[key]}" for key in sorted(counts))
    print(f"[{done}/{total}] {summary}", flush=True)


def main() -> int:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_root = args.output_root.resolve()

    if not args.input_dir.is_dir():
        print(f"[error] input dir does not exist: {args.input_dir}", file=sys.stderr)
        return 1

    wavs = collect_wavs(args.input_dir, args.recursive)
    if args.limit > 0:
        wavs = wavs[: args.limit]
    if not wavs:
        print(f"[error] no wav files found in {args.input_dir}", file=sys.stderr)
        return 1

    args.output_root.mkdir(parents=True, exist_ok=True)
    for idx in range(1, 7):
        (args.output_root / f"EQ_{idx}").mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(args, wavs)
    counts = {"processed": 0, "skipped": 0, "failed": 0}
    print(f"Input EQ_0: {args.input_dir}")
    print(f"Output root: {args.output_root}")
    print(f"WAV files: {len(wavs)}")
    print(f"Tasks: {len(tasks)}")
    print(f"Jobs: {args.jobs}")

    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [executor.submit(process_one, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
                counts[result["status"]] += 1
            except Exception as exc:  # noqa: BLE001 - keep batch running.
                counts["failed"] += 1
                print(f"[failed] {exc}", file=sys.stderr, flush=True)

            if done == len(tasks) or done % 100 == 0:
                print_counts(done, len(tasks), counts)

    return 2 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
