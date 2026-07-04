#!/usr/bin/env python3

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

# Allow running this script directly (python export_onnx.py) as well as
# via the package (python -m piper.train.export_onnx).
if __package__ is None:
    _src_dir = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(_src_dir))
    from piper.train.vits.lightning import VitsModel
else:
    from .vits.lightning import VitsModel

_LOGGER = logging.getLogger(__name__)
OPSET_VERSION = 18


def main() -> None:
    """Main entry point"""
    torch.manual_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--output-file", required=True, help="Path to output file (.onnx)"
    )

    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # -------------------------------------------------------------------------

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)

    # pylint: disable=no-value-for-parameter
    model = VitsModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model_g = model.model_g

    # Inference only
    model_g.eval()

    with torch.no_grad():
        model_g.dec.remove_weight_norm()

    use_eq_conditioning = getattr(model_g, "eq_encoder", None) is not None

    def _infer_forward(text, text_lengths, scales, sid=None, eq_params=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        audio = model_g.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid,
            eq_params=eq_params,
        )[0].unsqueeze(1)

        return audio

    if use_eq_conditioning and (model_g.n_speakers > 1):

        def infer_forward(text, text_lengths, scales, sid, eq_params):
            return _infer_forward(
                text, text_lengths, scales, sid=sid, eq_params=eq_params
            )

    elif use_eq_conditioning:

        def infer_forward(text, text_lengths, scales, eq_params):
            return _infer_forward(text, text_lengths, scales, eq_params=eq_params)

    elif model_g.n_speakers > 1:

        def infer_forward(text, text_lengths, scales, sid):
            return _infer_forward(text, text_lengths, scales, sid=sid)

    else:

        def infer_forward(text, text_lengths, scales):
            return _infer_forward(text, text_lengths, scales)

    model_g.forward = infer_forward  # type: ignore[method-assign,assignment]

    num_symbols = model_g.n_vocab
    num_speakers = model_g.n_speakers

    dummy_input_length = 50
    sequences = torch.randint(
        low=0, high=num_symbols, size=(1, dummy_input_length), dtype=torch.long
    )
    sequence_lengths = torch.LongTensor([sequences.size(1)])

    sid: Optional[torch.LongTensor] = None
    if num_speakers > 1:
        sid = torch.LongTensor([0])

    eq_params: Optional[torch.FloatTensor] = None
    if use_eq_conditioning:
        eq_params = torch.zeros(1, model_g.n_eq_bands, dtype=torch.float32)

    # noise, length, noise_w
    scales = torch.FloatTensor([0.667, 1.0, 0.8])
    dummy_input = [sequences, sequence_lengths, scales]
    input_names = ["input", "input_lengths", "scales"]
    dynamic_axes = {
        "input": {0: "batch_size", 1: "phonemes"},
        "input_lengths": {0: "batch_size"},
        "output": {0: "batch_size", 2: "time"},
    }

    if sid is not None:
        dummy_input.append(sid)
        input_names.append("sid")

    if eq_params is not None:
        dummy_input.append(eq_params)
        input_names.append("eq_params")
        dynamic_axes["eq_params"] = {0: "batch_size"}

    # Export
    # Use dynamo=False to avoid data-dependent control flow errors
    # (VITS flow layers contain dynamic indexing incompatible with torch.export)
    torch.onnx.export(
        model=model_g,
        args=tuple(dummy_input),
        f=output_path,
        verbose=False,
        opset_version=OPSET_VERSION,
        input_names=input_names,
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    _LOGGER.info("Exported model to %s", output_path)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
