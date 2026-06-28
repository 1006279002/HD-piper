import logging
import pathlib

import torch
from lightning.pytorch.cli import LightningCLI

# Allow pathlib.PosixPath in checkpoint loading for PyTorch 2.6+ compatibility.
# Use tuple (class, "pickle_path") because pathlib.PosixPath.__module__ is
# "pathlib._local" in Python 3.13, but the pickle stores "pathlib.PosixPath".
torch.serialization.add_safe_globals([(pathlib.PosixPath, "pathlib.PosixPath")])

from .vits.dataset import VitsDataModule
from .vits.lightning import VitsModel

_LOGGER = logging.getLogger(__package__)


class VitsLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments("data.num_symbols", "model.num_symbols")
        parser.link_arguments("model.num_speakers", "data.num_speakers")
        parser.link_arguments("model.sample_rate", "data.sample_rate")
        parser.link_arguments("model.filter_length", "data.filter_length")
        parser.link_arguments("model.hop_length", "data.hop_length")
        parser.link_arguments("model.win_length", "data.win_length")
        parser.link_arguments("model.segment_size", "data.segment_size")

    def _parse_ckpt_path(self) -> None:
        """Skip hyperparameter parsing from checkpoint.

        The checkpoint's hyperparameters may be incompatible with the current
        CLI schema (e.g., from a different training codebase or old Lightning
        version).  We bypass parsing entirely — the CLI arguments take full
        control of hyperparameters.  Model weights and optimizer states are
        still loaded later by trainer.fit(ckpt_path=...).
        """
        # Intentionally do nothing: let CLI args define all hyperparameters.
        return


def main():
    logging.basicConfig(level=logging.INFO)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    _cli = VitsLightningCLI(  # noqa: ignore=F841
        VitsModel, VitsDataModule, trainer_defaults={"max_epochs": -1}
    )


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
