"""PyTorch Lightning module."""

import ast
import json
import logging
import math
import operator
from functools import reduce
from pathlib import Path
from typing import Optional

import lightning as L
import torch
from torch import autocast
from torch.nn import functional as F

from .commons import slice_segments
from .dataset import Batch
from .losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from .mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from .models import MultiPeriodDiscriminator, SynthesizerTrn

_LOGGER = logging.getLogger(__name__)


class VitsModel(L.LightningModule):
    def __init__(
        self,
        batch_size: int = 32,
        sample_rate: int = 22050,
        num_symbols: int = 256,
        num_speakers: int = 1,
        # audio
        resblock="2",
        resblock_kernel_sizes=(3, 5, 7),
        resblock_dilation_sizes=(
            (1, 2),
            (2, 6),
            (3, 12),
        ),
        upsample_rates=(8, 8, 4),
        upsample_initial_channel=256,
        upsample_kernel_sizes=(16, 16, 8),
        # mel
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        mel_channels: int = 80,
        mel_fmin: float = 0.0,
        mel_fmax: Optional[float] = None,
        # model
        inter_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        n_layers_q: int = 3,
        use_spectral_norm: bool = False,
        gin_channels: int = 0,
        use_sdp: bool = True,
        segment_size: int = 8192,
        # training
        learning_rate: float = 2e-4,
        learning_rate_d: float = 1e-4,
        betas: tuple[float, float] = (0.8, 0.99),
        betas_d: tuple[float, float] = (0.5, 0.9),
        eps: float = 1e-9,
        lr_decay: float = 0.999875,
        lr_decay_d: float = 0.9999,
        init_lr_ratio: float = 1.0,
        warmup_epochs: int = 0,
        c_mel: int = 45,
        c_kl: float = 1.0,
        grad_clip: Optional[float] = None,
        vocoder_warmstart_ckpt: Optional[str] = None,
        # EQ conditioning
        use_eq_conditioning: bool = False,
        n_eq_bands: int = 8,
        eq_cond_dim: int = 256,
        eq_freq_bands_hz: Optional[tuple[float, ...]] = None,
        eq_gain_norm_factor: float = 15.0,
        # unused
        dataset: object = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        if isinstance(self.hparams.resblock_kernel_sizes, str):
            self.hparams.resblock_kernel_sizes = ast.literal_eval(
                self.hparams.resblock_kernel_sizes
            )

        if isinstance(self.hparams.resblock_dilation_sizes, str):
            self.hparams.resblock_dilation_sizes = ast.literal_eval(
                self.hparams.resblock_dilation_sizes
            )

        if isinstance(self.hparams.upsample_rates, str):
            self.hparams.upsample_rates = ast.literal_eval(self.hparams.upsample_rates)

        if isinstance(self.hparams.upsample_kernel_sizes, str):
            self.hparams.upsample_kernel_sizes = ast.literal_eval(
                self.hparams.upsample_kernel_sizes
            )

        if isinstance(self.hparams.betas, str):
            self.hparams.betas = ast.literal_eval(self.hparams.betas)

        if isinstance(self.hparams.eq_freq_bands_hz, str):
            self.hparams.eq_freq_bands_hz = ast.literal_eval(
                self.hparams.eq_freq_bands_hz
            )
        if self.hparams.eq_freq_bands_hz is not None:
            self.hparams.eq_freq_bands_hz = tuple(
                float(freq) for freq in self.hparams.eq_freq_bands_hz
            )

        expected_hop_length = reduce(operator.mul, self.hparams.upsample_rates, 1)
        if expected_hop_length != hop_length:
            raise ValueError("Upsample rates do not match hop length")

        # Need to use manual optimization because we have multiple optimizers
        self.automatic_optimization = False

        self.batch_size = batch_size

        if (self.hparams.num_speakers > 1) and (self.hparams.gin_channels <= 0):
            # Default gin_channels for multi-speaker model
            self.hparams.gin_channels = 512

        # Used to partially load the state dict from a checkpoint.
        # Only the text/phoneme agnostic portions are loaded.
        self._vocoder_warmstart_ckpt = vocoder_warmstart_ckpt

        # Set up models
        self.model_g = SynthesizerTrn(
            n_vocab=num_symbols,
            spec_channels=self.hparams.filter_length // 2 + 1,
            segment_size=self.hparams.segment_size // self.hparams.hop_length,
            inter_channels=self.hparams.inter_channels,
            hidden_channels=self.hparams.hidden_channels,
            filter_channels=self.hparams.filter_channels,
            n_heads=self.hparams.n_heads,
            n_layers=self.hparams.n_layers,
            kernel_size=self.hparams.kernel_size,
            p_dropout=self.hparams.p_dropout,
            resblock=self.hparams.resblock,
            resblock_kernel_sizes=self.hparams.resblock_kernel_sizes,
            resblock_dilation_sizes=self.hparams.resblock_dilation_sizes,
            upsample_rates=self.hparams.upsample_rates,
            upsample_initial_channel=self.hparams.upsample_initial_channel,
            upsample_kernel_sizes=self.hparams.upsample_kernel_sizes,
            n_speakers=self.hparams.num_speakers,
            gin_channels=self.hparams.gin_channels,
            use_sdp=self.hparams.use_sdp,
            n_eq_bands=(
                self.hparams.n_eq_bands if self.hparams.use_eq_conditioning else 8
            ),
            eq_cond_dim=(
                self.hparams.eq_cond_dim if self.hparams.use_eq_conditioning else 0
            ),
            eq_mel_channels=self.hparams.mel_channels,
            eq_sample_rate=self.hparams.sample_rate,
            eq_mel_fmin=self.hparams.mel_fmin,
            eq_mel_fmax=self.hparams.mel_fmax,
            eq_freq_bands_hz=self.hparams.eq_freq_bands_hz,
            eq_gain_norm_factor=self.hparams.eq_gain_norm_factor,
        )
        self.model_d = MultiPeriodDiscriminator(
            use_spectral_norm=self.hparams.use_spectral_norm
        )

    def forward(self, text, text_lengths, scales, sid=None, eq_params=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        audio, *_ = self.model_g.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid,
            eq_params=eq_params,
        )

        return audio

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """Allow loading checkpoints that don't have EQ-related layers.

        When use_eq_conditioning is enabled but the checkpoint was trained
        without EQ, the new EQ layers (eq_encoder, dec.eq_cond) are
        silently initialized randomly instead of raising an error.
        """
        if self.hparams.use_eq_conditioning:
            strict = False
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def _compute_loss(self, batch: Batch):
        # g step
        x, x_lengths, y, _, spec, spec_lengths, speaker_ids = (
            batch.phoneme_ids,
            batch.phoneme_lengths,
            batch.audios,
            batch.audio_lengths,
            batch.spectrograms,
            batch.spectrogram_lengths,
            batch.speaker_ids if batch.speaker_ids is not None else None,
        )

        # EQ parameters for conditioning
        eq_params = batch.eq_params if self.hparams.use_eq_conditioning else None
        target_audio = (
            batch.target_audios
            if self.hparams.use_eq_conditioning and batch.target_audios is not None
            else y
        )

        (
            y_hat,
            l_length,
            _attn,
            ids_slice,
            _x_mask,
            z_mask,
            (_z, z_p, m_p, logs_p, _m_q, logs_q),
        ) = self.model_g(
            x, x_lengths, spec, spec_lengths, speaker_ids, eq_params=eq_params
        )

        y_disc = slice_segments(
            target_audio,
            ids_slice * self.hparams.hop_length,
            self.hparams.segment_size,
        )  # slice

        # Trim to avoid padding issues
        y_hat = y_hat[..., : y_disc.shape[-1]]

        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            self.hparams.filter_length,
            self.hparams.mel_channels,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.mel_fmin,
            self.hparams.mel_fmax,
        )

        if self.hparams.use_eq_conditioning and batch.target_audios is not None:
            y_mel = mel_spectrogram_torch(
                y_disc.squeeze(1),
                self.hparams.filter_length,
                self.hparams.mel_channels,
                self.hparams.sample_rate,
                self.hparams.hop_length,
                self.hparams.win_length,
                self.hparams.mel_fmin,
                self.hparams.mel_fmax,
            )
        else:
            target_mel = spec_to_mel_torch(
                spec,
                self.hparams.filter_length,
                self.hparams.mel_channels,
                self.hparams.sample_rate,
                self.hparams.mel_fmin,
                self.hparams.mel_fmax,
            )
            y_mel = slice_segments(
                target_mel,
                ids_slice,
                self.hparams.segment_size // self.hparams.hop_length,
            )

        self._set_requires_grad(self.model_d, False)
        _y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.model_d(y_disc, y_hat)

        with autocast(self.device.type, enabled=False):
            # Generator loss
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hparams.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.hparams.c_kl

            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, _losses_gen = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

        # d step
        self._set_requires_grad(self.model_d, True)
        y_d_hat_r, y_d_hat_g, _, _ = self.model_d(y_disc, y_hat.detach())

        with autocast(self.device.type, enabled=False):
            # Discriminator
            loss_disc, _losses_disc_r, _losses_disc_g = discriminator_loss(
                y_d_hat_r, y_d_hat_g
            )
            loss_disc_all = loss_disc

        return loss_gen_all, loss_disc_all, loss_mel

    @staticmethod
    def _set_requires_grad(module: torch.nn.Module, requires_grad: bool):
        for parameter in module.parameters():
            parameter.requires_grad_(requires_grad)

    def training_step(self, batch: Batch, batch_idx: int):
        opt_g, opt_d = self.optimizers()
        loss_g, loss_d, loss_mel = self._compute_loss(batch)

        self.log("loss_g", loss_g, batch_size=self.batch_size)
        self.log("loss_mel", loss_mel, batch_size=self.batch_size)
        opt_g.zero_grad()
        self.manual_backward(loss_g)
        if self.hparams.grad_clip is not None:
            self.clip_gradients(
                opt_g,
                gradient_clip_val=self.hparams.grad_clip,
                gradient_clip_algorithm="norm",
            )
        opt_g.step()

        self.log("loss_d", loss_d, batch_size=self.batch_size)
        opt_d.zero_grad()
        self.manual_backward(loss_d)
        if self.hparams.grad_clip is not None:
            self.clip_gradients(
                opt_d,
                gradient_clip_val=self.hparams.grad_clip,
                gradient_clip_algorithm="norm",
            )
        opt_d.step()

    def validation_step(self, batch: Batch, batch_idx: int):
        loss_g, _loss_d, loss_mel = self._compute_loss(batch)
        val_loss = loss_g  # only generator loss matters
        self.log("val_loss", val_loss, batch_size=self.batch_size)
        self.log("val_mel_loss", loss_mel, batch_size=self.batch_size)
        return val_loss

    def on_validation_end(self) -> None:
        # Generate audio examples after validation, but not during sanity check
        if self.trainer.sanity_checking:
            return super().on_validation_end()

        if (
            getattr(self, "logger", None)
            and hasattr(self.logger, "experiment")
            and hasattr(self.logger.experiment, "add_audio")
        ):
            datamodule = self.trainer.datamodule
            eq_profiles = (
                datamodule.eq_template_params
                if self.hparams.use_eq_conditioning
                else [[0.0] * self.hparams.n_eq_bands]
            )

            for utt_idx in range(len(datamodule.test_dataset)):
                test_utt = self._get_dataset_utterance(
                    datamodule.test_dataset,
                    utt_idx,
                    eq_idx=0 if self.hparams.use_eq_conditioning else None,
                )
                text = test_utt.phoneme_ids.unsqueeze(0).to(self.device)
                text_lengths = torch.LongTensor([len(test_utt.phoneme_ids)]).to(
                    self.device
                )
                scales = [0.667, 1.0, 0.8]
                sid = (
                    test_utt.speaker_id.to(self.device)
                    if test_utt.speaker_id is not None
                    else None
                )

                tag_base = test_utt.text or str(utt_idx)

                # Log mel spectrograms as images.
                # Mel values are log-magnitude (e.g. [-12, 2]), so min-max
                # normalize each spectrogram to [0,1] for proper visualization.
                def _norm_mel(m):
                    m = m.squeeze(0).flip(0)  # [mel, T], freq low→high
                    m_min, m_max = m.min(), m.max()
                    if m_max > m_min:
                        m = (m - m_min) / (m_max - m_min)
                    return m.unsqueeze(0)  # [1, mel, T]

                for eq_idx, eq_profile in enumerate(eq_profiles):
                    eq_params = (
                        torch.tensor(
                            [eq_profile],
                            dtype=torch.float32,
                            device=self.device,
                        )
                        if self.hparams.use_eq_conditioning
                        else None
                    )
                    audio = self(
                        text,
                        text_lengths,
                        scales,
                        sid=sid,
                        eq_params=eq_params,
                    ).detach()
                    mel = mel_spectrogram_torch(
                        audio.squeeze(1),
                        self.hparams.filter_length,
                        self.hparams.mel_channels,
                        self.hparams.sample_rate,
                        self.hparams.hop_length,
                        self.hparams.win_length,
                        self.hparams.mel_fmin,
                        self.hparams.mel_fmax,
                    )
                    target_utt = self._get_dataset_utterance(
                        datamodule.test_dataset,
                        utt_idx,
                        eq_idx=eq_idx if self.hparams.use_eq_conditioning else None,
                        load_target_spectrogram=True,
                    )
                    target_spec = (
                        target_utt.target_spectrogram
                        if self.hparams.use_eq_conditioning
                        and target_utt.target_spectrogram is not None
                        else target_utt.spectrogram
                    )
                    target_mel = spec_to_mel_torch(
                        target_spec.unsqueeze(0).to(self.device),
                        self.hparams.filter_length,
                        self.hparams.mel_channels,
                        self.hparams.sample_rate,
                        self.hparams.mel_fmin,
                        self.hparams.mel_fmax,
                    )
                    min_len = min(mel.size(-1), target_mel.size(-1))
                    mel_loss = F.l1_loss(
                        mel[..., :min_len],
                        target_mel[..., :min_len],
                    )

                    self.logger.experiment.add_audio(
                        f"{tag_base}/EQ_{eq_idx}",
                        audio.squeeze(0),
                        sample_rate=self.hparams.sample_rate,
                    )
                    self.logger.experiment.add_scalar(
                        f"val_mel/{tag_base}_EQ_{eq_idx}_loss",
                        mel_loss.item(),
                        self.global_step,
                    )
                    self.logger.experiment.add_image(
                        f"{tag_base}/mel_EQ_{eq_idx}",
                        _norm_mel(mel),
                        self.global_step,
                    )

        return super().on_validation_end()

    @staticmethod
    def _get_dataset_utterance(
        dataset,
        idx: int,
        eq_idx: Optional[int] = None,
        load_target_spectrogram: bool = False,
    ):
        while hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
            idx = dataset.indices[idx]
            dataset = dataset.dataset

        if hasattr(dataset, "get_utterance"):
            return dataset.get_utterance(
                idx,
                eq_idx=eq_idx,
                load_target_spectrogram=load_target_spectrogram,
            )

        return dataset[idx]

    def configure_optimizers(self):
        optimizers = [
            torch.optim.AdamW(
                self.model_g.parameters(),
                lr=self.hparams.learning_rate,
                betas=self.hparams.betas,
                eps=self.hparams.eps,
            ),
            torch.optim.AdamW(
                self.model_d.parameters(),
                lr=self.hparams.learning_rate_d,
                betas=self.hparams.betas_d,
                eps=self.hparams.eps,
            ),
        ]
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                optimizers[0], gamma=self.hparams.lr_decay
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                optimizers[1], gamma=self.hparams.lr_decay_d
            ),
        ]

        return optimizers, schedulers

    def _warmstart_vocoder_from_ckpt(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        old_sd = ckpt["state_dict"]
        new_sd = self.state_dict()

        # Copy ALL matching parameters from old checkpoint.
        # New EQ layers (eq_encoder.*, dec.eq_cond.*) don't exist in
        # old checkpoint and will stay randomly initialized.
        KEEP_PREFIXES = (
            "model_g.dec.",
            "model_g.enc_q.",
            "model_g.flow.",
            "model_g.enc_p.",
            "model_g.dp.",
            "model_g.emb_g.",
            "model_d.",
        )

        copied = 0
        for k, v in old_sd.items():
            if not k.startswith(KEEP_PREFIXES):
                continue
            if (k in new_sd) and (new_sd[k].shape == v.shape):
                new_sd[k] = v
                copied += 1

        self.load_state_dict(new_sd, strict=False)
        _LOGGER.info(f"[warmstart] Copied {copied} parameters from {ckpt_path}")

    def on_fit_start(self):
        # Called once at the start of fit()
        self._validate_eq_profiles()

        if self._vocoder_warmstart_ckpt is not None:
            # Make sure we're on the correct device
            self._warmstart_vocoder_from_ckpt(self._vocoder_warmstart_ckpt)
            # Avoid re-running if Trainer restarts
            self._vocoder_warmstart_ckpt = None

        # Save EQ-aware model config for ONNX export / inference reference
        self._save_eq_model_config()

    def _validate_eq_profiles(self):
        if not self.hparams.use_eq_conditioning:
            return

        datamodule = self.trainer.datamodule
        eq_profiles = getattr(datamodule, "eq_template_params", None)
        if not eq_profiles:
            raise ValueError("EQ conditioning requires at least one EQ profile")

        n_bands = len(eq_profiles[0])
        if n_bands != self.hparams.n_eq_bands:
            raise ValueError(
                "EQ profile band count does not match model.n_eq_bands: "
                f"{n_bands} != {self.hparams.n_eq_bands}"
            )
        if (
            self.hparams.eq_freq_bands_hz is not None
            and len(self.hparams.eq_freq_bands_hz) != n_bands
        ):
            raise ValueError(
                "eq_freq_bands_hz length must match the EQ profile band count: "
                f"{len(self.hparams.eq_freq_bands_hz)} != {n_bands}"
            )

        for eq_idx, eq_profile in enumerate(eq_profiles):
            if len(eq_profile) != n_bands:
                raise ValueError(
                    f"EQ_{eq_idx} has {len(eq_profile)} bands, expected {n_bands}"
                )

        if any(float(value) != 0.0 for value in eq_profiles[0]):
            raise ValueError("EQ_0 is the clean profile and must contain only zeros")

    def _save_eq_model_config(self):
        """Save an EQ-prefixed ONNX companion voice config."""
        datamodule = self.trainer.datamodule
        config_path = Path(datamodule.config_path)
        eq_config_path = config_path.with_name("EQ.onnx.json")
        eq_profiles = (
            getattr(datamodule, "eq_template_params", [])
            if self.hparams.use_eq_conditioning
            else []
        )
        freq_bands_hz = self._get_eq_freq_bands_hz()

        if getattr(datamodule, "piper_config", None) is not None:
            config = datamodule.piper_config.to_dict()
        elif config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {}

        config["eq"] = {
            # Audio
            "sample_rate": self.hparams.sample_rate,
            "hop_length": self.hparams.hop_length,
            "filter_length": self.hparams.filter_length,
            "win_length": self.hparams.win_length,
            "mel_channels": self.hparams.mel_channels,
            "mel_fmin": self.hparams.mel_fmin,
            "mel_fmax": self.hparams.mel_fmax,
            # Generator (HiFi-GAN)
            "inter_channels": self.hparams.inter_channels,
            "resblock": self.hparams.resblock,
            "resblock_kernel_sizes": self.hparams.resblock_kernel_sizes,
            "resblock_dilation_sizes": self.hparams.resblock_dilation_sizes,
            "upsample_rates": self.hparams.upsample_rates,
            "upsample_initial_channel": self.hparams.upsample_initial_channel,
            "upsample_kernel_sizes": self.hparams.upsample_kernel_sizes,
            # ── EQ conditioning ──
            # eq_params is a RUNTIME INPUT to the ONNX model, not a fixed weight.
            # You control EQ effect by passing different eq_params at inference time.
            "use_eq_conditioning": self.hparams.use_eq_conditioning,
            "n_eq_bands": self.hparams.n_eq_bands,
            "n_eq_profiles": len(eq_profiles),
            "eq_cond_dim": self.hparams.eq_cond_dim,
            "eq_profiles": {
                f"EQ_{eq_idx}": eq_profile
                for eq_idx, eq_profile in enumerate(eq_profiles)
            },
            "eq_condition_type": "template_gain_curve",
            # What each eq_params value means:
            "eq_params_interface": {
                "description": "eq_params[i] = template EQ gain in dB at freq_bands_hz[i]",
                "input_shape": [1, self.hparams.n_eq_bands],
                "value_range": "template gain dB, typically -20 to +15",
                "normalization": "model divides input by eq_gain_norm_factor internally",
                "freq_bands_hz": freq_bands_hz,
                "eq_gain_norm_factor": self.hparams.eq_gain_norm_factor,
                "embedding": "log-frequency interpolation to mel bins plus control-point MLP",
            },
        }

        eq_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eq_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        _LOGGER.info(f"[EQ ONNX config] Saved to {eq_config_path}")

    def _get_eq_freq_bands_hz(self) -> list[float]:
        if self.hparams.eq_freq_bands_hz is not None:
            return [float(freq) for freq in self.hparams.eq_freq_bands_hz]

        if self.hparams.n_eq_bands == 8:
            return [125.0, 250.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0, 8000.0]

        min_freq = math.log10(125.0)
        max_freq = math.log10(8000.0)
        return [
            10
            ** (
                min_freq
                + (max_freq - min_freq) * idx / max(1, self.hparams.n_eq_bands - 1)
            )
            for idx in range(self.hparams.n_eq_bands)
        ]
