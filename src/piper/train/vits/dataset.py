"""PyTorch Lightning dataset."""

import csv
import itertools
import json
import logging
import math
import unicodedata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import librosa
import lightning as L
import numpy as np
import torch
from pysilero_vad import SileroVoiceActivityDetector
from torch import FloatTensor, LongTensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from piper.config import PhonemeType, PiperConfig
from piper.phoneme_ids import DEFAULT_PHONEME_ID_MAP
from piper.phoneme_ids import phonemes_to_ids as default_phonemes_to_ids
from piper.phonemize_espeak import EspeakPhonemizer

from .mel_processing import spectrogram_torch
from .utils import get_cache_id

_LOGGER = logging.getLogger(__name__)
VAD_SAMPLE_RATE = 16000
DEFAULT_EQ_TEMPLATE_PARAMS: List[List[float]] = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [-1, 0, 1, 3, 5, 5, 4, 1],
    [-1, 0, 1, 4, 7, 8, 7, 2],
    [0, 1, 2, 4, 5, 5, 4, 1],
    [-1, 0, 1, 3, 5, 7, 8, 2],
    [1, 2, 2, 3, 4, 4, 3, 1],
    [0, 1, 4, 6, 6, 4, 3, 0],
]
DEFAULT_EQ_TEMPLATE_FREQS_HZ = (125.0, 250.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0, 8000.0)

EQ_TARGET_CACHE_MODES = frozenset({"none", "audio"})


def _match_audio_length(audio: FloatTensor, target_length: int) -> FloatTensor:
    """Trim or zero-pad an audio tensor to match the clean profile length."""
    audio_length = audio.size(0)
    if audio_length == target_length:
        return audio
    if audio_length > target_length:
        return audio[:target_length]
    return torch.nn.functional.pad(audio, (0, target_length - audio_length))


def _apply_trim_bounds(
    audio_array: np.ndarray,
    trim_bounds: Tuple[Optional[int], Optional[int]],
    reference_num_samples: int,
) -> np.ndarray:
    """Apply clean-audio trim bounds to an aligned EQ target waveform."""
    first_sample, last_sample = trim_bounds
    if (first_sample is None) and (last_sample is None):
        return audio_array

    if len(audio_array) != reference_num_samples:
        scale = len(audio_array) / max(1, reference_num_samples)
        if first_sample is not None:
            first_sample = int(math.floor(first_sample * scale))
        if last_sample is not None:
            last_sample = int(math.ceil(last_sample * scale))

    return audio_array[first_sample:last_sample]


def _load_eq_target_audio(
    audio_path: Path,
    sample_rate: int,
    trim_bounds: Tuple[Optional[int], Optional[int]],
    reference_num_samples: int,
    target_num_samples: int,
) -> FloatTensor:
    """Load one EQ waveform and align it with the cached clean waveform."""
    audio_array, _ = librosa.load(path=str(audio_path), sr=sample_rate, mono=True)
    audio_array = _apply_trim_bounds(
        audio_array,
        trim_bounds,
        reference_num_samples=reference_num_samples,
    )
    return _match_audio_length(torch.FloatTensor(audio_array), target_num_samples)


@dataclass
class CachedUtterance:
    phoneme_ids_path: Path
    audio_norm_path: Path  # clean/profile 0 audio
    audio_spec_path: Path  # clean/profile 0 spectrogram
    eq_audio_source_paths: Optional[List[Path]] = None  # profiles 1..N -> source WAV
    eq_audio_cache_paths: Optional[List[Path]] = None  # optional profiles 1..N cache
    clean_trim_bounds: Tuple[Optional[int], Optional[int]] = (None, None)
    clean_source_length: int = 0
    text: Optional[str] = None
    speaker_id: Optional[int] = None


class DatasetType(str, Enum):
    TEXT = "text"
    PHONEME_IDS = "phoneme_ids"


class VitsDataModule(L.LightningDataModule):
    def __init__(
        self,
        csv_path: Union[str, Path],
        cache_dir: Union[str, Path],
        espeak_voice: str,
        config_path: Union[str, Path],
        voice_name: str,
        sample_rate: int = 22050,
        audio_dir: Optional[Union[str, Path]] = None,
        alignments_dir: Optional[Union[str, Path]] = None,
        num_symbols: int = 256,
        num_speakers: int = 1,
        batch_size: int = 32,
        validation_split: float = 0.1,
        num_test_examples: int = 5,
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        segment_size: int = 8192,
        num_workers: int = 1,
        prefetch_factor: Optional[int] = 2,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        trim_silence: bool = True,
        keep_seconds_before_silence: float = 0.25,
        keep_seconds_after_silence: float = 0.25,
        phoneme_type: Optional[str] = None,
        dataset_type: Union[str, DatasetType] = DatasetType.TEXT.value,
        phonemes_path: Optional[Union[str, Path]] = None,
        vowel_clusters: Optional[str] = None,
        # EQ conditioning
        eq_audio_base_dir: Optional[Union[str, Path]] = None,
        eq_template_params: Optional[List[List[float]]] = None,
        eq_target_cache: str = "none",
        eq_random_profile_probability: float = 0.85,
        eq_random_noise_std_db: float = 1.25,
        eq_random_global_std_db: float = 0.5,
        eq_random_tilt_std_db: float = 0.75,
        eq_random_local_std_db: float = 1.5,
        eq_random_min_gain_db: float = -8.0,
        eq_random_max_gain_db: float = 15.0,
        use_eq_conditioning: bool = False,
    ) -> None:
        super().__init__()

        self.csv_path = Path(csv_path)
        self.cache_dir = Path(cache_dir)
        self.espeak_voice = espeak_voice
        self.config_path = Path(config_path)
        self.voice_name = voice_name

        self.sample_rate = sample_rate
        self.num_symbols = num_symbols
        self.num_speakers = num_speakers

        if audio_dir is not None:
            self.audio_dir = Path(audio_dir)
        else:
            self.audio_dir = self.csv_path.parent

        if alignments_dir is not None:
            self.alignments_dir = Path(alignments_dir)
        else:
            self.alignments_dir = self.csv_path.parent / "alignments"

        self.batch_size = batch_size
        self.validation_split = validation_split
        self.num_test_examples = num_test_examples

        # Mel
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length

        self.segment_size = segment_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory

        # Silence trimming
        self.trim_silence = trim_silence
        self.keep_seconds_before_silence = keep_seconds_before_silence
        self.keep_seconds_after_silence = keep_seconds_after_silence

        self.piper_config: Optional[PiperConfig] = None
        self.is_multispeaker = self.num_speakers > 1

        # EQ conditioning
        self.eq_audio_base_dir: Optional[Path] = None
        if eq_audio_base_dir is not None:
            self.eq_audio_base_dir = Path(eq_audio_base_dir)

        if eq_template_params is not None:
            self.eq_template_params = eq_template_params
        else:
            # Default: EQ_0 clean + six TTS template gain curves.
            # Frequencies are [125, 250, 500, 1k, 2k, 3k, 4k, 8k] Hz.
            self.eq_template_params = DEFAULT_EQ_TEMPLATE_PARAMS

        self.num_eq_profiles = len(self.eq_template_params)
        self.eq_target_cache = eq_target_cache.strip().lower()
        if self.eq_target_cache not in EQ_TARGET_CACHE_MODES:
            allowed = ", ".join(sorted(EQ_TARGET_CACHE_MODES))
            raise ValueError(
                f"eq_target_cache must be one of: {allowed} (got {eq_target_cache!r})"
            )
        self.eq_random_profile_probability = float(eq_random_profile_probability)
        self.eq_random_noise_std_db = float(eq_random_noise_std_db)
        self.eq_random_global_std_db = float(eq_random_global_std_db)
        self.eq_random_tilt_std_db = float(eq_random_tilt_std_db)
        self.eq_random_local_std_db = float(eq_random_local_std_db)
        self.eq_random_min_gain_db = float(eq_random_min_gain_db)
        self.eq_random_max_gain_db = float(eq_random_max_gain_db)
        if not 0.0 <= self.eq_random_profile_probability <= 1.0:
            raise ValueError("eq_random_profile_probability must be in [0, 1]")
        if self.eq_random_noise_std_db < 0.0:
            raise ValueError("eq_random_noise_std_db must be non-negative")
        if self.eq_random_global_std_db < 0.0:
            raise ValueError("eq_random_global_std_db must be non-negative")
        if self.eq_random_tilt_std_db < 0.0:
            raise ValueError("eq_random_tilt_std_db must be non-negative")
        if self.eq_random_local_std_db < 0.0:
            raise ValueError("eq_random_local_std_db must be non-negative")
        if self.eq_random_min_gain_db > self.eq_random_max_gain_db:
            raise ValueError("eq_random_min_gain_db must not exceed eq_random_max_gain_db")
        if (
            use_eq_conditioning
            and (self.num_eq_profiles > 1)
            and (self.eq_audio_base_dir is None)
        ):
            raise ValueError(
                "eq_audio_base_dir is required when EQ conditioning has "
                "one or more non-clean profiles"
            )
        self.use_eq_conditioning = use_eq_conditioning and (self.num_eq_profiles > 0)
        if self.use_eq_conditioning and (self.eq_random_profile_probability > 0.0):
            expected_bands = len(DEFAULT_EQ_TEMPLATE_FREQS_HZ)
            if any(len(profile) != expected_bands for profile in self.eq_template_params):
                raise ValueError(
                    "Online random EQ profiles require the current 8-band template "
                    f"layout ({expected_bands} values per profile)"
                )

        # Phonemes
        if phoneme_type is None:
            self.phoneme_type = PhonemeType.ESPEAK
        else:
            self.phoneme_type = PhonemeType(phoneme_type)

        if isinstance(dataset_type, DatasetType):
            self.dataset_type = dataset_type
        else:
            self.dataset_type = DatasetType(dataset_type)

        self.phonemes_path: Optional[Path] = None
        if phonemes_path is not None:
            self.phonemes_path = Path(phonemes_path)

        # Vowel clusters to merge into single phonemes (diphthongs).
        # Expecting [["<vowel>", "<vowel>"], ...]
        self.vowel_clusters: Optional[Set[Tuple[str, ...]]] = None
        if vowel_clusters:
            self.vowel_clusters = {tuple(vc) for vc in json.loads(vowel_clusters)}

    def prepare_data(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        phoneme_id_map = DEFAULT_PHONEME_ID_MAP
        phonemes_to_ids = default_phonemes_to_ids

        if self.phonemes_path:
            _LOGGER.debug("Loading phoneme map from %s", self.phonemes_path)
            with open(self.phonemes_path, "r", encoding="utf-8") as phonemes_file:
                phoneme_id_map = json.load(phonemes_file)

            # Ensure ids are lists
            max_phoneme_id = 0
            for phoneme, phoneme_ids in list(phoneme_id_map.items()):
                if not isinstance(phoneme_ids, list):
                    phoneme_ids = [phoneme_ids]
                    phoneme_id_map[phoneme] = phoneme_ids

                max_phoneme_id = max(max_phoneme_id, max(phoneme_ids))
        elif self.phoneme_type == PhonemeType.PINYIN:
            from piper.phonemize_chinese import PHONEME_TO_ID
            from piper.phonemize_chinese import (
                phonemes_to_ids as chinese_phonemes_to_ids,
            )

            phoneme_id_map = PHONEME_TO_ID
            phonemes_to_ids = chinese_phonemes_to_ids

        self.piper_config = PiperConfig(
            num_symbols=self.num_symbols,
            num_speakers=self.num_speakers,
            sample_rate=self.sample_rate,
            espeak_voice=self.espeak_voice,
            phoneme_id_map=phoneme_id_map,
            phoneme_type=self.phoneme_type,
            piper_version="1.5.0",
            vowel_clusters=self.vowel_clusters,
        )

        if self.vowel_clusters:
            _LOGGER.info(
                "Vowel clusters will be merged. This voice will only work with Piper 1.5 or higher."
            )

        speaker_id_map: Dict[str, int] = {}
        if self.is_multispeaker:
            # Generate speaker id map
            with open(self.csv_path, "r", encoding="utf-8") as csv_file:
                reader = csv.reader(csv_file, delimiter="|")
                for row in reader:
                    assert (
                        len(row) >= 3
                    ), "Expected CSV columns for multi-speaker metadata: wav|speaker|text"
                    speaker_name = row[1]
                    if speaker_name in speaker_id_map:
                        continue

                    speaker_id_map[speaker_name] = len(speaker_id_map)

            assert (
                len(speaker_id_map) <= self.num_speakers
            ), "More speakers in metadata than num_speakers"

            if len(speaker_id_map) != self.num_speakers:
                _LOGGER.warning(
                    "Expected %s speakers in the dataset, got %s",
                    self.num_speakers,
                    len(speaker_id_map),
                )

            self.piper_config.speaker_id_map = speaker_id_map

        # Write config
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as config_file:
            json.dump(
                self.piper_config.to_dict(),
                config_file,
                ensure_ascii=False,
                indent=2,
            )

        if self.phoneme_type == PhonemeType.PINYIN:
            from piper.phonemize_chinese import ChinesePhonemizer

            # g2pW -> pinyin -> phonemes
            phonemizer = ChinesePhonemizer(model_dir=Path.cwd() / "local" / "g2pW")

            def phonemize(text: str) -> list[list[str]]:
                return phonemizer.phonemize(text)

        elif self.phoneme_type == PhonemeType.TEXT:
            # text = phonemes

            def phonemize(text: str) -> list[list[str]]:
                return [list(unicodedata.normalize("NFD", text))]

        else:
            # espeak-ng
            phonemizer = EspeakPhonemizer()

            def phonemize(text: str) -> list[list[str]]:
                return phonemizer.phonemize(
                    self.espeak_voice, text, vowel_clusters=self.vowel_clusters
                )

        vad = SileroVoiceActivityDetector()

        num_utterances = 0
        report_prepare: Optional[bool] = None
        with open(self.csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file, delimiter="|")
            for row_number, row in enumerate(reader, start=1):
                utt_id = row[0]
                speaker_id: Optional[int] = None
                if self.is_multispeaker:
                    assert (
                        len(row) >= 3
                    ), "Expected CSV columns for multi-speaker metadata: wav|speaker|text"
                    speaker_name = row[1]
                    speaker_id = speaker_id_map[speaker_name]

                audio_path = self.audio_dir / utt_id
                if not audio_path.exists():
                    audio_path = self.audio_dir / f"{utt_id}.wav"

                if not audio_path.exists():
                    _LOGGER.warning("Missing audio file: %s", audio_path)
                    continue

                if self.dataset_type == DatasetType.PHONEME_IDS:
                    # utt_id|text|phoneme_ids
                    # or
                    # utt_id|speaker_id|text|phoneme_ids
                    text = row[-2]
                else:
                    # utt_id|text
                    # or
                    # utt_id|speaker_id|text
                    text = row[-1]

                cache_id = get_cache_id(row_number, text, speaker_id=speaker_id)

                # text
                text_path = self.cache_dir / f"{cache_id}.txt"
                if not text_path.exists():
                    text_path.write_text(text, encoding="utf-8")

                if self.dataset_type == DatasetType.PHONEME_IDS:
                    phoneme_ids_str = row[-1]

                    # ids separated by whitespace
                    phoneme_ids = [int(p_id) for p_id in phoneme_ids_str.split()]
                    max_phoneme_id = max(phoneme_ids)
                    assert (
                        self.num_symbols > max_phoneme_id
                    ), f"Number of symbols ({self.num_symbols}) must be greater than max phoneme id ({max_phoneme_id})"

                    # phoneme ids
                    phoneme_ids_path = self.cache_dir / f"{cache_id}.phonemes.pt"
                    if not phoneme_ids_path.exists():
                        torch.save(torch.LongTensor(phoneme_ids), phoneme_ids_path)
                        if report_prepare is None:
                            report_prepare = True
                else:
                    # phonemes
                    phonemes: Optional[List[List[str]]] = None
                    phonemes_path = self.cache_dir / f"{cache_id}.phonemes.txt"
                    if not phonemes_path.exists():
                        phonemes = phonemize(text)
                        with open(
                            phonemes_path, "w", encoding="utf-8"
                        ) as phonemes_file:
                            for sentence_phonemes in phonemes:
                                print("".join(sentence_phonemes), file=phonemes_file)

                        if report_prepare is None:
                            report_prepare = True

                    # phoneme ids
                    phoneme_ids_path = self.cache_dir / f"{cache_id}.phonemes.pt"
                    if not phoneme_ids_path.exists():
                        if phonemes is None:
                            phonemes = phonemize(text)

                        phoneme_ids = list(
                            itertools.chain(
                                *(
                                    phonemes_to_ids(
                                        sentence_phonemes, id_map=phoneme_id_map
                                    )
                                    for sentence_phonemes in phonemes
                                )
                            )
                        )
                        torch.save(torch.LongTensor(phoneme_ids), phoneme_ids_path)
                        if report_prepare is None:
                            report_prepare = True

                # normalized clean audio (profile 0)
                norm_audio_path = self.cache_dir / f"{cache_id}.audio.pt"
                trim_metadata_path = self.cache_dir / f"{cache_id}.trim.json"
                audio_norm_tensor: Optional[torch.Tensor] = None
                clean_trim_bounds: Tuple[Optional[int], Optional[int]] = (None, None)
                clean_source_length: Optional[int] = None

                if trim_metadata_path.exists():
                    with trim_metadata_path.open("r", encoding="utf-8") as metadata_file:
                        trim_metadata = json.load(metadata_file)
                    clean_trim_bounds = (
                        trim_metadata.get("first_sample"),
                        trim_metadata.get("last_sample"),
                    )
                    clean_source_length = int(trim_metadata["source_num_samples"])

                needs_clean_source = (
                    (not norm_audio_path.exists()) or (clean_source_length is None)
                )
                refresh_clean_cache = needs_clean_source

                if needs_clean_source:
                    audio_norm_array, audio_sample_rate = librosa.load(
                        path=audio_path, sr=self.sample_rate, mono=True
                    )
                    clean_source_length = len(audio_norm_array)

                    if self.trim_silence:
                        if audio_sample_rate != VAD_SAMPLE_RATE:
                            audio_16khz_array, _sr = librosa.load(
                                path=audio_path, sr=VAD_SAMPLE_RATE, mono=True
                            )
                        else:
                            audio_16khz_array = audio_norm_array
                        clean_trim_bounds = self._get_trim_bounds(
                            audio_norm_array,
                            audio_16khz_array,
                            vad,
                        )
                        audio_norm_array = self._apply_trim_bounds(
                            audio_norm_array,
                            clean_trim_bounds,
                            reference_num_samples=clean_source_length,
                        )

                    audio_norm_tensor = torch.FloatTensor(audio_norm_array)

                if refresh_clean_cache:
                    assert audio_norm_tensor is not None
                    torch.save(audio_norm_tensor, norm_audio_path)
                    assert clean_source_length is not None
                    with trim_metadata_path.open("w", encoding="utf-8") as metadata_file:
                        json.dump(
                            {
                                "source_num_samples": clean_source_length,
                                "first_sample": clean_trim_bounds[0],
                                "last_sample": clean_trim_bounds[1],
                            },
                            metadata_file,
                        )
                    if report_prepare is None:
                        report_prepare = True

                # mel spectrogram (always from clean audio = profile 0)
                audio_spec_path = self.cache_dir / f"{cache_id}.spec.pt"
                if refresh_clean_cache or (not audio_spec_path.exists()):
                    if audio_norm_tensor is None:
                        audio_norm_tensor = torch.load(norm_audio_path)

                    assert audio_norm_tensor is not None
                    spec_audio = (
                        audio_norm_tensor[0]
                        if audio_norm_tensor.dim() == 2
                        else audio_norm_tensor
                    )

                    torch.save(
                        spectrogram_torch(
                            y=spec_audio.unsqueeze(0),
                            n_fft=self.filter_length,
                            sampling_rate=self.sample_rate,
                            hop_size=self.hop_length,
                            win_size=self.win_length,
                            center=False,
                        ).squeeze(0),
                        audio_spec_path,
                    )
                    if report_prepare is None:
                        report_prepare = True

                if self.use_eq_conditioning and (self.num_eq_profiles > 1):
                    assert self.eq_audio_base_dir is not None
                    for eq_idx in range(1, self.num_eq_profiles):
                        eq_audio_file = self._get_eq_audio_file(utt_id, eq_idx)
                        if not eq_audio_file.exists():
                            raise FileNotFoundError(
                                f"Missing EQ_{eq_idx} audio for {utt_id}: {eq_audio_file}"
                            )

                        if self.eq_target_cache != "audio":
                            continue

                        eq_audio_cache_path = (
                            self.cache_dir / f"{cache_id}.audio_{eq_idx}.pt"
                        )
                        if refresh_clean_cache or (not eq_audio_cache_path.exists()):
                            if audio_norm_tensor is None:
                                audio_norm_tensor = torch.load(norm_audio_path)
                                if audio_norm_tensor.dim() == 2:
                                    audio_norm_tensor = audio_norm_tensor[0]

                            assert clean_source_length is not None
                            eq_tensor = _load_eq_target_audio(
                                eq_audio_file,
                                sample_rate=self.sample_rate,
                                trim_bounds=clean_trim_bounds,
                                reference_num_samples=clean_source_length,
                                target_num_samples=audio_norm_tensor.size(0),
                            )
                            torch.save(eq_tensor, eq_audio_cache_path)
                            if report_prepare is None:
                                report_prepare = True

                num_utterances += 1
                if report_prepare:
                    _LOGGER.info("Processing utterances...")
                    report_prepare = False

        _LOGGER.info("Processed %s utterance(s)", num_utterances)

    def setup(self, stage: str) -> None:
        assert self.piper_config is not None

        all_utts: list[CachedUtterance] = []
        speaker_id_map = self.piper_config.speaker_id_map

        with open(self.csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file, delimiter="|")
            for row_number, row in enumerate(reader, start=1):
                utt_id = row[0]
                speaker_id: Optional[int] = None
                if self.is_multispeaker:
                    assert (
                        len(row) >= 3
                    ), "Expected CSV columns for multi-speaker metadata: wav|speaker|text"
                    speaker_name = row[1]
                    speaker_id = speaker_id_map[speaker_name]

                audio_path = self.audio_dir / utt_id
                if not audio_path.exists():
                    audio_path = self.audio_dir / f"{utt_id}.wav"

                if not audio_path.exists():
                    _LOGGER.warning("Missing audio file: %s", audio_path)
                    continue

                if self.dataset_type == DatasetType.PHONEME_IDS:
                    # utt_id|text|phoneme_ids or utt_id|speaker_id|text|phoneme_ids
                    text = row[-2]
                else:
                    # utt_id|text or utt_id|speaker_id|text
                    text = row[-1]

                cache_id = get_cache_id(row_number, text, speaker_id=speaker_id)

                phoneme_ids_path = self.cache_dir / f"{cache_id}.phonemes.pt"
                if not phoneme_ids_path.exists():
                    _LOGGER.warning(
                        "Missing phoneme ids for %s: %s",
                        audio_path,
                        phoneme_ids_path,
                    )
                    continue

                audio_norm_path = self.cache_dir / f"{cache_id}.audio.pt"
                if not audio_norm_path.exists():
                    _LOGGER.warning(
                        "Missing normalized audio for %s: %s",
                        audio_path,
                        audio_norm_path,
                    )
                    continue

                audio_spec_path = self.cache_dir / f"{cache_id}.spec.pt"
                if not audio_spec_path.exists():
                    _LOGGER.warning(
                        "Missing mel spec for %s: %s",
                        audio_path,
                        audio_spec_path,
                    )
                    continue

                eq_audio_source_paths: Optional[List[Path]] = None
                eq_audio_cache_paths: Optional[List[Path]] = None
                clean_trim_bounds: Tuple[Optional[int], Optional[int]] = (None, None)
                clean_source_length = 0
                if self.use_eq_conditioning and (self.num_eq_profiles > 1):
                    trim_metadata_path = self.cache_dir / f"{cache_id}.trim.json"
                    if not trim_metadata_path.exists():
                        raise FileNotFoundError(
                            "Missing EQ trim metadata for "
                            f"{audio_path}: {trim_metadata_path}. Re-run data preparation."
                        )

                    with trim_metadata_path.open("r", encoding="utf-8") as metadata_file:
                        trim_metadata = json.load(metadata_file)
                    clean_trim_bounds = (
                        trim_metadata.get("first_sample"),
                        trim_metadata.get("last_sample"),
                    )
                    clean_source_length = int(trim_metadata["source_num_samples"])

                    assert self.eq_audio_base_dir is not None
                    eq_audio_source_paths = []
                    if self.eq_target_cache == "audio":
                        eq_audio_cache_paths = []
                    for eq_idx in range(1, self.num_eq_profiles):
                        eq_audio_source_path = self._get_eq_audio_file(utt_id, eq_idx)
                        if not eq_audio_source_path.exists():
                            raise FileNotFoundError(
                                f"Missing EQ_{eq_idx} audio for {audio_path}: {eq_audio_source_path}"
                            )
                        eq_audio_source_paths.append(eq_audio_source_path)

                        if eq_audio_cache_paths is not None:
                            eq_audio_cache_path = (
                                self.cache_dir / f"{cache_id}.audio_{eq_idx}.pt"
                            )
                            if not eq_audio_cache_path.exists():
                                raise FileNotFoundError(
                                    "Missing cached EQ audio for "
                                    f"{audio_path}: {eq_audio_cache_path}. Re-run data preparation."
                                )
                            eq_audio_cache_paths.append(eq_audio_cache_path)

                text: Optional[str] = None
                text_path = self.cache_dir / f"{cache_id}.txt"
                if text_path.exists():
                    text = text_path.read_text(encoding="utf-8")

                all_utts.append(
                    CachedUtterance(
                        phoneme_ids_path=phoneme_ids_path,
                        audio_norm_path=audio_norm_path,
                        audio_spec_path=audio_spec_path,
                        eq_audio_source_paths=eq_audio_source_paths,
                        eq_audio_cache_paths=eq_audio_cache_paths,
                        clean_trim_bounds=clean_trim_bounds,
                        clean_source_length=clean_source_length,
                        text=text,
                        speaker_id=speaker_id,
                    )
                )

        full_dataset = VitsDataset(
            all_utts,
            eq_template_params=(
                self.eq_template_params if self.use_eq_conditioning else None
            ),
            sample_rate=self.sample_rate,
        )

        valid_set_size = int(len(full_dataset) * self.validation_split)
        train_set_size = len(full_dataset) - valid_set_size - self.num_test_examples
        train_indices, test_indices, val_indices = random_split(
            full_dataset, [train_set_size, self.num_test_examples, valid_set_size]
        )
        train_dataset = VitsDataset(
            all_utts,
            eq_template_params=(
                self.eq_template_params if self.use_eq_conditioning else None
            ),
            sample_rate=self.sample_rate,
            random_profile_probability=(
                self.eq_random_profile_probability
                if self.use_eq_conditioning
                else 0.0
            ),
            random_noise_std_db=self.eq_random_noise_std_db,
            random_global_std_db=self.eq_random_global_std_db,
            random_tilt_std_db=self.eq_random_tilt_std_db,
            random_local_std_db=self.eq_random_local_std_db,
            random_min_gain_db=self.eq_random_min_gain_db,
            random_max_gain_db=self.eq_random_max_gain_db,
        )
        self.train_dataset = Subset(train_dataset, train_indices.indices)
        self.test_dataset = Subset(full_dataset, test_indices.indices)
        self.val_dataset = Subset(full_dataset, val_indices.indices)

    def train_dataloader(self):
        return self._make_dataloader(
            self.train_dataset,
            collate_fn=UtteranceCollate(
                is_multispeaker=(self.num_speakers > 1),
                segment_size=self.segment_size,
                use_eq=self.use_eq_conditioning,
            ),
        )

    def test_dataloader(self):
        return self._make_dataloader(
            self.test_dataset,
            collate_fn=UtteranceCollate(
                is_multispeaker=(self.num_speakers > 1),
                segment_size=self.segment_size,
                use_eq=self.use_eq_conditioning,
            ),
        )

    def val_dataloader(self):
        return self._make_dataloader(
            self.val_dataset,
            collate_fn=UtteranceCollate(
                is_multispeaker=(self.num_speakers > 1),
                segment_size=self.segment_size,
                use_eq=self.use_eq_conditioning,
            ),
        )

    def _make_dataloader(self, dataset: Dataset, collate_fn) -> DataLoader:
        kwargs = {
            "dataset": dataset,
            "collate_fn": collate_fn,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(**kwargs)

    def _get_eq_audio_file(self, utt_id: str, eq_idx: int) -> Path:
        """Return the expected source audio path for an EQ profile."""
        assert self.eq_audio_base_dir is not None
        eq_audio_dir = self.eq_audio_base_dir / f"EQ_{eq_idx}"
        eq_audio_file = eq_audio_dir / utt_id
        if eq_audio_file.exists():
            return eq_audio_file

        if Path(utt_id).suffix:
            return eq_audio_file

        return eq_audio_dir / f"{utt_id}.wav"

    @staticmethod
    def _match_audio_length(audio: FloatTensor, target_length: int) -> FloatTensor:
        return _match_audio_length(audio, target_length)

    @staticmethod
    def _apply_trim_bounds(
        audio_array: np.ndarray,
        trim_bounds: Tuple[Optional[int], Optional[int]],
        reference_num_samples: int,
    ) -> np.ndarray:
        return _apply_trim_bounds(audio_array, trim_bounds, reference_num_samples)

    def _get_trim_bounds(
        self,
        audio_original_array: np.ndarray,
        audio_16khz_array: np.ndarray,
        vad: SileroVoiceActivityDetector,
        threshold: float = 0.2,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Returns the VAD crop bounds in original-audio sample indices."""
        vad.reset()

        first_chunk: Optional[int] = None
        last_chunk: Optional[int] = None

        samples_per_chunk = vad.chunk_samples()
        seconds_per_chunk: float = samples_per_chunk / VAD_SAMPLE_RATE
        num_chunks = len(audio_16khz_array) // samples_per_chunk

        for chunk_idx in range(num_chunks):
            chunk_offset = chunk_idx * samples_per_chunk
            chunk = audio_16khz_array[chunk_offset : chunk_offset + samples_per_chunk]
            if len(chunk) < samples_per_chunk:
                continue

            if vad.process_array(chunk) >= threshold:
                if first_chunk is None:
                    first_chunk = chunk_idx
                last_chunk = chunk_idx

        if (first_chunk is None) or (last_chunk is None):
            return None, None

        num_original_samples = len(audio_original_array)
        audio_seconds = len(audio_16khz_array) / VAD_SAMPLE_RATE

        first_sec = first_chunk * seconds_per_chunk
        first_sec = max(0, first_sec - self.keep_seconds_before_silence)
        first_sample = int(
            math.floor(num_original_samples * (first_sec / audio_seconds))
        )

        last_sec = (last_chunk + 1) * seconds_per_chunk
        last_sec = min(audio_seconds, last_sec + self.keep_seconds_after_silence)
        last_sample = int(math.ceil(num_original_samples * (last_sec / audio_seconds)))

        return first_sample, last_sample

    def _trim_silence(
        self,
        audio_original_array: np.ndarray,
        audio_16khz_array: np.ndarray,
        vad: SileroVoiceActivityDetector,
        threshold: float = 0.2,
    ) -> np.ndarray:
        """Trims silence from original array."""
        trim_bounds = self._get_trim_bounds(
            audio_original_array,
            audio_16khz_array,
            vad,
            threshold=threshold,
        )
        return self._apply_trim_bounds(
            audio_original_array,
            trim_bounds,
            reference_num_samples=len(audio_original_array),
        )


@dataclass
class UtteranceTensors:
    phoneme_ids: LongTensor
    spectrogram: FloatTensor
    audio_norm: FloatTensor
    speaker_id: Optional[LongTensor] = None
    text: Optional[str] = None
    target_audio: Optional[FloatTensor] = None
    eq_params: Optional[FloatTensor] = None  # [n_bands]
    eq_profile_id: Optional[int] = None

    @property
    def spec_length(self) -> int:
        return self.spectrogram.size(1)


@dataclass
class Batch:
    phoneme_ids: LongTensor
    phoneme_lengths: LongTensor
    spectrograms: FloatTensor
    spectrogram_lengths: LongTensor
    audios: FloatTensor
    audio_lengths: LongTensor
    speaker_ids: Optional[LongTensor] = None
    target_audios: Optional[FloatTensor] = None
    eq_params: Optional[FloatTensor] = None  # [B, n_bands]
    eq_profile_ids: Optional[LongTensor] = None


class VitsDataset(Dataset):
    def __init__(
        self,
        utts: list[CachedUtterance],
        eq_template_params: Optional[List[List[float]]] = None,
        sample_rate: int = 22050,
        random_profile_probability: float = 0.0,
        random_noise_std_db: float = 1.25,
        random_global_std_db: float = 0.5,
        random_tilt_std_db: float = 0.75,
        random_local_std_db: float = 1.5,
        random_min_gain_db: float = -8.0,
        random_max_gain_db: float = 15.0,
    ):
        self.utts = utts
        self.eq_template_params = eq_template_params
        self.use_eq = (eq_template_params is not None) and (len(eq_template_params) > 0)
        self.num_eq_profiles = len(eq_template_params) if self.use_eq else 0
        self.sample_rate = sample_rate
        self.random_profile_probability = random_profile_probability
        self.random_noise_std_db = random_noise_std_db
        self.random_global_std_db = random_global_std_db
        self.random_tilt_std_db = random_tilt_std_db
        self.random_local_std_db = random_local_std_db
        self.random_min_gain_db = random_min_gain_db
        self.random_max_gain_db = random_max_gain_db

        if self.random_profile_probability > 0.0:
            assert self.eq_template_params is not None
            if any(
                len(profile) != len(DEFAULT_EQ_TEMPLATE_FREQS_HZ)
                for profile in self.eq_template_params
            ):
                raise ValueError(
                    "Random EQ profiles require the current 8-band template layout"
                )

    def _sample_random_eq_profile(self) -> FloatTensor:
        """Interpolate template curves, then add smooth and local gain perturbations."""
        assert self.eq_template_params is not None
        template_profiles = torch.tensor(
            self.eq_template_params,
            dtype=torch.float32,
        )
        num_profiles, num_bands = template_profiles.shape

        if num_profiles == 1 or torch.rand(()) < 0.6:
            base_profile = template_profiles[torch.randint(num_profiles, ())]
        else:
            first_idx = torch.randint(num_profiles, ())
            second_idx = torch.randint(num_profiles, ())
            interpolation = torch.rand(())
            base_profile = torch.lerp(
                template_profiles[first_idx],
                template_profiles[second_idx],
                interpolation,
            )

        # Correlated per-band noise preserves the smooth geometry of an EQ curve.
        smooth_noise = F.avg_pool1d(
            torch.randn(1, 1, num_bands),
            kernel_size=3,
            stride=1,
            padding=1,
        ).squeeze(0).squeeze(0)
        smooth_noise = smooth_noise * self.random_noise_std_db

        band_positions = torch.linspace(-1.0, 1.0, num_bands)
        global_offset = torch.randn(()) * self.random_global_std_db
        tilt = torch.randn(()) * self.random_tilt_std_db * band_positions

        local_center = torch.rand(()) * (num_bands - 1)
        local_width = 0.75 + (torch.rand(()) * 1.75)
        local_shape = torch.exp(
            -0.5 * ((torch.arange(num_bands) - local_center) / local_width).square()
        )
        local_shape = local_shape * (torch.randn(()) * self.random_local_std_db)

        return torch.clamp(
            base_profile + smooth_noise + global_offset + tilt + local_shape,
            min=self.random_min_gain_db,
            max=self.random_max_gain_db,
        )

    def _render_random_eq_target(
        self,
        clean_audio: FloatTensor,
        eq_params: FloatTensor,
    ) -> FloatTensor:
        """Render an unpersisted static EQ target from the sampled gain curve."""
        if clean_audio.numel() == 0:
            return clean_audio

        rfft_freqs = np.fft.rfftfreq(clean_audio.numel(), d=1.0 / self.sample_rate)
        control_freqs = np.asarray(DEFAULT_EQ_TEMPLATE_FREQS_HZ, dtype=np.float64)
        control_gains = eq_params.detach().cpu().numpy().astype(np.float64)
        log_freqs = np.log(np.maximum(rfft_freqs, control_freqs[0]))
        gain_curve = np.interp(
            log_freqs,
            np.log(control_freqs),
            control_gains,
            left=control_gains[0],
            right=control_gains[-1],
        )
        gain_db = torch.from_numpy(gain_curve.astype(np.float32))
        linear_gain = torch.pow(10.0, gain_db / 20.0)

        rendered = torch.fft.irfft(
            torch.fft.rfft(clean_audio) * linear_gain,
            n=clean_audio.numel(),
        )
        peak = rendered.abs().amax()
        if peak > 0.99:
            rendered = rendered * (0.99 / peak)

        return rendered

    def __len__(self):
        return len(self.utts)

    def __getitem__(self, idx) -> UtteranceTensors:
        return self.get_utterance(idx)

    def get_utterance(
        self,
        idx: int,
        eq_idx: Optional[int] = None,
    ) -> UtteranceTensors:
        utt = self.utts[idx]
        target_audio: Optional[FloatTensor] = None
        eq_params: Optional[FloatTensor] = None
        eq_profile_id: Optional[int] = None

        audio_norm = torch.load(utt.audio_norm_path)
        if audio_norm.dim() == 2:
            # Backward compatibility with cache files written by old stacked layout.
            audio_norm = audio_norm[0]

        if self.use_eq:
            use_random_profile = (
                (eq_idx is None)
                and (self.random_profile_probability > 0.0)
                and (torch.rand(()) < self.random_profile_probability)
            )
            if use_random_profile:
                eq_profile_id = -1
                eq_params = self._sample_random_eq_profile()
                target_audio = self._render_random_eq_target(audio_norm, eq_params)
            else:
                if eq_idx is None:
                    eq_idx = torch.randint(0, self.num_eq_profiles, (1,)).item()
                eq_profile_id = eq_idx
                if eq_idx == 0:
                    target_audio = audio_norm
                else:
                    source_idx = eq_idx - 1
                    assert utt.eq_audio_source_paths is not None
                    eq_audio_cache_path = (
                        utt.eq_audio_cache_paths[source_idx]
                        if utt.eq_audio_cache_paths is not None
                        else None
                    )
                    if eq_audio_cache_path is not None:
                        target_audio = torch.load(eq_audio_cache_path)
                        if target_audio.dim() == 2:
                            target_audio = target_audio[0]
                    else:
                        target_audio = _load_eq_target_audio(
                            utt.eq_audio_source_paths[source_idx],
                            sample_rate=self.sample_rate,
                            trim_bounds=utt.clean_trim_bounds,
                            reference_num_samples=utt.clean_source_length,
                            target_num_samples=audio_norm.size(0),
                        )
                eq_params = FloatTensor(self.eq_template_params[eq_idx])

        return UtteranceTensors(
            phoneme_ids=torch.load(utt.phoneme_ids_path),
            audio_norm=audio_norm,
            spectrogram=torch.load(utt.audio_spec_path),
            speaker_id=(
                LongTensor([utt.speaker_id]) if utt.speaker_id is not None else None
            ),
            text=utt.text,
            target_audio=target_audio,
            eq_params=eq_params,
            eq_profile_id=eq_profile_id,
        )


class UtteranceCollate:
    def __init__(
        self,
        is_multispeaker: bool,
        segment_size: int,
        use_eq: bool = False,
    ):
        self.is_multispeaker = is_multispeaker
        self.segment_size = segment_size
        self.use_eq = use_eq

    def __call__(self, utterances: Sequence[UtteranceTensors]) -> Batch:
        num_utterances = len(utterances)
        assert num_utterances > 0, "No utterances"

        max_phonemes_length = 0
        max_spec_length = 0
        max_audio_length = 0
        max_target_audio_length = 0

        num_mels = 0

        # Determine lengths
        for utt_idx, utt in enumerate(utterances):
            assert utt.spectrogram is not None
            assert utt.audio_norm is not None

            phoneme_length = utt.phoneme_ids.size(0)
            spec_length = utt.spectrogram.size(1)
            audio_length = utt.audio_norm.size(0)

            max_phonemes_length = max(max_phonemes_length, phoneme_length)
            max_spec_length = max(max_spec_length, spec_length)
            max_audio_length = max(max_audio_length, audio_length)

            if self.use_eq:
                assert utt.target_audio is not None
                target_audio_length = utt.target_audio.size(0)
                max_target_audio_length = max(
                    max_target_audio_length,
                    target_audio_length,
                )

            num_mels = utt.spectrogram.size(0)
            if self.is_multispeaker:
                assert utt.speaker_id is not None, "Missing speaker id"

        # Audio cannot be smaller than segment size (8192)
        max_audio_length = max(max_audio_length, self.segment_size)
        max_target_audio_length = max(max_target_audio_length, self.segment_size)

        # Create padded tensors
        phonemes_padded = LongTensor(num_utterances, max_phonemes_length)
        spec_padded = FloatTensor(num_utterances, num_mels, max_spec_length)
        audio_padded = FloatTensor(num_utterances, 1, max_audio_length)

        phonemes_padded.zero_()
        spec_padded.zero_()
        audio_padded.zero_()

        phoneme_lengths = LongTensor(num_utterances)
        spec_lengths = LongTensor(num_utterances)
        audio_lengths = LongTensor(num_utterances)

        # EQ tensors
        target_audio_padded: Optional[FloatTensor] = None
        eq_params_padded: Optional[FloatTensor] = None
        eq_profile_ids: Optional[LongTensor] = None
        if self.use_eq:
            target_audio_padded = FloatTensor(
                num_utterances,
                1,
                max_target_audio_length,
            )
            target_audio_padded.zero_()
            n_bands = next(
                len(utt.eq_params) for utt in utterances if utt.eq_params is not None
            )
            eq_params_padded = FloatTensor(num_utterances, n_bands)
            eq_params_padded.zero_()
            eq_profile_ids = LongTensor(num_utterances)

        speaker_ids: Optional[LongTensor] = None
        if self.is_multispeaker:
            speaker_ids = LongTensor(num_utterances)

        # Sort by decreasing spectrogram length
        sorted_utterances = sorted(
            utterances, key=lambda u: u.spectrogram.size(1), reverse=True
        )
        for utt_idx, utt in enumerate(sorted_utterances):
            phoneme_length = utt.phoneme_ids.size(0)

            spec_length = utt.spectrogram.size(1)
            audio_length = utt.audio_norm.size(0)

            phonemes_padded[utt_idx, :phoneme_length] = utt.phoneme_ids
            phoneme_lengths[utt_idx] = phoneme_length

            spec_padded[utt_idx, :, :spec_length] = utt.spectrogram
            spec_lengths[utt_idx] = spec_length

            audio_padded[utt_idx, :, :audio_length] = utt.audio_norm
            audio_lengths[utt_idx] = audio_length

            # Selected target profile tensors
            if self.use_eq:
                assert target_audio_padded is not None
                assert eq_params_padded is not None
                assert eq_profile_ids is not None
                assert utt.target_audio is not None
                assert utt.eq_params is not None
                assert utt.eq_profile_id is not None
                target_audio_length = utt.target_audio.size(0)
                target_audio_padded[utt_idx, :, :target_audio_length] = utt.target_audio
                eq_params_padded[utt_idx] = utt.eq_params
                eq_profile_ids[utt_idx] = utt.eq_profile_id

            if utt.speaker_id is not None:
                assert speaker_ids is not None
                speaker_ids[utt_idx] = utt.speaker_id

        return Batch(
            phoneme_ids=phonemes_padded,
            phoneme_lengths=phoneme_lengths,
            spectrograms=spec_padded,
            spectrogram_lengths=spec_lengths,
            audios=audio_padded,
            audio_lengths=audio_lengths,
            speaker_ids=speaker_ids,
            target_audios=target_audio_padded,
            eq_params=eq_params_padded,
            eq_profile_ids=eq_profile_ids,
        )
