"""
Speaker Diarization Pipeline
Combines: pyannote diarization (preferred) -> fallback VAD + ECAPA-TDNN + AHC clustering
"""

import tempfile
import time
from pathlib import Path
from typing import Optional, List, Union, BinaryIO
from dataclasses import dataclass, field

import numpy as np
import torch
import torchaudio
from loguru import logger

from models.embedder import EcapaTDNNEmbedder
from models.clusterer import SpeakerClusterer


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: str
    duration: float = field(init=False)

    def __post_init__(self):
        self.duration = round(self.end - self.start, 3)

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": self.duration,
            "speaker": self.speaker,
        }


@dataclass
class DiarizationResult:
    segments: List[DiarizationSegment]
    num_speakers: int
    audio_duration: float
    processing_time: float
    sample_rate: int

    def to_dict(self) -> dict:
        speakers = sorted(set(s.speaker for s in self.segments))
        return {
            "num_speakers": self.num_speakers,
            "audio_duration": round(self.audio_duration, 3),
            "processing_time": round(self.processing_time, 3),
            "sample_rate": self.sample_rate,
            "speakers": speakers,
            "segments": [s.to_dict() for s in self.segments],
        }


class DiarizationPipeline:
    """End-to-end speaker diarization with pyannote-first fallback behavior."""

    SAMPLE_RATE = 16000
    WINDOW_DURATION = 2.0
    WINDOW_STEP = 1.0
    MIN_SEGMENT_DURATION = 0.8

    def __init__(
        self,
        device: str = "auto",
        use_pyannote_vad: bool = True,
        use_pyannote_diarization: bool = True,
        pyannote_diarization_model: str = "pyannote/speaker-diarization-3.1",
        hf_token: Optional[str] = None,
        num_speakers: Optional[int] = None,
        max_speakers: int = 10,
        cache_dir: str = "./model_cache",
    ):
        self.device = self._resolve_device(device)
        self.use_pyannote_vad = use_pyannote_vad
        self.use_pyannote_diarization = use_pyannote_diarization
        self.pyannote_diarization_model = pyannote_diarization_model
        self.hf_token = hf_token
        self.num_speakers = num_speakers
        self.max_speakers = max_speakers
        self.cache_dir = Path(cache_dir)

        self.embedder = EcapaTDNNEmbedder(device=self.device, cache_dir=str(cache_dir))
        self.clusterer = SpeakerClusterer(max_speakers=max_speakers, distance_threshold=0.55)

        self._vad_pipeline = None
        self._full_diar_pipeline = None
        logger.info(f"DiarizationPipeline ready | device={self.device}")

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _to_mono_1d(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 1:
            return audio
        if audio.dim() >= 2:
            if audio.shape[0] == 1:
                return audio[0]
            return audio.mean(dim=0)
        return audio.reshape(-1)

    def _load_pyannote_pipeline(self, model_id: str):
        from pyannote.audio import Pipeline

        try:
            if self.hf_token:
                try:
                    pipeline = Pipeline.from_pretrained(model_id, use_auth_token=self.hf_token)
                except TypeError:
                    pipeline = Pipeline.from_pretrained(model_id, token=self.hf_token)
            else:
                pipeline = Pipeline.from_pretrained(model_id)
        except TypeError:
            pipeline = Pipeline.from_pretrained(model_id)

        if pipeline is None:
            raise RuntimeError(f"Pipeline.from_pretrained returned None for {model_id}")

        try:
            pipeline.to(torch.device(self.device))
        except Exception:
            pass

        return pipeline

    def _load_full_diarization(self):
        if self._full_diar_pipeline is not None:
            return
        try:
            logger.info(f"Loading pyannote diarization pipeline: {self.pyannote_diarization_model}")
            self._full_diar_pipeline = self._load_pyannote_pipeline(self.pyannote_diarization_model)
            logger.success("Pyannote speaker diarization pipeline loaded.")
        except Exception as e:
            logger.warning(f"Could not load pyannote diarization pipeline: {e}.")
            self._full_diar_pipeline = "unavailable"

    def _load_vad(self):
        if self._vad_pipeline is not None:
            return
        try:
            logger.info("Loading pyannote VAD pipeline...")
            self._vad_pipeline = self._load_pyannote_pipeline("pyannote/voice-activity-detection")
            logger.success("Pyannote VAD loaded.")
        except Exception as e:
            logger.warning(f"Could not load pyannote VAD: {e}. Falling back to energy-based VAD.")
            self._vad_pipeline = "energy"

    def _merge_named_segments(
        self, segments: List[DiarizationSegment], gap_tolerance: float = 0.35
    ) -> List[DiarizationSegment]:
        if not segments:
            return []

        merged = [segments[0]]
        for seg in segments[1:]:
            last = merged[-1]
            if seg.speaker == last.speaker and seg.start - last.end <= gap_tolerance:
                merged[-1] = DiarizationSegment(start=last.start, end=seg.end, speaker=last.speaker)
            else:
                merged.append(seg)
        return merged

    def _run_full_pyannote(
        self,
        audio: Union[str, Path, torch.Tensor],
        sample_rate: int,
        num_speakers: Optional[int],
        audio_duration: float,
        t_start: float,
    ) -> Optional[DiarizationResult]:
        if not self.use_pyannote_diarization:
            return None

        self._load_full_diarization()
        if self._full_diar_pipeline == "unavailable":
            return None

        tmp_path = None
        source = audio
        try:
            if not isinstance(audio, (str, Path)):
                mono = self._to_mono_1d(audio).detach().cpu().float()
                wav = mono.unsqueeze(0)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                torchaudio.save(tmp_path, wav, sample_rate)
                source = tmp_path

            kwargs = {}
            if num_speakers is not None:
                kwargs["num_speakers"] = int(num_speakers)

            diar_output = self._full_diar_pipeline(str(source), **kwargs)

            raw_segments = []
            speaker_map = {}
            next_id = 0
            for turn, _, speaker in diar_output.itertracks(yield_label=True):
                start = float(turn.start)
                end = float(turn.end)
                if end - start < 0.2:
                    continue
                if speaker not in speaker_map:
                    speaker_map[speaker] = f"SPEAKER_{next_id:02d}"
                    next_id += 1
                raw_segments.append(
                    DiarizationSegment(start=start, end=end, speaker=speaker_map[speaker])
                )

            if not raw_segments:
                return None

            raw_segments.sort(key=lambda s: (s.start, s.end))
            merged_segments = self._merge_named_segments(raw_segments)
            num_unique = len(set(s.speaker for s in merged_segments))

            logger.success(
                f"Pyannote diarization complete: {num_unique} speakers, {len(merged_segments)} segments"
            )
            return DiarizationResult(
                segments=merged_segments,
                num_speakers=num_unique,
                audio_duration=audio_duration,
                processing_time=time.time() - t_start,
                sample_rate=sample_rate,
            )
        except Exception as e:
            logger.warning(f"Full pyannote diarization failed: {e}. Falling back to ECAPA+AHC.")
            return None
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    def _energy_vad(
        self, audio: torch.Tensor, frame_duration: float = 0.02, threshold_db: float = -40.0
    ) -> List[tuple]:
        frame_samples = int(frame_duration * self.SAMPLE_RATE)
        audio_np = audio.numpy()
        frames = [
            audio_np[i: i + frame_samples]
            for i in range(0, len(audio_np) - frame_samples, frame_samples)
        ]

        energies_db = []
        for frame in frames:
            rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
            energies_db.append(20 * np.log10(rms))

        is_speech = np.array(energies_db) > threshold_db

        speech_regions = []
        in_speech = False
        start = 0.0

        for i, active in enumerate(is_speech):
            t = i * frame_duration
            if active and not in_speech:
                start = t
                in_speech = True
            elif not active and in_speech:
                speech_regions.append((start, t))
                in_speech = False

        if in_speech:
            speech_regions.append((start, len(audio_np) / self.SAMPLE_RATE))

        return speech_regions

    def _get_speech_regions(self, audio: torch.Tensor) -> List[tuple]:
        if self.use_pyannote_vad:
            self._load_vad()

        if self._vad_pipeline == "energy" or not self.use_pyannote_vad:
            return self._energy_vad(audio)

        try:
            audio_dict = {
                "waveform": audio.unsqueeze(0).to(self.device),
                "sample_rate": self.SAMPLE_RATE,
            }
            vad_output = self._vad_pipeline(audio_dict)
            regions = [(seg.start, seg.end) for seg in vad_output.get_timeline().support()]
            logger.info(f"Pyannote VAD: {len(regions)} speech regions found")
            return regions
        except Exception as e:
            logger.warning(f"Pyannote VAD failed: {e}. Using energy VAD.")
            return self._energy_vad(audio)

    def _sliding_window_segments(self, speech_regions: List[tuple]) -> List[tuple]:
        segments = []
        for region_start, region_end in speech_regions:
            duration = region_end - region_start
            if duration < self.MIN_SEGMENT_DURATION:
                continue

            t = region_start
            while t + self.WINDOW_DURATION <= region_end:
                segments.append((t, t + self.WINDOW_DURATION))
                t += self.WINDOW_STEP

            if region_end - t >= self.MIN_SEGMENT_DURATION:
                segments.append((t, region_end))

        return segments

    def load_audio(self, path: Union[str, Path, BinaryIO]) -> tuple:
        waveform, sample_rate = torchaudio.load(path)
        return waveform, sample_rate

    def process(
        self,
        audio: Union[str, Path, torch.Tensor],
        sample_rate: int = None,
        num_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        t_start = time.time()

        if isinstance(audio, (str, Path)):
            waveform, sample_rate = self.load_audio(audio)
            audio_tensor = self._to_mono_1d(waveform)
        else:
            assert sample_rate is not None, "sample_rate required when passing tensor"
            audio_tensor = self._to_mono_1d(audio)

        num_samples = int(audio_tensor.numel())
        audio_duration = num_samples / float(sample_rate)
        logger.info(f"Processing {audio_duration:.1f}s audio at {sample_rate}Hz")

        if num_samples == 0:
            logger.warning("Received empty audio input.")
            return DiarizationResult(
                segments=[],
                num_speakers=0,
                audio_duration=0.0,
                processing_time=time.time() - t_start,
                sample_rate=sample_rate,
            )

        k = num_speakers or self.num_speakers

        pyannote_result = self._run_full_pyannote(
            audio=audio,
            sample_rate=sample_rate,
            num_speakers=k,
            audio_duration=audio_duration,
            t_start=t_start,
        )
        if pyannote_result is not None:
            return pyannote_result

        processed = self.embedder.preprocess_audio(audio_tensor, sample_rate)

        speech_regions = self._get_speech_regions(processed)
        if not speech_regions:
            logger.warning("No speech detected in audio.")
            return DiarizationResult(
                segments=[],
                num_speakers=0,
                audio_duration=audio_duration,
                processing_time=time.time() - t_start,
                sample_rate=sample_rate,
            )

        windows = self._sliding_window_segments(speech_regions)
        logger.info(f"Generated {len(windows)} embedding windows")

        embeddings, valid_windows = self.embedder.extract_embeddings_from_segments(
            processed, self.SAMPLE_RATE, windows
        )

        if len(embeddings) == 0:
            logger.warning("No valid embeddings extracted.")
            return DiarizationResult(
                segments=[],
                num_speakers=0,
                audio_duration=audio_duration,
                processing_time=time.time() - t_start,
                sample_rate=sample_rate,
            )

        labels = self.clusterer.cluster(embeddings, num_speakers=k)
        merged = self.clusterer.merge_consecutive_same_speaker(
            valid_windows, labels, gap_tolerance=0.45
        )

        speaker_names = {i: f"SPEAKER_{i:02d}" for i in range(self.max_speakers)}
        segments = [
            DiarizationSegment(start=start, end=end, speaker=speaker_names[spk_id])
            for start, end, spk_id in merged
        ]

        num_unique = len(set(labels))
        processing_time = time.time() - t_start

        logger.success(
            f"Fallback diarization complete: {num_unique} speakers, "
            f"{len(segments)} segments, {processing_time:.2f}s"
        )

        return DiarizationResult(
            segments=segments,
            num_speakers=num_unique,
            audio_duration=audio_duration,
            processing_time=processing_time,
            sample_rate=sample_rate,
        )
