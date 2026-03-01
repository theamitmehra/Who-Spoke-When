"""Audio utility functions for the diarization pipeline."""

import io
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Union, Tuple, Iterator
from loguru import logger

SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
TARGET_SAMPLE_RATE = 16000


def load_audio(source, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[torch.Tensor, int]:
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    waveform, sr = torchaudio.load(source)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    return waveform.squeeze(0), sr


def pcm_bytes_to_tensor(data: bytes, dtype=np.float32) -> torch.Tensor:
    arr = np.frombuffer(data, dtype=dtype).copy()
    return torch.from_numpy(arr)


def chunk_audio(audio, sample_rate, chunk_duration=30.0, overlap=1.0):
    chunk_samples = int(chunk_duration * sample_rate)
    step_samples = int((chunk_duration - overlap) * sample_rate)
    n = len(audio)
    for start in range(0, n, step_samples):
        end = min(start + chunk_samples, n)
        yield audio[start:end], start / sample_rate
        if end == n:
            break


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def segments_to_rttm(segments, audio_name: str = "audio") -> str:
    lines = []
    for seg in segments:
        duration = seg.end - seg.start
        lines.append(
            f"SPEAKER {audio_name} 1 {seg.start:.3f} {duration:.3f} "
            f"<NA> <NA> {seg.speaker} <NA> <NA>"
        )
    return "\n".join(lines)


def segments_to_srt(segments) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg.start).replace(".", ",")
        end = format_timestamp(seg.end).replace(".", ",")
        lines.append(f"{i}\n{start} --> {end}\n[{seg.speaker}]\n")
    return "\n".join(lines)
