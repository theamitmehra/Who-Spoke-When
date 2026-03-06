"""
Speaker Embedding Extraction using ECAPA-TDNN architecture via SpeechBrain.
Handles audio preprocessing, feature extraction, and L2-normalized embeddings.
"""

import inspect
from pathlib import Path
from typing import Union, List, Tuple

import numpy as np
import torch
import torchaudio
from loguru import logger


class EcapaTDNNEmbedder:
    """
    Speaker embedding extractor using ECAPA-TDNN architecture.
    Produces 192-dim L2-normalized speaker embeddings per audio segment.
    """

    MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
    SAMPLE_RATE = 16000
    EMBEDDING_DIM = 192

    def __init__(self, device: str = "auto", cache_dir: str = "./model_cache"):
        self.device = self._resolve_device(device)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        logger.info(f"EcapaTDNNEmbedder initialized on device: {self.device}")

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _build_hparams_kwargs(self, encoder_cls, savedir: Path, hf_cache: Path) -> dict:
        kwargs = {
            "source": self.MODEL_SOURCE,
            "savedir": str(savedir),
            "run_opts": {"device": self.device},
        }

        sig = inspect.signature(encoder_cls.from_hparams)
        if "huggingface_cache_dir" in sig.parameters:
            kwargs["huggingface_cache_dir"] = str(hf_cache)
        if "local_strategy" in sig.parameters:
            try:
                from speechbrain.utils.fetching import LocalStrategy

                kwargs["local_strategy"] = LocalStrategy.COPY
            except Exception:
                pass

        return kwargs

    def _load_model(self):
        if self._model is not None:
            return

        try:
            try:
                from speechbrain.inference.classifiers import EncoderClassifier
            except ImportError:
                # Backward compatibility with older SpeechBrain versions.
                from speechbrain.pretrained import EncoderClassifier

            savedir = self.cache_dir / "ecapa_tdnn"
            hf_cache = self.cache_dir / "hf_cache"
            savedir.mkdir(parents=True, exist_ok=True)
            hf_cache.mkdir(parents=True, exist_ok=True)

            logger.info(f"Loading ECAPA-TDNN from {self.MODEL_SOURCE}...")
            logger.info(f"Savedir: {savedir}, exists: {savedir.exists()}")

            kwargs = self._build_hparams_kwargs(EncoderClassifier, savedir, hf_cache)
            model = EncoderClassifier.from_hparams(**kwargs)

            if model is None:
                # Some SpeechBrain/HF hub combinations ignore optional kwargs.
                logger.warning("ECAPA load returned None; retrying with minimal from_hparams kwargs.")
                model = EncoderClassifier.from_hparams(
                    source=self.MODEL_SOURCE,
                    savedir=str(savedir),
                    run_opts={"device": self.device},
                )

            if model is None:
                raise RuntimeError("EncoderClassifier.from_hparams returned None")

            self._model = model
            self._model.eval()
            logger.success("ECAPA-TDNN model loaded successfully.")
        except ImportError as exc:
            raise ImportError("SpeechBrain not installed.") from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load ECAPA-TDNN model: {exc}") from exc

    def preprocess_audio(
        self, audio: Union[np.ndarray, torch.Tensor], sample_rate: int
    ) -> torch.Tensor:
        """Resample and normalize audio to 16kHz mono float32 tensor."""
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        if sample_rate != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.SAMPLE_RATE
            )
            audio = resampler(audio)

        max_val = audio.abs().max()
        if max_val > 0:
            audio = audio / max_val

        return audio.squeeze(0)

    def extract_embedding(self, audio: torch.Tensor) -> np.ndarray:
        """
        Extract L2-normalized ECAPA-TDNN embedding from a preprocessed audio tensor.
        Returns L2-normalized embedding of shape (192,)
        """
        self._load_model()

        with torch.no_grad():
            audio_batch = audio.unsqueeze(0).to(self.device)
            lengths = torch.tensor([1.0]).to(self.device)
            embedding = self._model.encode_batch(audio_batch, lengths)
            embedding = embedding.squeeze().cpu().numpy()

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def extract_embeddings_from_segments(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        segments: List[Tuple[float, float]],
        min_duration: float = 0.5,
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """Extract embeddings for a list of (start, end) time segments."""
        processed = self.preprocess_audio(audio, sample_rate)
        embeddings = []
        valid_segments = []

        for start, end in segments:
            duration = end - start
            if duration < min_duration:
                continue

            start_sample = int(start * self.SAMPLE_RATE)
            end_sample = int(end * self.SAMPLE_RATE)
            segment_audio = processed[start_sample:end_sample]

            if segment_audio.shape[0] == 0:
                continue

            emb = self.extract_embedding(segment_audio)
            embeddings.append(emb)
            valid_segments.append((start, end))

        if not embeddings:
            return np.empty((0, self.EMBEDDING_DIM)), []

        return np.stack(embeddings), valid_segments

