# 🎙 Speaker Diarization System
### *Who Spoke When — Multi-Speaker Audio Segmentation*

> **Tech Stack:** Python · PyTorch · SpeechBrain · Pyannote.audio · Transformers · FastAPI

---

## Architecture

```
Audio Input
    │
    ▼
┌─────────────────────────────┐
│  Voice Activity Detection   │  ← pyannote/voice-activity-detection
│  (VAD)                      │    fallback: energy-based VAD
└────────────┬────────────────┘
             │  speech regions (start, end)
             ▼
┌─────────────────────────────┐
│  Sliding Window Segmentation│  ← 1.5s windows, 50% overlap
│                             │
└────────────┬────────────────┘
             │  segment list
             ▼
┌─────────────────────────────┐
│  ECAPA-TDNN Embedding       │  ← speechbrain/spkrec-ecapa-voxceleb
│  Extraction                 │    192-dim L2-normalized vectors
└────────────┬────────────────┘
             │  embeddings (N × 192)
             ▼
┌─────────────────────────────┐
│  Agglomerative Hierarchical │  ← cosine distance metric
│  Clustering (AHC)           │    silhouette-based auto k-selection
└────────────┬────────────────┘
             │  speaker labels
             ▼
┌─────────────────────────────┐
│  Post-processing            │  ← merge consecutive same-speaker segs
│  & Output Formatting        │    timestamped JSON / RTTM / SRT
└─────────────────────────────┘
```

---

## Project Structure

```
speaker-diarization/
├── app/
│   ├── main.py          # FastAPI app — REST + WebSocket endpoints
│   └── pipeline.py      # Core end-to-end diarization pipeline
├── models/
│   ├── embedder.py      # ECAPA-TDNN speaker embedding extractor
│   └── clusterer.py     # Agglomerative Hierarchical Clustering (AHC)
├── utils/
│   └── audio.py         # Audio loading, chunking, RTTM/SRT export
├── tests/
│   └── test_diarization.py  # Unit + integration tests
├── static/
│   └── index.html       # Web demo UI
├── demo.py              # CLI interface
└── requirements.txt
```

---

## Installation

```bash
# 1. Clone / navigate to project
cd speaker-diarization

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set HuggingFace token for pyannote VAD
#    Accept terms at: https://huggingface.co/pyannote/voice-activity-detection
export HF_TOKEN=your_token_here
```

---

## Usage

### CLI Demo

```bash
# Basic usage (auto-detect speaker count)
python demo.py --audio meeting.wav

# Specify 3 speakers
python demo.py --audio call.wav --speakers 3

# Export all formats
python demo.py --audio audio.mp3 \
    --output result.json \
    --rttm output.rttm \
    --srt subtitles.srt
```

**Example output:**
```
✅ Done in 4.83s
   Speakers found : 3
   Audio duration : 120.50s
   Segments       : 42

   START       END       DUR  SPEAKER
   ────────────────────────────────────
   0.000     3.250    3.250  SPEAKER_00
   3.500     8.120    4.620  SPEAKER_01
   8.200    11.800    3.600  SPEAKER_00
   12.000   17.340    5.340  SPEAKER_02
   ...
```

### FastAPI Server

```bash
# Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Open the web UI
open http://localhost:8000

# Swagger documentation
open http://localhost:8000/docs
```

### REST API

**POST /diarize** — Upload audio file
```bash
curl -X POST http://localhost:8000/diarize \
  -F "file=@meeting.wav" \
  -F "num_speakers=3"
```

**Response:**
```json
{
  "status": "success",
  "num_speakers": 3,
  "audio_duration": 120.5,
  "processing_time": 4.83,
  "sample_rate": 16000,
  "speakers": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
  "segments": [
    { "start": 0.000, "end": 3.250, "duration": 3.250, "speaker": "SPEAKER_00" },
    { "start": 3.500, "end": 8.120, "duration": 4.620, "speaker": "SPEAKER_01" }
  ]
}
```

**GET /health** — Service health
```bash
curl http://localhost:8000/health
# {"status":"healthy","device":"cuda","version":"1.0.0"}
```

### WebSocket Streaming

```python
import asyncio, websockets, json, numpy as np

async def stream_audio():
    async with websockets.connect("ws://localhost:8000/ws/stream") as ws:
        # Send config
        await ws.send(json.dumps({"sample_rate": 16000, "num_speakers": 2}))
        
        # Send audio chunks (raw float32 PCM)
        with open("audio.raw", "rb") as f:
            while chunk := f.read(4096):
                await ws.send(chunk)
        
        # Signal end
        await ws.send(json.dumps({"type": "eof"}))
        
        # Receive results
        async for msg in ws:
            data = json.loads(msg)
            if data["type"] == "segment":
                print(f"[{data['data']['speaker']}] {data['data']['start']:.2f}s – {data['data']['end']:.2f}s")
            elif data["type"] == "done":
                break

asyncio.run(stream_audio())
```

---

## Key Design Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Speaker Embeddings | ECAPA-TDNN (SpeechBrain) | State-of-the-art speaker verification accuracy on VoxCeleb |
| Clustering | AHC + cosine distance | No predefined k required; works well with L2-normalized embeddings |
| k-selection | Silhouette analysis | Unsupervised, parameter-free speaker count estimation |
| VAD | pyannote (energy fallback) | pyannote VAD reduces false embeddings on silence/noise |
| Embedding window | 1.5s, 50% overlap | Balances temporal resolution vs. embedding stability |
| Post-processing | Merge consecutive same-speaker | Reduces over-segmentation artifact |

---

## Evaluation Metrics

Standard speaker diarization evaluation uses **Diarization Error Rate (DER)**:

```
DER = (Miss + False Alarm + Speaker Error) / Total Speech Duration
```

Export RTTM files for evaluation with `md-eval` or `dscore`:
```bash
python demo.py --audio test.wav --rttm hypothesis.rttm
dscore -r reference.rttm -s hypothesis.rttm
```

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v -k "clusterer"  # run specific test class
```

---

## Limitations & Future Work

- Long audio (>1hr) should use chunked processing (`utils.audio.chunk_audio`)
- Real-time streaming requires low-latency VAD (not yet implemented in WS endpoint)
- Speaker overlap (cross-talk) is assigned to a single speaker
- Consider fine-tuning ECAPA-TDNN on domain-specific data for call analytics
