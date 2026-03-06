---
title: Who Spoke When
emoji: '🎙️'
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app/main.py
pinned: false
---

# Who Spoke When
Speaker diarization service and web app: upload audio and get **who spoke when** segments.

The project now runs with a **hybrid pipeline**:
- Preferred: `pyannote/speaker-diarization-3.1` (best quality)
- Fallback: VAD + ECAPA-TDNN embeddings + agglomerative clustering

---

## What You Get
- FastAPI backend (`/diarize`, `/diarize/url`, `/health`)
- Web UI (`/`) for file upload and timeline view
- CLI demo (`demo.py`)
- Automatic fallback if pyannote models are unavailable

---

## Project Structure
```text
app/
  main.py         FastAPI app and endpoints
  pipeline.py     Hybrid diarization pipeline
models/
  embedder.py     ECAPA-TDNN embedding extractor
  clusterer.py    Speaker clustering logic
utils/
  audio.py        Audio and export helpers
static/
  index.html      Web UI
Dockerfile
requirements.txt
README.md
```

---

## Quick Start (Local)

### 1. Create and activate a virtual environment

Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Recommended) Set Hugging Face token
`pyannote` models are gated. Create a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

Windows PowerShell:
```powershell
$env:HF_TOKEN="your_token_here"
```

Linux/macOS:
```bash
export HF_TOKEN="your_token_here"
```

### 4. Run API server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open:
- UI: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

---

## Web UI Notes
- The UI now defaults to **same-origin** API (`/diarize`), so it works on Hugging Face Spaces.
- If you manually set a custom endpoint, ensure it allows CORS and is reachable from browser.

---

## Hugging Face Spaces Deployment

### Requirements
1. Space created (Docker SDK)
2. Space secret `HF_TOKEN` configured
3. Terms accepted for:
   - [https://huggingface.co/pyannote/voice-activity-detection](https://huggingface.co/pyannote/voice-activity-detection)
   - [https://huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

### Push code
Push `main` branch to your Space repo remote:
```bash
git push huggingface main
```

If push fails with unauthorized:
- Use a token with **Write** role (not Read)
- Confirm token owner has access to the target namespace

---

## API

### `GET /health`
Returns service health and device.

### `POST /diarize`
Upload an audio file.

Form fields:
- `file`: audio file
- `num_speakers` (optional): force known number of speakers

Example:
```bash
curl -X POST http://localhost:8000/diarize \
  -F "file=@meeting.mp3" \
  -F "num_speakers=2"
```

### `POST /diarize/url`
Diarize audio from a remote URL.

Example:
```bash
curl -X POST "http://localhost:8000/diarize/url?audio_url=https://example.com/sample.wav"
```

---

## CLI Usage
```bash
python demo.py --audio meeting.wav
python demo.py --audio meeting.wav --speakers 2
python demo.py --audio meeting.wav --output result.json --rttm result.rttm --srt result.srt
```

---

## Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | unset | Hugging Face token for gated pyannote models |
| `CACHE_DIR` | temp model cache path | Model download/cache directory |
| `USE_PYANNOTE_DIARIZATION` | `true` | Enable full pyannote diarization first |
| `PYANNOTE_DIARIZATION_MODEL` | `pyannote/speaker-diarization-3.1` | pyannote diarization model id |

---

## How the Pipeline Works
1. Load and normalize audio
2. Try full pyannote diarization (best quality)
3. If unavailable/fails, fallback to:
   - VAD (pyannote VAD or energy VAD)
   - Sliding windows
   - ECAPA embeddings
   - Agglomerative clustering
4. Merge adjacent same-speaker segments

---

## Troubleshooting

### 1) UI shows `Error: Failed to fetch`
Likely wrong API endpoint. Use same-origin `/diarize` in deployed UI.

### 2) Logs show pyannote download/auth warnings
You need:
- valid `HF_TOKEN`
- accepted model terms on both pyannote model pages

### 3) Poor speaker separation
- Provide `num_speakers` when known
- Ensure clean audio (minimal background noise)
- Prefer pyannote path (set token + accept terms)

### 4) `500` during embedding load
This is usually model download/cache/auth mismatch. Confirm `HF_TOKEN`, cache path write access, and internet connectivity.

---

## Limitations
- Overlapped speech may still be imperfect in fallback mode
- Quality depends on audio clarity, language mix, and noise
- Very short utterances are harder to classify reliably

---

## License
Add your preferred license file (`LICENSE`) if this project is public.

