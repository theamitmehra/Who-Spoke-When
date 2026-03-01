"""
Speaker Diarization API - FastAPI Application
"""

import io
import time
import asyncio
import tempfile
import traceback
from pathlib import Path
from typing import Optional, List
import os

import torch
from fastapi import (
    FastAPI, File, UploadFile, Form, WebSocket,
    WebSocketDisconnect, HTTPException, Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from loguru import logger

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SegmentOut(BaseModel):
    start: float
    end: float
    duration: float
    speaker: str


class DiarizationResponse(BaseModel):
    status: str = "success"
    num_speakers: int
    audio_duration: float
    processing_time: float
    sample_rate: int
    speakers: List[str]
    segments: List[SegmentOut]


class HealthResponse(BaseModel):
    status: str
    device: str
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Speaker Diarization API",
    description="Who Spoke When — Speaker diarization using ECAPA-TDNN + AHC Clustering",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from app.pipeline import DiarizationPipeline
        _pipeline = DiarizationPipeline(
            device="auto",
            use_pyannote_vad=True,
            hf_token=os.getenv("HF_TOKEN"),
            max_speakers=10,
            cache_dir="./model_cache",
        )
    return _pipeline


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HealthResponse(status="healthy", device=device)


@app.post("/diarize", response_model=DiarizationResponse, tags=["Diarization"])
async def diarize_audio(
    file: UploadFile = File(...),
    num_speakers: Optional[int] = Form(None, ge=1, le=20),
):
    """Diarize an uploaded audio file. Returns timestamped speaker labels."""
    allowed = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=415, detail=f"Unsupported format '{suffix}'")

    audio_bytes = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        pipeline = get_pipeline()
        result = pipeline.process(tmp_path, num_speakers=num_speakers)
        return DiarizationResponse(
            num_speakers=result.num_speakers,
            audio_duration=result.audio_duration,
            processing_time=result.processing_time,
            sample_rate=result.sample_rate,
            speakers=sorted(set(s.speaker for s in result.segments)),
            segments=[SegmentOut(**s.to_dict()) for s in result.segments],
        )
    except Exception as e:
        logger.error(f"Diarization failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/diarize/url", response_model=DiarizationResponse, tags=["Diarization"])
async def diarize_from_url(
    audio_url: str = Query(...),
    num_speakers: Optional[int] = Query(None, ge=1, le=20),
):
    """Diarize audio from a URL."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(audio_url)
            resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch audio: {e}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name

    try:
        pipeline = get_pipeline()
        result = pipeline.process(tmp_path, num_speakers=num_speakers)
        return DiarizationResponse(
            num_speakers=result.num_speakers,
            audio_duration=result.audio_duration,
            processing_time=result.processing_time,
            sample_rate=result.sample_rate,
            speakers=sorted(set(s.speaker for s in result.segments)),
            segments=[SegmentOut(**s.to_dict()) for s in result.segments],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.websocket("/ws/stream")
async def stream_diarization(websocket: WebSocket):
    """Real-time streaming diarization via WebSocket."""
    await websocket.accept()
    import numpy as np
    audio_buffer = bytearray()
    sample_rate = 16000
    num_speakers = None
    chunk_count = 0

    try:
        config_msg = await websocket.receive_json()
        sample_rate = config_msg.get("sample_rate", 16000)
        num_speakers = config_msg.get("num_speakers", None)

        await websocket.send_json({
            "type": "progress",
            "data": {"message": "Config received. Send audio chunks.", "chunks_received": 0},
        })

        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "error", "data": {"message": "Timeout"}})
                break

            if "bytes" in msg:
                audio_buffer.extend(msg["bytes"])
                chunk_count += 1
                await websocket.send_json({
                    "type": "progress",
                    "data": {"message": f"Received chunk {chunk_count}", "chunks_received": chunk_count},
                })
            elif "text" in msg:
                import json
                data = json.loads(msg["text"])
                if data.get("type") == "eof":
                    break

        if not audio_buffer:
            await websocket.send_json({"type": "error", "data": {"message": "No audio received"}})
            return

        import torch
        audio_np = np.frombuffer(audio_buffer, dtype=np.float32).copy()
        audio_tensor = torch.from_numpy(audio_np)

        await websocket.send_json({
            "type": "progress",
            "data": {"message": "Running diarization pipeline..."},
        })

        loop = asyncio.get_event_loop()
        pipeline = get_pipeline()
        result = await loop.run_in_executor(
            None,
            lambda: pipeline.process(audio_tensor, sample_rate=sample_rate, num_speakers=num_speakers),
        )

        for seg in result.segments:
            await websocket.send_json({"type": "segment", "data": seg.to_dict()})

        await websocket.send_json({
            "type": "done",
            "data": {
                "num_speakers": result.num_speakers,
                "total_segments": len(result.segments),
                "audio_duration": result.audio_duration,
                "processing_time": result.processing_time,
            },
        })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {traceback.format_exc()}")
        try:
            await websocket.send_json({"type": "error", "data": {"message": str(e)}})
        except Exception:
            pass


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    ui_path = Path("static/index.html")
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text())
    return HTMLResponse("<h1>Speaker Diarization API</h1><p><a href='/docs'>API Docs →</a></p>")


static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")
