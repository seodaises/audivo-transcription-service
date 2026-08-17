"""
Audivo Transcription Service — standalone FastAPI microservice.

Single responsibility: accept an uploaded audio file, run it through
Demucs (vocal isolation) then faster-whisper (transcription), and return
raw text + word-level timestamps. It does NOT decide how those words get
grouped into karaoke-style lines — that's a business-logic/presentation
decision that lives in audivoBackend's lyricsService, so this service can
stay a narrow, swappable engine.

Multipart file upload (not a shared filesystem path): this service is
expected to eventually move to its own machine/VM, and a Node backend on
a different host has no way to hand over a local file path. Sending the
actual bytes over HTTP means this contract doesn't need to change later.

Whisper is loaded ONCE at startup (see load_models), not per-request —
model load is the expensive part, and this is CPU-only hardware with no
GPU, so paying that cost on every call would be far too slow.
"""

import logging
import os
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from transcription import TranscriptionEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audivo-transcription")

app = FastAPI(title="Audivo Transcription Service")

# Temp uploads and Demucs output both land here instead of the OS default
# temp dir. On Windows that default is normally on C:, which is often the
# smaller/more crowded drive — pointing at a folder on this project's own
# drive avoids failures like "No space left on device" when C: fills up
# with unrelated Windows/app data. Created once at import time; safe to
# reuse across requests since files inside are cleaned up per-request.
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Populated once at startup by load_models(). None until then — /transcribe
# checks for this and returns 503 rather than crashing if a request arrives
# before the model finishes loading.
engine: TranscriptionEngine | None = None


@app.on_event("startup")
def load_models():
    global engine
    logger.info("Loading Whisper model (once, at startup)...")
    engine = TranscriptionEngine()
    logger.info("Model loaded. Service ready.")


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": engine is not None}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    # Demucs and faster-whisper both need a real path on disk, not an
    # in-memory stream, so persist the upload to a temp file first.
    suffix = os.path.splitext(file.filename)[1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TEMP_DIR) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = engine.transcribe(tmp_path)
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        # Clean up the uploaded temp file regardless of outcome — this is a
        # laptop with limited disk, not a server we can let fill up.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)