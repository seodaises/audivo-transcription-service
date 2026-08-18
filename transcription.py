"""
TranscriptionEngine, the actual Demucs -> Whisper pipeline.

Kept separate from main.py so the FastAPI route stays a thin
request/response layer and this file holds only the ML pipeline logic —
mirrors the controller/service split used across the rest of Audivo.
"""

import logging
import os
import shutil
import tempfile

import demucs.separate
from faster_whisper import WhisperModel

logger = logging.getLogger("audivo-transcription")

# "small" (not "medium") chosen deliberately for this hardware: 4-core CPU,
# no dedicated GPU. int8 compute type is the fastest viable precision
# without a GPU — accuracy tradeoff is acceptable for a first pass that the
# artist can still hand-correct afterward.
WHISPER_MODEL_SIZE = "small"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# Same reasoning as main.py's TEMP_DIR: keep Demucs' output on this
# project's own drive rather than the OS default temp dir, which is
# frequently on a smaller/more crowded C: drive on Windows.
DEMUCS_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
os.makedirs(DEMUCS_OUTPUT_DIR, exist_ok=True)


class TranscriptionEngine:
    """
    Loads Whisper once at construction (i.e. once at FastAPI startup) and
    reuses it for every request. Demucs has no equivalent persistent
    in-process handle — it's invoked per-request via its own entrypoint —
    but that per-request cost is unavoidable either way; what matters is
    that the expensive Whisper model load is NOT repeated per request.
    """

    def __init__(self):
        self.whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )

    def _isolate_vocals(self, audio_path: str) -> str:
        """
        Runs Demucs on audio_path and returns the path to the isolated
        vocals stem. Demucs writes into its own {model}/{track}/ folder
        structure inside out_dir, so we search for the vocals.wav it
        produces rather than assuming a fixed path.
        """
        out_dir = tempfile.mkdtemp(prefix="audivo-demucs-", dir=DEMUCS_OUTPUT_DIR)
        demucs.separate.main(
            [
                "--two-stems",
                "vocals",
                "-o",
                out_dir,
                audio_path,
            ]
        )

        track_name = os.path.splitext(os.path.basename(audio_path))[0]
        for model_dir in os.listdir(out_dir):
            candidate = os.path.join(out_dir, model_dir, track_name, "vocals.wav")
            if os.path.exists(candidate):
                return candidate

        raise RuntimeError("Demucs did not produce a vocals stem — check its output layout.")

    def transcribe(self, audio_path: str) -> dict:
        """
        Returns raw text and word-level timestamps only. Grouping words
        into karaoke-style lines is intentionally NOT done here — that's
        audivoBackend's lyricsService's job (business/presentation logic,
        tunable without touching this service or reloading the model).
        """
        vocals_path = self._isolate_vocals(audio_path)
        # The model-name dir one level up from vocals.wav — safe to remove
        # wholesale once we're done reading from it.
        vocals_dir = os.path.dirname(os.path.dirname(vocals_path))

        try:
            segments, info = self.whisper_model.transcribe(
                vocals_path,
                word_timestamps=True,
                vad_filter=True,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
            )

            words = []
            raw_text_parts = []
            for segment in segments:
                raw_text_parts.append(segment.text.strip())
                if segment.words:
                    for w in segment.words:
                        words.append(
                            {
                                "word": w.word.strip(),
                                "start": round(w.start, 2),
                                "end": round(w.end, 2),
                            }
                        )

            return {
                "raw_text": " ".join(raw_text_parts).strip(),
                "words": words,
                "language": info.language,
            }
        finally:
            # Demucs' temp output has served its purpose — this laptop has
            # no disk to spare for leftover audio stems.
            if os.path.exists(vocals_dir):
                shutil.rmtree(vocals_dir, ignore_errors=True)