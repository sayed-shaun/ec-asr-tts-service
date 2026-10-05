"""Save each ASR request's audio and transcript to disk for later debugging.

A bad transcript is only debuggable with the audio that produced it, and by
the time someone reports one the clip is gone. So every request through the
gateway leaves a folder behind:

    TRACE_DIR/2026-10-05/143012-আমার_ফি_কত/
        audio_0.wav        the clip exactly as the caller sent it
        transcript.json    route, status, timing, config and LitServe's reply

The gateway rather than the model worker, because it is the one process every
request passes through and it already holds both halves: the bytes the client
sent and the response that went back. Failures are traced too -- a 502 or a
422 from the worker is exactly the case someone will want to replay.

Audio is stored as received, not decoded or resampled, so a trace replays
the exact input; the extension is sniffed from the magic bytes only so the
file opens in a player. Stdlib only: the gateway image has no librosa or
numpy (see tests/test_api.py's import-graph test).

Tracing must never cost a caller their transcript, so a write that fails is
logged and dropped, and the write runs on a thread so a slow disk cannot
stall the event loop. Day folders older than TRACE_RETENTION_DAYS are pruned
when a new day starts, which bounds disk use without a cron job.
"""

import asyncio
import json
import shutil
import unicodedata
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from src.core.config import settings

_MAGIC = (
    (b"RIFF", "wav"),
    (b"fLaC", "flac"),
    (b"OggS", "ogg"),
    (b"ID3", "mp3"),
    (b"\xff\xfb", "mp3"),
    (b"\xff\xf3", "mp3"),
    (b"\xff\xf2", "mp3"),
    (b"\x1a\x45\xdf\xa3", "webm"),
)

_MAX_NAME_CHARS = 60
"""Bengali is 3 bytes a character in UTF-8, so 60 stays well under the
255-byte filename limit with the time prefix and a collision suffix."""

_pruned_for: date | None = None


def _extension(audio: bytes) -> str:
    for magic, ext in _MAGIC:
        if audio.startswith(magic):
            return ext
    if audio[4:8] == b"ftyp":
        return "m4a"
    return "bin"


def _name_from(meta: dict[str, Any]) -> str:
    """The first transcribed sentence, made safe to use as a folder name.

    Naming the folder after what the caller said lets someone find "the one
    where it misheard the fee" by browsing, rather than opening every
    transcript.json. Falls back to the status when there is no transcript.
    """
    output = (meta.get("response") or {}).get("output") or []
    text = output[0].get("source", "") if output else ""
    # Letters, combining marks and digits -- not isalnum(), which drops the
    # Bengali vowel signs and virama and would split every word apart.
    name = "".join(
        ch if unicodedata.category(ch)[0] in "LMN" or ch in "-_" else " "
        for ch in text
    )
    name = "_".join(name.split())[:_MAX_NAME_CHARS].rstrip("_")
    if name:
        return name
    status_code = meta.get("status_code")
    return "error" if status_code and status_code >= 400 else "empty"


def _prune(root: Path, today: date) -> None:
    """Drop day folders past retention, once per day per process."""
    global _pruned_for
    if _pruned_for == today:
        return
    _pruned_for = today
    if settings.TRACE_RETENTION_DAYS <= 0:
        return
    cutoff = today - timedelta(days=settings.TRACE_RETENTION_DAYS)
    for day_dir in root.iterdir():
        try:
            day = date.fromisoformat(day_dir.name)
        except ValueError:
            continue
        if day_dir.is_dir() and day < cutoff:
            shutil.rmtree(day_dir, ignore_errors=True)


def _write(
    route: str,
    clips: list[bytes],
    meta: dict[str, Any],
    started: datetime,
) -> Path:
    root = Path(settings.TRACE_DIR)
    root.mkdir(parents=True, exist_ok=True)
    _prune(root, started.date())

    trace_id = uuid.uuid4().hex[:8]
    day_dir = root / started.date().isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{started:%H%M%S}-{_name_from(meta)}"
    trace_dir = day_dir / stem
    suffix = 2
    while True:
        try:
            trace_dir.mkdir()
            break
        except FileExistsError:
            trace_dir = day_dir / f"{stem}-{suffix}"
            suffix += 1

    files = []
    for index, clip in enumerate(clips):
        name = f"audio_{index}.{_extension(clip)}"
        (trace_dir / name).write_bytes(clip)
        files.append(name)

    record = {
        "trace_id": trace_id,
        "route": route,
        "received_at": started.isoformat(timespec="milliseconds"),
        "audio_files": files,
        **meta,
    }
    (trace_dir / "transcript.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return trace_dir


async def save_trace(
    route: str,
    clips: list[bytes],
    meta: dict[str, Any],
    started: datetime,
) -> None:
    """Write one request's trace folder; log and swallow any failure."""
    if not settings.TRACE_ENABLED:
        return
    try:
        trace_dir = await asyncio.to_thread(_write, route, clips, meta, started)
        logger.debug(f"ASR trace saved to {trace_dir}")
    except Exception as exc:  # noqa: BLE001 -- a trace must never fail a request
        logger.warning(f"Could not save ASR trace: {exc!r}")
