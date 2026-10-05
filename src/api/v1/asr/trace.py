"""Save each ASR conversation's audio and transcripts to disk for debugging.

A bad transcript is only debuggable with the audio that produced it, and by
the time someone reports one the clip is gone. So the gateway keeps, per day,
one JSON per conversation with every turn in it, and the clips beside it:

    TRACE_DIR/2026-10-05/
        143012-আমার_ফি_কত.json         the whole conversation, turn by turn
        143012-আমার_ফি_কত_t1_0.wav     turn 1's clip, exactly as sent
        143012-আমার_ফি_কত_t2_0.wav     turn 2's clip

The service itself has no notion of a session: every ASR call is independent.
A conversation is whatever the caller says it is, through an optional
`X-Conversation-Id` header. Requests sharing an id append turns to one JSON;
a request without the header is a one-turn conversation of its own. The file
is named from the first transcribed sentence of the conversation's first turn
and the name never changes afterwards, so a later turn cannot move the file.

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

A conversation stays in the day folder it started in, even when it runs past
midnight. Its id -> file mapping is remembered in memory; after a restart it
is recovered by scanning today's JSONs for the id.
"""

import asyncio
import json
import shutil
import threading
import unicodedata
import uuid
from collections import OrderedDict
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
    the conversation's JSON. Falls back to the status when there is no transcript.
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


_MAX_TRACKED = 1000
_lock = threading.Lock()
_open: OrderedDict[str, Path] = OrderedDict()
"""conversation id -> its JSON, for conversations seen by this process.

Bounded because ids come from callers; the coldest is forgotten first and is
simply re-found by scanning on its next turn. Guarded by _lock together with
the read-modify-write of the JSON: _write runs on worker threads, and two
turns of one conversation landing at once would otherwise lose one.
"""


def _find_on_disk(day_dir: Path, conversation_id: str) -> Path | None:
    for path in sorted(day_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if record.get("conversation_id") == conversation_id:
            return path
    return None


def _create(day_dir: Path, started: datetime, meta: dict[str, Any]) -> Path:
    """Claim a fresh JSON with exclusive create so no two conversations share
    a stem, even in the same second with the same first sentence."""
    base = f"{started:%H%M%S}-{_name_from(meta)}"
    stem, suffix = base, 2
    while True:
        path = day_dir / f"{stem}.json"
        try:
            path.open("x").close()
            return path
        except FileExistsError:
            stem = f"{base}-{suffix}"
            suffix += 1


def _append_turn(
    json_path: Path,
    record: dict[str, Any],
    route: str,
    clips: list[bytes],
    meta: dict[str, Any],
    started: datetime,
) -> None:
    """Write this turn's clips, then atomically replace the conversation JSON."""
    turn = len(record["turns"]) + 1
    files = []
    for index, clip in enumerate(clips):
        name = f"{json_path.stem}_t{turn}_{index}.{_extension(clip)}"
        (json_path.parent / name).write_bytes(clip)
        files.append(name)

    record["turns"].append(
        {
            "turn": turn,
            "route": route,
            "received_at": started.isoformat(timespec="milliseconds"),
            "audio_files": files,
            **meta,
        }
    )
    record["updated_at"] = started.isoformat(timespec="milliseconds")
    tmp = json_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(json_path)


def _write(
    route: str,
    clips: list[bytes],
    meta: dict[str, Any],
    started: datetime,
    conversation_id: str | None = None,
) -> Path:
    root = Path(settings.TRACE_DIR)
    root.mkdir(parents=True, exist_ok=True)
    _prune(root, started.date())
    day_dir = root / started.date().isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)

    with _lock:
        json_path = _open.get(conversation_id) if conversation_id else None
        if json_path is None and conversation_id:
            json_path = _find_on_disk(day_dir, conversation_id)
        record: dict[str, Any] | None = None
        if json_path is not None:
            try:
                record = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                record = None
        created = record is None
        if created:
            json_path = _create(day_dir, started, meta)
            record = {
                "conversation_id": conversation_id or uuid.uuid4().hex[:8],
                "started_at": started.isoformat(timespec="milliseconds"),
                "turns": [],
            }
        try:
            _append_turn(json_path, record, route, clips, meta, started)
        except BaseException:
            # _create left an empty file claiming the name; a later turn
            # would find it and fail to parse it, so do not leave it behind.
            if created:
                json_path.unlink(missing_ok=True)
            raise
        if conversation_id:
            _open[conversation_id] = json_path
            _open.move_to_end(conversation_id)
            while len(_open) > _MAX_TRACKED:
                _open.popitem(last=False)
    return json_path


async def save_trace(
    route: str,
    clips: list[bytes],
    meta: dict[str, Any],
    started: datetime,
    conversation_id: str | None = None,
) -> None:
    """Append one request to its conversation's JSON; log and swallow failures."""
    if not settings.TRACE_ENABLED:
        return
    try:
        trace_path = await asyncio.to_thread(
            _write, route, clips, meta, started, conversation_id
        )
        logger.debug(f"ASR trace saved to {trace_path}")
    except Exception as exc:  # noqa: BLE001 -- a trace must never fail a request
        logger.warning(f"Could not save ASR trace: {exc!r}")
