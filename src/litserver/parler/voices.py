"""The voice table, kept free of numpy/torch imports on purpose.

The gateway image installs only the base dependencies, so the TTS router
reads VOICES from here rather than from parler.py.
"""

VOICES: dict[str, str] = {
    "Aditi": (
        "Aditi speaks in a warm, clear female voice with natural intonation at a "
        "moderate, measured pace. Very high quality audio, no background noise."
    ),
    "Rashmi": (
        "Rashmi speaks in a gentle, expressive female voice with smooth intonation "
        "at a moderate pace. Very high quality audio, no background noise."
    ),
    "Riya": (
        "Riya speaks in a low, mellow female voice with a soft warm timbre, measured "
        "and composed. Very high quality audio, no background noise."
    ),
    "Arjun": (
        "Arjun speaks in a clear, steady male voice at a moderate pace. Very high "
        "quality audio, no background noise."
    ),
}

DEFAULT_VOICE = "Aditi"
