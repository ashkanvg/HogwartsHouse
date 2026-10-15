"""The hat's voice, with a fake speech engine (the real model is 350 MB)."""
import io
import wave

import numpy as np
import pytest

import voice


class FakeEngine:
    """Stands in for Kokoro: half a second of a quiet hum."""

    def __init__(self):
        self.calls = 0

    def create(self, text, voice, speed, lang):
        self.calls += 1
        t = np.linspace(0, 0.5, 12000, endpoint=False)
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 24000


@pytest.fixture
def hat_voice(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "CACHE_DIR", tmp_path)
    return voice.HatVoice(engine=FakeEngine())


def test_a_line_becomes_a_slower_deeper_equally_loud_wav(hat_voice):
    with wave.open(io.BytesIO(hat_voice.wav("Hello"))) as f:
        assert f.getnchannels() == 1 and f.getsampwidth() == 2
        assert f.getframerate() == int(24000 * voice.DEEPER)  # played slower than the voice's own 24 kHz
        samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.int32)
    assert np.abs(samples).max() == pytest.approx(0.9 * 32767, rel=0.01)


def test_each_line_is_only_generated_once(hat_voice):
    first = hat_voice.wav("Hello")
    assert hat_voice.wav("Hello") == first
    assert hat_voice.engine.calls == 1


def test_unknown_voices_are_refused(hat_voice):
    with pytest.raises(ValueError):
        hat_voice.wav("Hello", "bm_nobody")
