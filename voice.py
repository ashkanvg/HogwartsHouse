"""The Sorting Hat's voice: a natural-sounding neural voice that runs on your own computer.

It uses Kokoro (an open text-to-speech model, Apache-2.0 licence) through the kokoro-onnx
package, with its British male voices. The hat speaks a little slower and deeper than normal,
like a wise old hat. The model files (about 350 MB) download once into models/voice/, and every
spoken line is saved in models/voice/cache/, so each line is only generated once.

Try it:
    python voice.py "Hmm... difficult. VERY difficult." --voice bm_fable   # writes hat_voice.wav
"""
import argparse
import hashlib
import threading
import urllib.request
import wave
from pathlib import Path

import numpy as np

from common import ROOT, sha256

VOICE_DIR = ROOT / "models" / "voice"
CACHE_DIR = VOICE_DIR / "cache"
DOWNLOADS = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
MODEL_FILES = {  # file name: its fingerprint (SHA-256), checked before use
    "kokoro-v1.0.onnx": "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
    "voices-v1.0.bin": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
}
VOICES = {"bm_george": "George", "bm_fable": "Fable", "bm_lewis": "Lewis", "bm_daniel": "Daniel"}  # British men
DEFAULT_VOICE = "bm_george"  # George and Fable are Kokoro's best-rated British male voices
DEEPER = 0.92                # play 8% slower, which also makes the voice a little deeper


def download_model_files():
    """Fetch the model files once, and only use files whose fingerprint is the expected one."""
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    for name, fingerprint in MODEL_FILES.items():
        path = VOICE_DIR / name
        if path.exists() and sha256(path) == fingerprint:
            continue
        print(f"Downloading the hat's voice: {name} (once)...")
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(DOWNLOADS + name, partial)
        if sha256(partial) != fingerprint:
            partial.unlink()
            raise RuntimeError(f"{name} isn't the expected file, so it won't be used.")
        partial.replace(path)


def write_wav(path, samples, rate):
    """Save float samples (-1..1) as a 16-bit mono WAV file."""
    audio = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(audio.tobytes())


class HatVoice:
    """Loads the voice once; .wav(text) returns the spoken line as WAV file bytes."""

    def __init__(self, engine=None):
        if engine is None:  # the real thing; tests pass in a fake engine instead
            from kokoro_onnx import Kokoro  # imported here because it's optional
            download_model_files()
            engine = Kokoro(str(VOICE_DIR / "kokoro-v1.0.onnx"), str(VOICE_DIR / "voices-v1.0.bin"))
        self.engine = engine
        self.lock = threading.Lock()  # one line at a time

    def wav(self, text, voice=DEFAULT_VOICE):
        if voice not in VOICES:
            raise ValueError(f"unknown voice {voice!r}")
        key = hashlib.sha1(f"{voice}|{DEEPER}|{text}".encode()).hexdigest()
        path = CACHE_DIR / f"{key}.wav"
        if not path.exists():
            with self.lock:
                samples, rate = self.engine.create(text, voice=voice, speed=1.0, lang="en-gb")
            samples = samples / max(float(np.abs(samples).max()), 1e-6) * 0.9  # every line equally loud
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".part")
            write_wav(partial, samples, int(rate * DEEPER))  # a slower sample rate = slower and deeper
            partial.replace(path)
        return path.read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", help="what the hat should say")
    parser.add_argument("--voice", default=DEFAULT_VOICE, choices=VOICES)
    parser.add_argument("--out", default="hat_voice.wav")
    args = parser.parse_args()
    Path(args.out).write_bytes(HatVoice().wav(args.text, args.voice))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
