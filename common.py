"""Shared pieces used by every script: houses, paths, faces and their fingerprints, the neural network."""
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # hide OpenCV's harmless backend warnings
import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from torch import nn

register_heif_opener()  # lets Pillow open iPhone photos (.HEIC)

HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"          # pictures downloaded by scrape.py
FACES_DIR = ROOT / "data" / "faces"      # face crops made by prepare.py
REJECTED_DIR = ROOT / "data" / "rejected"  # face crops prepare.py moved out: they don't look like their character
FINGERPRINTS_PATH = FACES_DIR / "fingerprints.npz"  # face fingerprints of the crops, computed once
MODEL_PATH = ROOT / "models" / "sorting_hat.pt"
QUESTIONS_PATH = ROOT / "questions.json"

# YuNet: a tiny face-detection network from the official OpenCV model zoo (downloaded once, 230 KB).
# The link points at one fixed version, and the file must match its fingerprint (SHA-256) before we use it.
FACE_DETECTOR_URL = ("https://github.com/opencv/opencv_zoo/raw/f12e12798e8314f7c074a6656816c048dcc95b7a"
                     "/models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
FACE_DETECTOR_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
FACE_DETECTOR_PATH = ROOT / "models" / "face_detection_yunet_2023mar.onnx"

# SFace: a face-recognition network from the same model zoo (downloaded once, 39 MB). It turns a face
# into 128 numbers, a "face fingerprint": photos of the same person get similar fingerprints.
FACE_RECOGNIZER_URL = ("https://github.com/opencv/opencv_zoo/raw/ba91a3b91d00d76e86540d4013f944bd6b514e39"
                       "/models/face_recognition_sface/face_recognition_sface_2021dec.onnx")
FACE_RECOGNIZER_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
FACE_RECOGNIZER_PATH = ROOT / "models" / "face_recognition_sface_2021dec.onnx"

FINGERPRINT_SIZE = 128  # SFace describes a face with 128 numbers


# ---------------------------------------------------------------- images & faces

def load_image(path):
    """Open any picture as RGB, rotated the right way up (phone photos often aren't)."""
    img = ImageOps.exif_transpose(Image.open(path))
    return img.convert("RGB")


_face_detector = None
_face_recognizer = None


def sha256(path):
    """A file's fingerprint, read in 1 MB pieces so big model files don't fill the memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_file(path, url, fingerprint, what):
    """A model file: downloaded once, and only used if its fingerprint (SHA-256) is the expected one."""
    if path.exists() and sha256(path) == fingerprint:
        return path
    print(f"Downloading the {what} (once)...")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    urllib.request.urlretrieve(url, partial)
    if sha256(partial) != fingerprint:
        partial.unlink()
        raise SystemExit(f"The downloaded {what} isn't the expected file, so it won't be used. Try again later.")
    partial.replace(path)
    return path


def detect_face_rows(img):
    """Find faces with YuNet, biggest first. Each row is x, y, w, h, then five landmarks
    (eyes, nose tip, mouth corners) as x, y pairs, then how sure the detector is."""
    global _face_detector
    if _face_detector is None:
        path = model_file(FACE_DETECTOR_PATH, FACE_DETECTOR_URL, FACE_DETECTOR_SHA256, "face detector")
        _face_detector = cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.8, 0.3, 50)

    scale = min(1.0, 800 / max(img.size))  # big phone photos: detect on a smaller copy, it's faster
    small = img.resize((round(img.width * scale), round(img.height * scale))) if scale < 1 else img
    _face_detector.setInputSize(small.size)
    _, faces = _face_detector.detect(np.asarray(small)[:, :, ::-1].copy())  # OpenCV wants BGR
    if faces is None:
        return []
    rows = [np.concatenate([face[:14] / scale, face[14:]]).astype(np.float32) for face in faces]
    return sorted(rows, key=lambda row: row[2] * row[3], reverse=True)


def detect_faces(img):
    """Find faces with YuNet. Returns (x, y, w, h) boxes, biggest first."""
    return [tuple(int(v) for v in row[:4]) for row in detect_face_rows(img)]


def face_fingerprint(img):
    """The face fingerprint of the biggest face: 128 numbers from SFace, scaled to length 1, so
    multiplying two fingerprints gives their similarity (1 = identical, around 0 = strangers).

    The face is first straightened using its eyes, nose and mouth. Returns (fingerprint, face_found);
    without a face it fingerprints the middle of the picture instead.
    """
    global _face_recognizer
    if _face_recognizer is None:
        path = model_file(FACE_RECOGNIZER_PATH, FACE_RECOGNIZER_URL, FACE_RECOGNIZER_SHA256, "face recognizer")
        _face_recognizer = cv2.FaceRecognizerSF.create(str(path), "")
    rows = detect_face_rows(img)
    if rows:
        aligned = _face_recognizer.alignCrop(np.asarray(img)[:, :, ::-1].copy(), rows[0])
    else:
        aligned = cv2.resize(np.asarray(center_square(img))[:, :, ::-1].copy(), (112, 112))
    feature = _face_recognizer.feature(aligned).flatten()
    return feature / np.linalg.norm(feature), bool(rows)


def crop_fingerprints(paths):
    """Face fingerprints of face-crop files, remembered in data/faces/fingerprints.npz so each one is
    only computed once (delete that file to recompute them all)."""
    cache = {}
    if FINGERPRINTS_PATH.exists():
        saved = np.load(FINGERPRINTS_PATH)
        cache = dict(zip(saved["paths"].tolist(), saved["prints"]))
    keys = [str(Path(p).relative_to(FACES_DIR)) for p in paths]
    missing = [(key, p) for key, p in zip(keys, paths) if key not in cache]
    for key, p in missing:
        cache[key] = face_fingerprint(load_image(p))[0]
    if missing:
        np.savez(FINGERPRINTS_PATH, paths=np.array(list(cache)), prints=np.stack(list(cache.values())))
    return np.stack([cache[key] for key in keys]) if keys else np.zeros((0, FINGERPRINT_SIZE), np.float32)


def crop_face(img, margin=0.4, strict=False):
    """Cut out a square around the biggest face (with some hair and collar around it).

    Returns None when there is no face. strict=True is for training pictures: it ignores small
    background faces, and skips group photos where another face is nearly as big (we couldn't
    tell which person the picture is about).
    """
    faces = detect_faces(img)
    if strict:
        faces = [f for f in faces if min(f[2], f[3]) >= max(48, min(img.size) // 10)]
        if len(faces) > 1 and faces[1][2] * faces[1][3] > 0.5 * faces[0][2] * faces[0][3]:
            return None
    if not faces:
        return None
    x, y, w, h = faces[0]
    side = min(max(w, h) * (1 + 2 * margin), img.width, img.height)
    left = min(max(x + w / 2 - side / 2, 0), img.width - side)   # keep the square inside the picture
    top = min(max(y + h / 2 - side / 2, 0), img.height - side)
    return img.crop((int(left), int(top), int(left + side), int(top + side)))


def center_square(img):
    """Fallback when no face is found: the middle square of the picture."""
    side = min(img.size)
    left, top = (img.width - side) // 2, (img.height - side) // 2
    return img.crop((left, top, left + side, top + side))


# ---------------------------------------------------------------- the neural network

def house_network():
    """The small neural network that guesses the house from a face fingerprint:
    128 numbers in, a score for each of the 4 houses out."""
    return nn.Sequential(nn.Linear(FINGERPRINT_SIZE, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, len(HOUSES)))


# ---------------------------------------------------------------- terminal output

_COLORS = {"Gryffindor": "\033[31m", "Hufflepuff": "\033[33m", "Ravenclaw": "\033[34m", "Slytherin": "\033[32m"}


def color(text, house):
    """Paint text in the house colour (only in a real terminal)."""
    return f"{_COLORS[house]}{text}\033[0m" if sys.stdout.isatty() else text


def bar(p, width=20):
    filled = round(p * width)
    return "█" * filled + "░" * (width - filled)


def print_house_scores(probs, indent="  "):
    """probs: {house: probability}. Prints one bar per house, best first."""
    for house, p in sorted(probs.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{indent}{color(f'{house:<11}', house)} {color(bar(p), house)} {p:4.0%}")
