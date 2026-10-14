"""Shared pieces used by every script: houses, paths, face cropping, the neural network."""
import os
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # hide OpenCV's harmless backend warnings
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from torch import nn
from torchvision import models, transforms

register_heif_opener()  # lets Pillow open iPhone photos (.HEIC)

HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"          # pictures downloaded by scrape.py
FACES_DIR = ROOT / "data" / "faces"      # face crops made by prepare.py
MODEL_PATH = ROOT / "models" / "sorting_hat.pt"
QUESTIONS_PATH = ROOT / "questions.json"

# YuNet: a tiny face-detection network from the official OpenCV model zoo (downloaded once, 230 KB)
FACE_DETECTOR_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
FACE_DETECTOR_PATH = ROOT / "models" / "face_detection_yunet_2023mar.onnx"

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet colour statistics


# ---------------------------------------------------------------- images & faces

def load_image(path):
    """Open any picture as RGB, rotated the right way up (phone photos often aren't)."""
    img = ImageOps.exif_transpose(Image.open(path))
    return img.convert("RGB")


_face_detector = None


def detect_faces(img):
    """Find faces with YuNet. Returns (x, y, w, h) boxes, biggest first."""
    global _face_detector
    if _face_detector is None:
        if not FACE_DETECTOR_PATH.exists():
            print("Downloading the face detector (once)...")
            FACE_DETECTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
            partial = FACE_DETECTOR_PATH.with_suffix(".part")
            urllib.request.urlretrieve(FACE_DETECTOR_URL, partial)
            partial.rename(FACE_DETECTOR_PATH)
        _face_detector = cv2.FaceDetectorYN.create(str(FACE_DETECTOR_PATH), "", (320, 320), 0.8, 0.3, 50)

    scale = min(1.0, 800 / max(img.size))  # big phone photos: detect on a smaller copy, it's faster
    small = img.resize((round(img.width * scale), round(img.height * scale))) if scale < 1 else img
    _face_detector.setInputSize(small.size)
    _, faces = _face_detector.detect(np.asarray(small)[:, :, ::-1].copy())  # OpenCV wants BGR
    if faces is None:
        return []
    boxes = [tuple(int(v / scale) for v in f[:4]) for f in faces]
    return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


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


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ---------------------------------------------------------------- the neural network

class HouseNet(nn.Module):
    """ResNet-18 pretrained on ImageNet, with a new last layer that outputs the 4 houses.

    The backbone turns a face into 512 numbers ("features"); the head turns those into house scores.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, len(HOUSES)))

    def forward(self, x, with_features=False):
        features = self.backbone(x)
        logits = self.head(features)
        return (logits, features) if with_features else logits


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return torch.device("mps")
    return torch.device("cpu")


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
