"""Reading a photo, with a tiny made-up model (no downloads or trained model needed)."""
import numpy as np
import pytest
import torch
from PIL import Image

import predict
from common import HOUSES, house_network


def test_a_photo_gives_house_scores_and_the_closest_look_alikes(tmp_path, monkeypatch):
    torch.manual_seed(0)
    model = tmp_path / "model.pt"
    torch.save({"network": house_network().state_dict(),
                "characters": ["Harry Potter", "Luna Lovegood", "Draco Malfoy"],
                "character_houses": ["Gryffindor", "Ravenclaw", "Slytherin"],
                "character_prints": torch.as_tensor(np.eye(3, 128, dtype=np.float32)),  # three very different faces
                "temperature": 2.0, "house_lean": torch.full((4,), 0.25), "cv_accuracy": 0.3}, model)
    face = Image.new("RGB", (64, 64))
    looks_like_luna = np.array([0.1, 0.9, 0.3] + [0.0] * 125, dtype=np.float32)
    monkeypatch.setattr(predict, "load_image", lambda image: face)
    monkeypatch.setattr(predict, "crop_face", lambda img: img)
    monkeypatch.setattr(predict, "face_fingerprint", lambda img: (looks_like_luna / np.linalg.norm(looks_like_luna), True))

    result = predict.HouseClassifier(model).predict("photo.jpg")
    assert set(result["probs"]) == set(HOUSES) and abs(sum(result["probs"].values()) - 1) < 1e-5
    assert [name for name, _, _ in result["lookalikes"]] == ["Luna Lovegood", "Draco Malfoy", "Harry Potter"]
    assert result["face_found"] and result["face"] is face


def test_a_model_from_an_older_version_gives_a_clear_message(tmp_path):
    old = tmp_path / "old.pt"
    torch.save({"model": {}}, old)
    with pytest.raises(SystemExit, match="older version"):
        predict.HouseClassifier(old)
