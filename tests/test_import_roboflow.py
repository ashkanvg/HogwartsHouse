"""Importing the Roboflow film stills, with a made-up dataset zip (no internet or account needed)."""
import io
import json
import zipfile

import numpy as np
from PIL import Image

import import_roboflow
import scrape

CHARACTERS = [{"name": "Harry Potter", "house": "Gryffindor", "actor": "Daniel Radcliffe"},
              {"name": "Lord Voldemort", "house": "Slytherin", "actor": "Ralph Fiennes"},
              {"name": "Minerva McGonagall", "house": "Gryffindor", "actor": "Dame Maggie Smith"}]


def test_labels_are_matched_to_characters():
    labels = {"HARRY-POTTER", "LORD-VOLDEMORT", "MINERVA-MCGONAGALL", "harry-potter-characters", "DOBBY"}
    cast = import_roboflow.match_characters(labels, CHARACTERS)
    assert {label: c["name"] for label, c in cast.items()} == {
        "HARRY-POTTER": "Harry Potter", "LORD-VOLDEMORT": "Lord Voldemort", "MINERVA-MCGONAGALL": "Minerva McGonagall"}


def test_changed_copies_of_a_still_have_the_same_name():
    assert import_roboflow.original_still("train/scene12_jpg.rf.5f2a9c.jpg") == "scene12_jpg"
    assert import_roboflow.original_still("valid/scene12_jpg.rf.0b17ee.jpg") == "scene12_jpg"
    assert import_roboflow.original_still("still.jpg") == "still.jpg"


def test_a_box_is_cut_out_with_room_around_it():
    img = Image.new("RGB", (400, 300))
    cutout, box = import_roboflow.crop_box(img, (100, 100, 40, 60))
    assert cutout.size == (80, 120) and box == (20, 30, 40, 60)
    cutout, box = import_roboflow.crop_box(img, (370, 0, 30, 30))  # at the edge: the cut-out stays inside the picture
    assert cutout.size == (45, 45) and box == (15, 0, 30, 30)


def test_a_box_is_only_used_when_the_biggest_face_is_inside_it(monkeypatch):
    img, box = Image.new("RGB", (100, 100)), (25, 25, 50, 50)
    monkeypatch.setattr(import_roboflow, "detect_faces", lambda img: [(30, 30, 40, 40), (0, 0, 10, 10)])
    assert import_roboflow.face_is_inside(img, box)
    monkeypatch.setattr(import_roboflow, "detect_faces", lambda img: [(60, 60, 40, 40)])  # someone next to them
    assert not import_roboflow.face_is_inside(img, box)
    monkeypatch.setattr(import_roboflow, "detect_faces", lambda img: [])
    assert not import_roboflow.face_is_inside(img, box)


def jpeg(seed):
    """A 400x300 picture, as JPEG bytes."""
    noise = np.random.default_rng(seed).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    data = io.BytesIO()
    Image.fromarray(noise).save(data, "JPEG")
    return data.getvalue()


def roboflow_zip(path, parts):
    """A zip laid out like Roboflow's COCO export. parts: {"train": {file name: [(label, box), ...]}, ...}"""
    labels = sorted({label for files in parts.values() for boxes in files.values() for label, _ in boxes})
    categories = [{"id": 0, "name": "harry-potter-characters", "supercategory": "none"}]
    categories += [{"id": i, "name": label, "supercategory": "harry-potter-characters"} for i, label in enumerate(labels, 1)]
    ids = {c["name"]: c["id"] for c in categories}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("README.roboflow.txt", "a made-up dataset")
        for part, files in parts.items():
            images, annotations = [], []
            for i, (file, boxes) in enumerate(files.items()):
                images.append({"id": i, "file_name": file, "width": 400, "height": 300})
                archive.writestr(f"{part}/{file}", jpeg(i))
                annotations += [{"id": len(annotations), "image_id": i, "category_id": ids[label], "bbox": list(box)}
                                for label, box in boxes]
            coco = {"categories": categories, "images": images, "annotations": annotations}
            archive.writestr(f"{part}/_annotations.coco.json", json.dumps(coco))
    return path


def test_importing_the_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(scrape, "RAW_DIR", tmp_path)
    # a pretend face detector: a face in the middle of every cut-out, except in Voldemort's (120x120),
    # where the biggest face is in the corner - someone standing next to him
    monkeypatch.setattr(import_roboflow, "detect_faces", lambda img: [(0, 0, 20, 20)] if img.size == (120, 120)
                        else [(img.width // 2 - 10, img.height // 2 - 10, 20, 20)])
    zip_path = roboflow_zip(tmp_path / "dataset.zip", {
        "train": {
            "scene1_jpg.rf.aaa.jpg": [("HARRY-POTTER", (100, 100, 60, 80)), ("LORD-VOLDEMORT", (300, 100, 60, 60)),
                                      ("DOBBY", (10, 10, 60, 60))],     # not a character with a house
            "scene1_jpg.rf.bbb.jpg": [("HARRY-POTTER", (110, 100, 60, 80))],  # a changed copy of the same still
            "scene2_jpg.rf.ccc.jpg": [("MINERVA-MCGONAGALL", (50, 50, 20, 20))],  # too small for a face
        },
        "valid": {"scene3_jpg.rf.ddd.jpg": [("HARRY-POTTER", (200, 150, 50, 50))]},
    })
    harry = scrape.character_folder(CHARACTERS[0])
    harry.mkdir(parents=True)
    (harry / "sources.json").write_text(json.dumps({"000.jpg": "https://wiki/harry.jpg"}))

    assert import_roboflow.import_stills(zip_path, CHARACTERS) == {"Harry Potter": 2}
    assert sorted(p.name for p in tmp_path.glob("*/*/*.jpg")) == ["roboflow_000.jpg", "roboflow_001.jpg"]
    assert Image.open(harry / "roboflow_000.jpg").size == (120, 160)  # the 60x80 box with room around it
    sources = json.loads((harry / "sources.json").read_text())
    assert sources["000.jpg"] == "https://wiki/harry.jpg"  # the other pictures' sources are kept
    assert sources["roboflow_000.jpg"] == {"url": import_roboflow.DATASET_URL, "still": "scene1_jpg",
                                           "box": [100, 100, 60, 80], **import_roboflow.CREDIT}
    assert sources["roboflow_001.jpg"]["still"] == "scene3_jpg"

    assert import_roboflow.import_stills(zip_path, CHARACTERS) == {}  # running it again adds nothing
    assert len(list(harry.glob("roboflow_*.jpg"))) == 2
