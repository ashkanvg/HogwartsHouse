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


def still(seed, size=(400, 300)):
    """A made-up single still: a smooth gradient, different for every seed."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size[1], 0:size[0]]
    gray = rng.uniform(60, 190) + rng.uniform(-0.15, 0.15) * x + rng.uniform(-0.15, 0.15) * y
    return Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).convert("RGB")


def mosaic(seed, size=(400, 300)):
    """A made-up mosaic: four flat pieces of different brightness, glued together."""
    rng = np.random.default_rng(seed)
    cx, cy = rng.integers(size[0] // 4, 3 * size[0] // 4), rng.integers(size[1] // 4, 3 * size[1] // 4)
    gray = np.empty((size[1], size[0]), np.uint8)
    gray[:cy, :cx], gray[:cy, cx:], gray[cy:, :cx], gray[cy:, cx:] = 30, 200, 120, 250
    return Image.fromarray(gray).convert("RGB")


def test_labels_are_matched_to_characters():
    labels = {"HARRY-POTTER", "LORD-VOLDEMORT", "MINERVA-MCGONAGALL", "movie-characters", "DOBBY"}
    cast = import_roboflow.match_characters(labels, CHARACTERS)
    assert {label: c["name"] for label, c in cast.items()} == {
        "HARRY-POTTER": "Harry Potter", "LORD-VOLDEMORT": "Lord Voldemort", "MINERVA-MCGONAGALL": "Minerva McGonagall"}


def test_copies_of_a_still_have_the_same_name():
    assert import_roboflow.original_still("train/scene12_jpg.rf.5f2a9c.jpg") == "scene12_jpg"
    assert import_roboflow.original_still("train/scene12_jpg.rf.0b17ee.jpg") == "scene12_jpg"
    assert import_roboflow.original_still("still.jpg") == "still.jpg"


def test_mosaics_have_seams_and_stills_dont():
    assert import_roboflow.seam_score(still(1)) < import_roboflow.SEAM
    assert import_roboflow.seam_score(mosaic(1)) > import_roboflow.SEAM
    noisy = np.asarray(still(2)).copy()
    noisy[np.random.default_rng(2).random(noisy.shape[:2]) < 0.01] = 255  # Roboflow's salt-and-pepper noise
    assert import_roboflow.seam_score(Image.fromarray(noisy)) < import_roboflow.SEAM


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


def roboflow_zip(path, parts):
    """A zip laid out like Roboflow's COCO export. parts: {"train": {file: (picture, [(label, box), ...])}, ...}"""
    labels = sorted({label for files in parts.values() for _, boxes in files.values() for label, _ in boxes})
    categories = [{"id": 0, "name": "movie-characters", "supercategory": "none"}]
    categories += [{"id": i, "name": label, "supercategory": "movie-characters"} for i, label in enumerate(labels, 1)]
    ids = {c["name"]: c["id"] for c in categories}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("README.roboflow.txt", "a made-up dataset")
        for part, files in parts.items():
            images, annotations = [], []
            for i, (file, (picture, boxes)) in enumerate(files.items()):
                images.append({"id": i, "file_name": file, "width": picture.width, "height": picture.height})
                data = io.BytesIO()
                picture.save(data, "JPEG", quality=95)
                archive.writestr(f"{part}/{file}", data.getvalue())
                annotations += [{"id": len(annotations), "image_id": i, "category_id": ids[label], "bbox": list(box)}
                                for label, box in boxes]
            coco = {"categories": categories, "images": images, "annotations": annotations}
            archive.writestr(f"{part}/_annotations.coco.json", json.dumps(coco))
    return path


def test_one_single_still_is_kept_per_still(tmp_path):
    harry = [("HARRY-POTTER", (100, 100, 60, 60))]
    train = {"a_jpg.rf.1.jpg": (still(1), harry), "a_jpg.rf.2.jpg": (mosaic(2), harry), "a_jpg.rf.3.jpg": (mosaic(3), harry),
             "b_jpg.rf.4.jpg": (mosaic(4), harry), "b_jpg.rf.5.jpg": (still(5), harry), "b_jpg.rf.6.jpg": (mosaic(6), harry),
             # two different stills that were both called "image.jpg": 2 x 3 copies
             "image_jpg.rf.7.jpg": (still(7), harry), "image_jpg.rf.8.jpg": (still(8), harry)}
    train.update({f"image_jpg.rf.{n}.jpg": (mosaic(n), harry) for n in range(9, 13)})
    valid = {"c_jpg.rf.13.jpg": (still(13), harry), "d_jpg.rf.14.jpg": (mosaic(14), harry)}  # no copies here
    with zipfile.ZipFile(roboflow_zip(tmp_path / "dataset.zip", {"train": train, "valid": valid})) as archive:
        files = sorted({file for file, _, _ in import_roboflow.read_coco(archive)})
        assert import_roboflow.single_stills(archive, files) == {
            "train/a_jpg.rf.1.jpg", "train/b_jpg.rf.5.jpg", "train/image_jpg.rf.7.jpg", "train/image_jpg.rf.8.jpg",
            "valid/c_jpg.rf.13.jpg"}


def test_importing_the_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(scrape, "RAW_DIR", tmp_path)
    # a pretend face detector: a face in the middle of every cut-out, except in Voldemort's 120x120 one,
    # where the biggest face is in the corner - someone standing next to him
    monkeypatch.setattr(import_roboflow, "detect_faces", lambda img: [(0, 0, 20, 20)] if img.size == (120, 120)
                        else [(img.width // 2 - 10, img.height // 2 - 10, 20, 20)])
    zip_path = roboflow_zip(tmp_path / "dataset.zip", {
        "train": {
            "scene1_jpg.rf.aaa.jpg": (still(1), [("HARRY-POTTER", (100, 100, 60, 80)),
                                                 ("LORD-VOLDEMORT", (300, 100, 60, 60)),
                                                 ("DOBBY", (10, 10, 60, 60))]),        # not a character with a house
            "scene1_jpg.rf.bbb.jpg": (mosaic(2), [("HARRY-POTTER", (110, 100, 60, 80))]),  # mosaics of the same still
            "scene1_jpg.rf.ccc.jpg": (mosaic(3), [("HARRY-POTTER", (0, 0, 60, 60))]),
            "scene2_jpg.rf.ddd.jpg": (still(4), [("MINERVA-MCGONAGALL", (50, 50, 20, 20))]),  # too small for a face
            "scene2_jpg.rf.eee.jpg": (mosaic(5), [("MINERVA-MCGONAGALL", (50, 50, 80, 80))]),
            "scene2_jpg.rf.fff.jpg": (mosaic(6), [("HARRY-POTTER", (0, 0, 60, 60))]),
        },
        "valid": {"scene3_jpg.rf.ggg.jpg": (still(7), [("HARRY-POTTER", (200, 150, 50, 50))]),
                  "scene4_jpg.rf.hhh.jpg": (still(8), [("LORD-VOLDEMORT", (50, 50, 60, 70))])},
    })
    harry = scrape.character_folder(CHARACTERS[0])
    harry.mkdir(parents=True)
    (harry / "sources.json").write_text(json.dumps({"000.jpg": "https://wiki/harry.jpg"}))

    assert import_roboflow.import_stills(zip_path, CHARACTERS) == {"Harry Potter": 2, "Lord Voldemort": 1}
    assert sorted(p.name for p in harry.glob("*.jpg")) == ["roboflow_000.jpg", "roboflow_001.jpg"]
    assert Image.open(harry / "roboflow_000.jpg").size == (120, 160)  # the 60x80 box with room around it
    sources = json.loads((harry / "sources.json").read_text())
    assert sources["000.jpg"] == "https://wiki/harry.jpg"  # the other pictures' sources are kept
    assert sources["roboflow_000.jpg"] == {"url": import_roboflow.DATASET_URL, "still": "train/scene1_jpg.rf.aaa.jpg",
                                           "box": [100, 100, 60, 80], **import_roboflow.CREDIT}
    assert sources["roboflow_001.jpg"]["still"] == "valid/scene3_jpg.rf.ggg.jpg"

    assert import_roboflow.import_stills(zip_path, CHARACTERS) == {}  # running it again adds nothing
    assert len(list(harry.glob("roboflow_*.jpg"))) == 2
