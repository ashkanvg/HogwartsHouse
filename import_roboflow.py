"""Extra pictures (optional) - film stills from the Roboflow dataset "Harry potter character detection".

The dataset (https://universe.roboflow.com/leanse-sangeerth-6o0b2/harry-potter-character-detection,
by Leanse Sangeerth, licence CC BY 4.0) has 675 pictures, mostly film stills, with a labelled box around
each of 13 main characters. Downloading it needs a free Roboflow account: on that page, click
"Download Dataset", choose the format "COCO JSON" and "Download zip to computer".

Roboflow squeezed every still to 640x640 and made 3 copies of each training still: the still itself
(slightly changed) and two "mosaics" glued together from pieces of four stills. Mosaics give cut-off
faces and copies of other stills, so only the single stills are used (see single_stills()).
The README says how much these stills help (in short: not with the house guess).

Every labelled box is cut out with some room around it (so prepare.py still finds the whole face) and
saved in the character's folder as data/raw/<House>/<Character>/roboflow_000.jpg, ... A box is only
used when the biggest face in the cut-out is inside the box - otherwise it might be someone standing
next to the character. Each folder's sources.json says which still every picture came from.
Stills that were imported before are skipped, so it's safe to run it again.

Usage:
    python import_roboflow.py ~/Downloads/"Harry potter character detection.v1i.coco.zip"
    python prepare.py && python train.py
"""
import argparse
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from common import HOUSES, RAW_DIR, detect_faces
from scrape import character_folder, plain_name

DATASET_URL = "https://universe.roboflow.com/leanse-sangeerth-6o0b2/harry-potter-character-detection"
CREDIT = {"author": "Leanse Sangeerth (Roboflow Universe)", "licence": "CC BY 4.0"}
MARGIN = 0.5   # room added on every side of a box, as a fraction of its size
MIN_BOX = 48   # boxes smaller than this (in pixels) are too small for a usable face
SEAM = 0.3     # single stills have weaker seams than this, mosaics stronger ones (see seam_score())


def read_coco(archive):
    """Every labelled box in a Roboflow COCO export: [(picture file in the zip, label, (x, y, w, h)), ...]."""
    boxes = []
    for name in sorted(archive.namelist()):
        if not name.endswith("_annotations.coco.json"):  # one per part: train/, valid/, test/
            continue
        folder = name[:name.rfind("/") + 1]
        coco = json.loads(archive.read(name))
        labels = {c["id"]: c["name"] for c in coco["categories"]}
        files = {img["id"]: folder + img["file_name"] for img in coco["images"]}
        boxes += [(files[a["image_id"]], labels[a["category_id"]], tuple(a["bbox"])) for a in coco["annotations"]]
    return boxes


def original_still(file_name):
    """Roboflow renames 'frame12.jpg' to 'frame12_jpg.rf.<code>.jpg', with a new code for every
    copy it makes. This gives back the still's own name."""
    return Path(file_name).name.split(".rf.")[0]


def seam_score(img):
    """How much a picture looks glued together from four pieces, like Roboflow's mosaics: the share
    of the picture's width where the brightness jumps along its strongest straight line across, or
    of its height along the strongest line from top to bottom, whichever is smaller.
    Single stills mostly score below 0.2, mosaics 0.3 to 0.9."""
    gray = np.asarray(img.convert("L").filter(ImageFilter.MedianFilter(3)), np.float32)  # no salt-and-pepper noise
    across = (np.abs(np.diff(gray, axis=0)) > 24).mean(1)[4:-4]  # every row (but the edges): how much of it jumps
    down = (np.abs(np.diff(gray, axis=1)) > 24).mean(0)[4:-4]    # every column, the same
    return float(min(across.max(), down.max()))


def single_stills(archive, files):
    """The files that are single stills, not mosaics.

    Copies of a still share its name. So for every name we keep the copy with the weakest seams -
    or a few copies, as different stills can share a name like 'image.jpg'.
    """
    by_name = defaultdict(list)
    for file in files:
        by_name[(Path(file).parent, original_still(file))].append(file)
    copies = {}  # how many copies Roboflow made of every still, per part: 3 in train/, 1 in valid/ and test/
    for part in {part for part, _ in by_name}:
        copies[part] = Counter(len(group) for (p, _), group in by_name.items() if p == part).most_common(1)[0][0]
    singles = set()
    for (part, _), group in by_name.items():
        scored = sorted((seam_score(Image.open(io.BytesIO(archive.read(file)))), file) for file in group)
        stills = max(1, round(len(group) / copies[part]))
        singles.update(file for score, file in scored[:stills] if score < SEAM)
    return singles


def match_characters(labels, characters):
    """{label: character} for the labels that name a character: 'HARRY-POTTER' is Harry Potter."""
    by_name = {plain_name(c["name"]): c for c in characters}
    return {label: by_name[plain_name(label)] for label in labels if plain_name(label) in by_name}


def crop_box(img, box, margin=MARGIN):
    """Cut out the box (x, y, w, h) with `margin` of its size added on every side, inside the picture.
    Returns the cut-out and where the box is in it."""
    x, y, w, h = box
    left, top = max(0, round(x - margin * w)), max(0, round(y - margin * h))
    right, bottom = min(img.width, round(x + w * (1 + margin))), min(img.height, round(y + h * (1 + margin)))
    return img.crop((left, top, right, bottom)), (x - left, y - top, w, h)


def face_is_inside(img, box):
    """Is the biggest face in the picture inside the box - so it is the labelled character's face?"""
    faces = detect_faces(img)
    if not faces:
        return False
    fx, fy, fw, fh = faces[0]
    x, y, w, h = box
    return x <= fx + fw / 2 <= x + w and y <= fy + fh / 2 <= y + h


def import_stills(zip_path, characters):
    """Import every labelled box of the zip into its character's folder. Returns pictures added per character."""
    added = Counter()
    with zipfile.ZipFile(zip_path) as archive:
        boxes = read_coco(archive)
        if not boxes:
            raise SystemExit("No COCO labels in that zip. Download it again in the \"COCO JSON\" format.")
        cast = match_characters({label for _, label, _ in boxes}, characters)
        if unknown := sorted({label for _, label, _ in boxes} - set(cast)):
            print("Labels that aren't a character with a house (skipped):", ", ".join(unknown))

        singles = single_stills(archive, sorted({file for file, _, _ in boxes}))
        by_character = {}  # character name -> {still: box}
        for file, label, box in boxes:
            if file in singles and label in cast and min(box[2], box[3]) >= MIN_BOX:
                by_character.setdefault(cast[label]["name"], {}).setdefault(file, box)

        for character in characters:
            folder = character_folder(character)
            sources_file = folder / "sources.json"
            sources = json.loads(sources_file.read_text()) if sources_file.exists() else {}
            done = {s.get("still") for s in sources.values() if isinstance(s, dict)}  # imported before
            number = 0
            for still, box in sorted(by_character.get(character["name"], {}).items()):
                if still in done:
                    continue
                img = Image.open(io.BytesIO(archive.read(still))).convert("RGB")
                cutout, box_in_cutout = crop_box(img, box)
                if not face_is_inside(cutout, box_in_cutout):
                    continue
                while (folder / f"roboflow_{number:03d}.jpg").exists():
                    number += 1
                name = f"roboflow_{number:03d}.jpg"
                folder.mkdir(parents=True, exist_ok=True)
                cutout.save(folder / name, quality=90)
                sources[name] = {"url": DATASET_URL, "still": still, "box": [round(v) for v in box], **CREDIT}
                sources_file.write_text(json.dumps(sources, indent=1))  # straight away: a stopped run keeps every credit
                added[character["name"]] += 1
    return added


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("zip", type=Path, help="the dataset zip downloaded from Roboflow (COCO JSON format)")
    args = parser.parse_args()

    characters_file = RAW_DIR / "characters.json"
    if not characters_file.exists():
        raise SystemExit(f"No {characters_file} - run python scrape.py first.")
    characters = json.loads(characters_file.read_text())

    added = import_stills(args.zip, characters)
    houses = Counter()
    for character in characters:
        if added[character["name"]]:
            houses[character["house"]] += added[character["name"]]
            print(f"  {character['house']:<11} {character['name']:<22} {added[character['name']]:4d} pictures")
    print(f"\nAdded {sum(added.values())} pictures:", ", ".join(f"{h} {houses[h]}" for h in HOUSES))
    print("Next step: python prepare.py")


if __name__ == "__main__":
    main()
