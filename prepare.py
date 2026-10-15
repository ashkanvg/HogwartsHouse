"""Step 2 - Find the face in every downloaded picture, cut it out, and clean out wrong pictures.

    data/raw/<House>/<Character>/*.jpg  ->  data/faces/<House>/<Character>/*.jpg  (square face crops)

1. Pictures with no face, or group photos where we can't tell who is who, are skipped.
2. Cleaning: every face gets a face fingerprint (SFace, see common.py). Faces that don't look like
   the rest of their character's faces (a cartoon, a LEGO figure, or simply the wrong person) are
   moved to data/rejected/, so you can look at them. They aren't added again on the next run.

Usage:
    python prepare.py
    python prepare.py --no-clean      # keep every face
"""
import argparse
import shutil
from collections import Counter

import numpy as np

from common import FACES_DIR, HOUSES, RAW_DIR, REJECTED_DIR, crop_face, crop_fingerprints, load_image

SAME_CHARACTER = 0.25  # a face at least this similar to its character's typical face stays
MIN_FACES = 5          # characters with fewer faces aren't cleaned: there's too little to compare with


def outliers(prints, threshold=SAME_CHARACTER):
    """Which faces (fingerprints, one per row) don't look like the rest?

    The typical face is worked out three times, each time only from the faces that agree with it,
    so a few wrong pictures can't pull it towards themselves. If more than half the faces disagree,
    we can't tell which ones are right, so nothing is moved.
    """
    keep = np.arange(len(prints))
    for _ in range(3):
        center = prints[keep].mean(0)
        center /= np.linalg.norm(center)
        similarity = prints @ center
        if not (similarity >= threshold).any():
            break
        keep = np.flatnonzero(similarity >= threshold)
    wrong = [i for i in range(len(prints)) if similarity[i] < threshold]
    return wrong if len(wrong) <= len(prints) / 2 else []


def clean():
    """Move faces that don't look like their character to data/rejected/. Returns the count per house."""
    moved = Counter()
    for folder in sorted(FACES_DIR.glob("*/*")):
        files = sorted(folder.glob("*.jpg"))
        if len(files) < MIN_FACES:
            continue
        for i in outliers(crop_fingerprints(files)):
            target = REJECTED_DIR / files[i].relative_to(FACES_DIR)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(files[i], target)
            moved[folder.parent.name] += 1
    return moved


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=256, help="size of the saved face crops in pixels (default 256)")
    parser.add_argument("--no-clean", action="store_true", help="keep every face, even ones unlike their character")
    args = parser.parse_args()

    pictures = sorted(RAW_DIR.glob("*/*/*.jpg"))
    if not pictures:
        raise SystemExit(f"No pictures in {RAW_DIR} - run python scrape.py first.")

    skipped = 0
    for i, path in enumerate(pictures, 1):
        if i % 200 == 0:
            print(f"  {i}/{len(pictures)} pictures checked...")
        out = FACES_DIR / path.relative_to(RAW_DIR)
        if out.exists() or (REJECTED_DIR / path.relative_to(RAW_DIR)).exists():
            continue  # done before (or cleaned out before)
        face = crop_face(load_image(path), strict=True)
        if face is None:
            skipped += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        face.resize((args.size, args.size)).save(out, quality=92)
    print(f"\nSkipped {skipped} pictures (no face, or a group photo).")

    if not args.no_clean:
        print("Cleaning: comparing every face with the rest of its character...")
        moved = clean()
        print(f"Moved {sum(moved.values())} faces that don't look like their character to {REJECTED_DIR} "
              f"(have a look!): " + ", ".join(f"{house} {moved[house]}" for house in HOUSES))

    print("\nFaces ready for training:")
    for house in HOUSES:
        faces = list((FACES_DIR / house).glob("*/*.jpg"))
        print(f"  {house:<11} {len(faces):5d} faces from {len({f.parent for f in faces})} characters")
    print(f"Saved in {FACES_DIR}. Next step: python train.py")


if __name__ == "__main__":
    main()
