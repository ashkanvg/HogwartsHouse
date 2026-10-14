"""Step 2 - Find the face in every downloaded picture and cut it out.

    data/raw/<House>/<Character>/*.jpg  ->  data/faces/<House>/<Character>/*.jpg  (square face crops)

Pictures with no face, or group photos where we can't tell who is who, are skipped.
Have a look in data/faces afterwards: deleting wrong crops by hand makes the model better.

Usage:
    python prepare.py
"""
import argparse
from collections import Counter

from common import FACES_DIR, HOUSES, RAW_DIR, crop_face, load_image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=256, help="size of the saved face crops in pixels (default 256)")
    args = parser.parse_args()

    pictures = sorted(RAW_DIR.glob("*/*/*.jpg"))
    if not pictures:
        raise SystemExit(f"No pictures in {RAW_DIR} - run python scrape.py first.")

    faces, characters, skipped = Counter(), {house: set() for house in HOUSES}, 0
    for i, path in enumerate(pictures, 1):
        if i % 200 == 0:
            print(f"  {i}/{len(pictures)} pictures checked...")
        house, character = path.parent.parent.name, path.parent.name
        out = FACES_DIR / path.relative_to(RAW_DIR)
        if not out.exists():
            face = crop_face(load_image(path), strict=True)
            if face is None:
                skipped += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            face.resize((args.size, args.size)).save(out, quality=92)
        faces[house] += 1
        characters[house].add(character)

    print(f"\nKept {sum(faces.values())} faces, skipped {skipped} pictures (no face or group photo).")
    for house in HOUSES:
        print(f"  {house:<11} {faces[house]:5d} faces from {len(characters[house])} characters")
    print(f"Saved in {FACES_DIR}. Next step: python train.py")


if __name__ == "__main__":
    main()
