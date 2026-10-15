"""Step 4 - Ask the trained network which house a face belongs to (photo only, no questions).

Usage:
    python predict.py photo.jpg
    python predict.py photo1.jpg photo2.jpg
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from common import (HOUSES, MODEL_PATH, center_square, color, crop_face, face_fingerprint, house_network, load_image,
                    print_house_scores)


class HouseClassifier:
    """Loads the trained network once, then .predict(photo) as often as you like."""

    def __init__(self, model_path=MODEL_PATH):
        model_path = Path(model_path)
        if not model_path.exists():
            raise SystemExit(f"No trained model at {model_path} - run python train.py first.")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if "network" not in checkpoint:
            raise SystemExit(f"{model_path} was made by an older version - run python train.py again.")
        self.network = house_network()
        self.network.load_state_dict(checkpoint["network"])
        self.network.eval()
        self.characters = checkpoint["characters"]
        self.character_houses = checkpoint["character_houses"]
        self.character_prints = checkpoint["character_prints"].numpy()
        # calibration from train.py: how sure the network may be, and its favourite houses on new faces
        self.temperature = checkpoint["temperature"]
        self.house_lean = checkpoint["house_lean"]

    @torch.no_grad()
    def predict(self, image, lookalikes=3):
        """Read a photo (a file path or an open file).

        Returns {"probs": {house: probability}, "lookalikes": [(character, house, similarity), ...],
                 "face_found": bool, "face": the cropped face as a PIL image}
        """
        img = load_image(image)
        face = crop_face(img)
        face_found = face is not None
        if not face_found:
            face = center_square(img)
        fingerprint, _ = face_fingerprint(img)
        logits = self.network(torch.as_tensor(fingerprint, dtype=torch.float32).unsqueeze(0))[0]
        probs = (logits / self.temperature).softmax(0) / self.house_lean
        probs = probs / probs.sum()

        similarity = self.character_prints @ fingerprint  # 1 = the same face, around 0 = strangers
        closest = np.argsort(-similarity)[:lookalikes]
        look = [(self.characters[i], self.character_houses[i], float(similarity[i])) for i in closest]
        return {"probs": dict(zip(HOUSES, probs.tolist())), "lookalikes": look, "face_found": face_found, "face": face}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", help="photo(s) of a person")
    parser.add_argument("--model", default=MODEL_PATH, help="trained model (default models/sorting_hat.pt)")
    args = parser.parse_args()

    classifier = HouseClassifier(args.model)
    for path in args.images:
        result = classifier.predict(path)
        probs = result["probs"]
        best = max(probs, key=probs.get)
        print(f"\n{path}" + ("" if result["face_found"] else "   (no face found - used the whole picture)"))
        print_house_scores(probs)
        print("  Looks like: " + ", ".join(f"{name} ({house}, {similarity:.0%} alike)"
                                           for name, house, similarity in result["lookalikes"]))
        print(f"  -> {color(best.upper(), best)}")


if __name__ == "__main__":
    main()
