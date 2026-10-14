"""Step 4 - Ask the trained network which house a face belongs to (photo only, no questions).

Usage:
    python predict.py photo.jpg
    python predict.py photo1.jpg photo2.jpg
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from common import (HOUSES, MODEL_PATH, HouseNet, center_square, color, crop_face, eval_transform, get_device,
                    load_image, print_house_scores)


class HouseClassifier:
    """Loads the trained network once, then .predict(photo) as often as you like."""

    def __init__(self, model_path=MODEL_PATH):
        model_path = Path(model_path)
        if not model_path.exists():
            raise SystemExit(f"No trained model at {model_path} - run python train.py first.")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.device = get_device()
        self.model = HouseNet(pretrained=False)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()
        self.characters = checkpoint["characters"]
        self.character_houses = checkpoint["character_houses"]
        self.character_features = checkpoint["character_features"]
        self.house_lean = checkpoint["house_lean"]  # the network's favourite houses on new faces

    @torch.no_grad()
    def predict(self, image_path, lookalikes=3):
        """Returns ({house: probability}, [(character, house, similarity), ...], face_found)."""
        img = load_image(image_path)
        face = crop_face(img)
        face_found = face is not None
        if not face_found:
            face = center_square(img)
        x = eval_transform(face).unsqueeze(0)
        x = torch.cat([x, x.flip(-1)]).to(self.device)  # look at the face and its mirror image
        logits, features = self.model(x, with_features=True)
        probs = logits.softmax(1).mean(0).cpu() / self.house_lean  # don't send every new face to one house
        probs = probs / probs.sum()

        similarity = F.normalize(features.mean(0, keepdim=True), dim=1).cpu() @ self.character_features.T
        top = similarity[0].topk(min(lookalikes, len(self.characters)))
        look = [(self.characters[i], self.character_houses[i], float(s)) for s, i in zip(top.values, top.indices)]
        return dict(zip(HOUSES, probs.tolist())), look, face_found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", help="photo(s) of a person")
    parser.add_argument("--model", default=MODEL_PATH, help="trained model (default models/sorting_hat.pt)")
    args = parser.parse_args()

    classifier = HouseClassifier(args.model)
    for path in args.images:
        probs, lookalikes, face_found = classifier.predict(path)
        best = max(probs, key=probs.get)
        print(f"\n{path}" + ("" if face_found else "   (no face found - used the whole picture)"))
        print_house_scores(probs)
        print("  Looks a bit like: " + ", ".join(f"{name} ({house})" for name, house, _ in lookalikes))
        print(f"  -> {color(best.upper(), best)}")


if __name__ == "__main__":
    main()
