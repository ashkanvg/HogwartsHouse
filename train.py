"""Step 3 - Train the Sorting Hat's eyes.

1. Face fingerprints: every face crop gets its SFace fingerprint, 128 numbers from a face-recognition
   network trained on millions of faces (see common.py). They're saved, so the next run is quick.
2. Look-alikes: a character's typical fingerprint is the average of its faces' fingerprints. Your
   look-alikes are the characters whose typical fingerprint is closest to yours.
3. House hint: a small neural network learns to guess the house from a fingerprint.
4. Honest testing with cross-validation: the characters are split into 5 groups, and each group is
   guessed by a network trained only on the other 4. So every guess is about characters the network
   has never seen. Those guesses give the accuracy (random guessing scores 25%) and the
   calibration: how sure the network may be, and which houses it leans towards.
5. Finally the network is trained on every character and saved together with the look-alikes.

Usage:
    python train.py
    python train.py --folds 10 --epochs 500
"""
import argparse
import random
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from common import FACES_DIR, HOUSES, MODEL_PATH, crop_fingerprints, house_network


def load_samples():
    """[(path, house_index, character), ...] from data/faces/<House>/<Character>/*.jpg"""
    return [(path, h, path.parent.name)
            for h, house in enumerate(HOUSES)
            for path in sorted((FACES_DIR / house).glob("*/*.jpg"))]


def character_folds(characters_by_house, folds, seed):
    """{character: group number}: every house's characters are spread evenly over the groups."""
    rng = random.Random(seed)
    group = {}
    for characters in characters_by_house:
        characters = sorted(characters)
        rng.shuffle(characters)
        group.update({c: i % folds for i, c in enumerate(characters)})
    return group


def train_network(prints, labels, epochs, seed):
    """Train the house network on fingerprints. Rarer houses count more, as there are far more Gryffindors."""
    torch.manual_seed(seed)
    network = house_network()
    x, y = torch.as_tensor(prints), torch.as_tensor(labels)
    counts = Counter(labels.tolist())
    weights = torch.tensor([len(labels) / (len(HOUSES) * max(counts[h], 1)) for h in range(len(HOUSES))])
    loss_function = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(network.parameters(), lr=3e-3, weight_decay=1e-2)
    network.train()
    for _ in range(epochs):  # the data is small, so every step looks at all of it
        optimizer.zero_grad()
        loss_function(network(x), y).backward()
        optimizer.step()
    return network.eval()


def balanced_accuracy(logits, labels):
    """The average of each house's accuracy, so always guessing Gryffindor can't score well."""
    guesses = logits.argmax(1)
    return float(np.mean([(guesses[labels == h] == h).mean() for h in range(len(HOUSES)) if (labels == h).any()]))


def calibrate(logits, labels):
    """Make the network honest about faces it has never seen (the cross-validation guesses).

    1. Temperature: dividing the scores by a temperature softens the probabilities until they
       match how often the network is really right.
    2. Lean: which houses it still favours on new faces (averaged per true house, so a perfect
       network would give 25% each). predict.py divides it out.
    """
    logits, targets = torch.as_tensor(logits), torch.as_tensor(labels)
    every_house_counts_the_same = 1 / torch.bincount(targets, minlength=len(HOUSES)).clamp(min=1).float()
    temperatures = torch.linspace(0.5, 10, 96)
    losses = torch.stack([F.cross_entropy(logits / t, targets, weight=every_house_counts_the_same) for t in temperatures])
    temperature = temperatures[losses.argmin()].item()
    probs = (logits / temperature).softmax(1)
    lean = torch.stack([probs[targets == h].mean(0) for h in range(len(HOUSES)) if (targets == h).any()]).mean(0)
    return temperature, lean


def character_fingerprints(prints, samples):
    """Every character's typical fingerprint: the average of its faces' fingerprints."""
    rows = defaultdict(list)
    for i, (_, house, character) in enumerate(samples):
        rows[(character, house)].append(i)
    keys = sorted(rows)
    centers = np.stack([prints[rows[key]].mean(0) for key in keys])
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    return [c.replace("_", " ") for c, _ in keys], [HOUSES[h] for _, h in keys], centers


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, default=5, help="groups of characters for cross-validation (default 5)")
    parser.add_argument("--epochs", type=int, default=300, help="training steps for the house network (default 300)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = load_samples()
    if not samples:
        raise SystemExit(f"No faces in {FACES_DIR} - run python prepare.py first.")
    print(f"Face fingerprints for {len(samples)} faces (only new ones are computed)...")
    prints = crop_fingerprints([path for path, _, _ in samples]).astype(np.float32)
    labels = np.array([house for _, house, _ in samples])
    characters = [character for _, _, character in samples]
    counts = Counter(labels.tolist())
    print("Faces per house:", ", ".join(f"{house} {counts[h]}" for h, house in enumerate(HOUSES)))

    by_house = [{c for (_, h, c) in samples if h == house} for house in range(len(HOUSES))]
    group = character_folds(by_house, args.folds, args.seed)
    folds = np.array([group[c] for c in characters])
    unseen_logits = np.zeros((len(samples), len(HOUSES)), np.float32)
    for fold in range(args.folds):
        train, test = folds != fold, folds == fold
        network = train_network(prints[train], labels[train], args.epochs, args.seed)
        with torch.no_grad():
            unseen_logits[test] = network(torch.as_tensor(prints[test])).numpy()

    accuracy = balanced_accuracy(unseen_logits, labels)
    guesses = unseen_logits.argmax(1)
    print(f"\nCross-validation: on characters it has never seen, the network gets {accuracy:.0%} right "
          f"(random guessing: 25%)")
    print("  " + "  ".join(f"{house} {(guesses[labels == h] == h).mean():.0%}" for h, house in enumerate(HOUSES)))
    temperature, lean = calibrate(unseen_logits, labels)
    print(f"Calibration (predict.py applies it): temperature {temperature:.1f}, still leans towards "
          + "  ".join(f"{house} {p:.0%}" for house, p in zip(HOUSES, lean.tolist())))

    network = train_network(prints, labels, args.epochs, args.seed)  # the final network: every character
    names, houses, centers = character_fingerprints(prints, samples)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"network": network.state_dict(), "characters": names, "character_houses": houses,
                "character_prints": torch.as_tensor(centers), "temperature": temperature,
                "house_lean": lean, "cv_accuracy": accuracy}, MODEL_PATH)
    print(f"\nSaved the network and {len(names)} characters' look-alike fingerprints to {MODEL_PATH}.")
    print("Try it: python predict.py <photo>  or  python app.py")


if __name__ == "__main__":
    main()
