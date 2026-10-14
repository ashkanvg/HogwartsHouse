"""Step 3 - Train the Sorting Hat's eyes: a neural network that guesses the house from a face.

How it works (transfer learning):
  * Start from ResNet-18, a network that already learned to see shapes, colours and textures
    from 1.2 million ImageNet photos.
  * Replace its last layer with a new one that outputs the 4 houses.
  * Fine-tune its last block and the new layer on the face crops made by prepare.py.

Validation holds out whole *characters*, not just some of their pictures, so the validation
score shows how the hat does on faces it has never seen - which is the real game.
"Accuracy" is the average over the 4 houses, so random guessing scores about 25%.
The held-out characters are also used to calibrate how confident the network may be (calibrate()).

Usage:
    python train.py
    python train.py --epochs 20 --train-all   # fine-tune the whole network
    python train.py --val-split 0             # train on every character (no validation score)
"""
import argparse
import random
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import FACES_DIR, HOUSES, MODEL_PATH, HouseNet, eval_transform, get_device, train_transform


class FaceDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, house, _ = self.samples[i]
        return self.transform(Image.open(path).convert("RGB")), house


def load_samples():
    """[(path, house_index, character), ...] from data/faces/<House>/<Character>/*.jpg"""
    return [(path, h, path.parent.name)
            for h, house in enumerate(HOUSES)
            for path in sorted((FACES_DIR / house).glob("*/*.jpg"))]


def split_by_character(samples, val_split, seed):
    """Put `val_split` of each house's characters (with all their pictures) aside for validation."""
    rng = random.Random(seed)
    val_characters = set()
    for h in range(len(HOUSES)):
        characters = sorted({c for _, house, c in samples if house == h})
        rng.shuffle(characters)
        n_val = round(len(characters) * val_split)
        if val_split > 0 and len(characters) > 1:
            n_val = max(1, n_val)  # test every house at least once
        val_characters.update(characters[:n_val])
    train = [s for s in samples if s[2] not in val_characters]
    val = [s for s in samples if s[2] in val_characters]
    return train, val, sorted(val_characters)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over the data (training if an optimizer is given). Returns loss, accuracy, per-house accuracy."""
    model.train(optimizer is not None)
    total_loss, hits, counts = 0.0, torch.zeros(len(HOUSES)), torch.zeros(len(HOUSES))
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct, labels = (logits.argmax(1) == labels).cpu(), labels.cpu()
            hits += torch.bincount(labels[correct], minlength=len(HOUSES))
            counts += torch.bincount(labels, minlength=len(HOUSES))
    per_house = hits / counts.clamp(min=1)
    return total_loss / counts.sum().item(), per_house[counts > 0].mean().item(), per_house


@torch.no_grad()
def calibrate(model, samples, device):
    """Make the network honest about faces it has never seen (the validation characters).

    1. Temperature: the network is far too sure of itself on new faces. Dividing its scores by a
       temperature softens its probabilities until they match how often it is really right.
    2. Lean: which houses it still favours on new faces (averaged per true house, so a perfect
       network would give 25% each).
    predict.py applies both, so a new face isn't sent to one house with fake confidence.
    """
    model.eval()
    loader = DataLoader(FaceDataset(samples, eval_transform), batch_size=64)
    logits = torch.cat([model(x.to(device)).cpu() for x, _ in loader])
    labels = torch.tensor([h for _, h, _ in samples])
    every_house_counts_the_same = 1 / torch.bincount(labels, minlength=len(HOUSES)).clamp(min=1).float()
    temperatures = torch.linspace(0.5, 10, 96)
    losses = torch.stack([F.cross_entropy(logits / t, labels, weight=every_house_counts_the_same) for t in temperatures])
    temperature = temperatures[losses.argmin()].item()
    probs = (logits / temperature).softmax(1)
    lean = torch.stack([probs[labels == h].mean(0) for h in range(len(HOUSES)) if (labels == h).any()]).mean(0)
    return temperature, lean


@torch.no_grad()
def character_fingerprints(model, samples, device):
    """The average face features of every character - used to tell you who you look like."""
    model.eval()
    loader = DataLoader(FaceDataset(samples, eval_transform), batch_size=64)
    features = torch.cat([F.normalize(model(x.to(device), with_features=True)[1], dim=1).cpu() for x, _ in loader])
    groups = defaultdict(list)
    for (_, house, character), feature in zip(samples, features):
        groups[(character, house)].append(feature)
    keys = sorted(groups)
    centers = F.normalize(torch.stack([torch.stack(groups[k]).mean(0) for k in keys]), dim=1)
    return [c.replace("_", " ") for c, _ in keys], [HOUSES[h] for _, h in keys], centers


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of the new layer (the backbone gets lr/10)")
    parser.add_argument("--val-split", type=float, default=0.2, help="fraction of characters held out for validation")
    parser.add_argument("--train-all", action="store_true", help="fine-tune the whole network, not only the last block")
    parser.add_argument("--workers", type=int, default=2, help="processes loading pictures in parallel")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    samples = load_samples()
    if not samples:
        raise SystemExit(f"No faces in {FACES_DIR} - run python prepare.py first.")
    train, val, val_characters = split_by_character(samples, args.val_split, args.seed)
    counts = Counter(h for _, h, _ in train)
    print(f"{len(train)} training faces, {len(val)} validation faces")
    print("Training faces per house:", ", ".join(f"{house} {counts[h]}" for h, house in enumerate(HOUSES)))
    if val:
        print("Held-out characters:", ", ".join(c.replace("_", " ") for c in val_characters))

    device = get_device()
    print(f"Training on {device}\n")
    model = HouseNet(pretrained=True).to(device)
    for name, param in model.backbone.named_parameters():
        param.requires_grad = args.train_all or name.startswith("layer4")

    # There are many more Gryffindor pictures than Hufflepuff ones: give rarer houses more weight
    weights = torch.tensor([len(train) / (len(HOUSES) * max(counts[h], 1)) for h in range(len(HOUSES))])
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)
    optimizer = torch.optim.AdamW([
        {"params": model.head.parameters(), "lr": args.lr},
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr / 10},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    loader_args = {"num_workers": args.workers, "persistent_workers": args.workers > 0}
    train_loader = DataLoader(FaceDataset(train, train_transform), batch_size=args.batch_size, shuffle=True, **loader_args)
    val_loader = DataLoader(FaceDataset(val, eval_transform), batch_size=64, **loader_args) if val else None

    best = {"score": -1.0}
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        scheduler.step()
        line = f"epoch {epoch:2d}/{args.epochs}  train loss {train_loss:.3f} acc {train_acc:4.0%}"
        if val_loader:
            val_loss, val_acc, per_house = run_epoch(model, val_loader, criterion, device)
            line += f"  |  val loss {val_loss:.3f} acc {val_acc:4.0%}"
            score = val_acc
        else:
            val_acc, per_house, score = None, None, epoch  # no validation: keep the last epoch
        if score > best["score"]:
            best = {"score": score, "val_acc": val_acc, "per_house": per_house,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            line += "  <- best so far"
        print(line)

    model.load_state_dict(best["state"])
    names, houses, centers = character_fingerprints(model, samples, device)
    temperature, lean = calibrate(model, val, device) if val else (1.0, torch.full((len(HOUSES),), 1 / len(HOUSES)))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best["state"], "characters": names, "character_houses": houses,
                "character_features": centers, "temperature": temperature, "house_lean": lean,
                "val_accuracy": best["val_acc"]}, MODEL_PATH)

    if best["val_acc"] is not None:
        print(f"\nBest validation accuracy on unseen characters: {best['val_acc']:.0%} (random guessing: 25%)")
        print("  " + "  ".join(f"{h} {a:.0%}" for h, a in zip(HOUSES, best["per_house"].tolist())))
        print(f"Calibration (predict.py applies it): temperature {temperature:.1f}, still leans towards "
              + "  ".join(f"{h} {p:.0%}" for h, p in zip(HOUSES, lean.tolist())))
    print(f"Model saved to {MODEL_PATH}. Try it: python predict.py <photo>  or  python sorting_hat.py")


if __name__ == "__main__":
    main()
