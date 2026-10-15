"""Training: groups for cross-validation, the house network, calibration and look-alikes."""
from pathlib import Path

import numpy as np
import torch

import train


def test_each_character_is_in_one_group_and_every_house_is_spread_out():
    by_house = [{f"g{i}" for i in range(10)}, {f"h{i}" for i in range(5)}, {f"r{i}" for i in range(5)},
                {f"s{i}" for i in range(7)}]
    group = train.character_folds(by_house, folds=5, seed=0)
    assert set(group.values()) == set(range(5))
    for characters in by_house:
        sizes = [sum(group[c] == g for c in characters) for g in range(5)]
        assert max(sizes) - min(sizes) <= 1


def test_the_network_learns_houses_it_can_tell_apart():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(4, 128))
    labels = np.repeat(np.arange(4), 30)
    prints = (centers[labels] + 0.3 * rng.normal(size=(120, 128))).astype(np.float32)
    network = train.train_network(prints, labels, epochs=200, seed=0)
    with torch.no_grad():
        logits = network(torch.as_tensor(prints)).numpy()
    assert train.balanced_accuracy(logits, labels) > 0.95


def test_calibration_softens_an_overconfident_network():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 4, 400)
    logits = (rng.normal(size=(400, 4)) * 10).astype(np.float32)  # very sure of itself, but guessing
    temperature, lean = train.calibrate(logits, labels)
    assert temperature > 3
    assert abs(float(lean.sum()) - 1) < 1e-5


def test_look_alike_fingerprints_are_each_characters_average():
    samples = [(Path("a.jpg"), 0, "Harry_Potter"), (Path("b.jpg"), 0, "Harry_Potter"), (Path("c.jpg"), 3, "Draco_Malfoy")]
    prints = np.array([[1, 0], [0.8, 0.6], [0, 1]], dtype=np.float32)
    names, houses, centers = train.character_fingerprints(prints, samples)
    assert names == ["Draco Malfoy", "Harry Potter"] and houses == ["Slytherin", "Gryffindor"]
    assert np.allclose(np.linalg.norm(centers, axis=1), 1)
