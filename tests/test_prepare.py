"""The cleaning step: faces that don't look like the rest of their character are found."""
import numpy as np

import prepare


def unit(vector):
    return vector / np.linalg.norm(vector)


def test_faces_unlike_their_character_are_found():
    rng = np.random.default_rng(0)
    person = unit(rng.normal(size=128))
    same = [unit(person + 0.5 * unit(rng.normal(size=128))) for _ in range(8)]  # one person, different photos
    strangers = [unit(rng.normal(size=128)) for _ in range(2)]
    assert prepare.outliers(np.stack(same + strangers)) == [8, 9]


def test_nothing_is_moved_when_most_faces_disagree():
    rng = np.random.default_rng(1)
    faces = np.stack([unit(rng.normal(size=128)) for _ in range(6)])
    assert prepare.outliers(faces, threshold=0.99) == []  # all six disagree: we can't tell who is right
