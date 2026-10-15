"""The hat's thinking, checked with simulated students."""
import random
from collections import Counter

import pytest

import hat_brain as brain
from common import HOUSES

QUESTIONS = brain.load_questions()


def sort(rng, answer_for, face=None):
    """Play one game; answer_for(question) returns True for yes. Returns (house, number of questions)."""
    points, asked = [0] * len(HOUSES), []
    probs = brain.combine(face, points)
    for number in range(1, brain.MAX_QUESTIONS + 1):
        question = brain.next_question(QUESTIONS, asked, probs, rng)
        asked.append(question["id"])
        points = brain.add_answer(points, question, answer_for(question))
        probs = brain.combine(face, points)
        if brain.done_asking(probs, number):
            break
    return brain.decide(probs)[0], len(asked)


def honest(house, rng):
    """A student who always answers in favour of their own house, and guesses on the other questions."""
    return lambda q: (house in q["yes"]) if house in brain.houses_in(q) else rng.random() < 0.5


def test_the_first_two_questions_cover_all_four_houses():
    rng = random.Random(0)
    for _ in range(200):
        weights = [rng.random() for _ in HOUSES]
        probs = brain.combine([w / sum(weights) for w in weights], [0] * len(HOUSES))  # any face reading
        asked, covered = [], set()
        for _ in range(2):
            question = brain.next_question(QUESTIONS, asked, probs, rng)
            asked.append(question["id"])
            covered |= brain.houses_in(question)
        assert covered == set(HOUSES)


@pytest.mark.parametrize("house", HOUSES)
def test_an_honest_student_is_sorted_into_their_house(house):
    rng = random.Random(1)
    results = [sort(rng, honest(house, rng)) for _ in range(300)]
    assert all(got == house for got, _ in results)
    assert all(brain.MIN_QUESTIONS <= n <= brain.MAX_QUESTIONS for _, n in results)


@pytest.mark.parametrize("house", HOUSES)
def test_a_misleading_face_cannot_outvote_honest_answers(house):
    rng = random.Random(4)
    for wrong in (h for h in HOUSES if h != house):
        face = [0.4 if h == wrong else 0.2 for h in HOUSES]  # stronger than real (calibrated) face readings
        assert all(sort(rng, honest(house, rng), face)[0] == house for _ in range(100))


def test_random_answers_spread_evenly_over_the_houses():
    rng = random.Random(2)
    counts = Counter(sort(rng, lambda q: rng.random() < 0.5)[0] for _ in range(4000))
    for house in HOUSES:
        assert 0.21 < counts[house] / 4000 < 0.29, counts


def test_students_who_like_nicer_answers_are_still_sorted_fairly():
    """Even when students often give the nicer-sounding answer, every house is recognised about equally well."""
    rng = random.Random(3)
    found = {}
    for house in HOUSES:
        def answer_for(q, be_honest=honest(house, rng)):
            if "nicer" in q and rng.random() < 0.3:
                return q["nicer"] == "yes"
            return be_honest(q)
        found[house] = sum(sort(rng, answer_for)[0] == house for _ in range(1000)) / 1000
    assert min(found.values()) > 0.85, found
    assert max(found.values()) - min(found.values()) < 0.08, found


def test_a_near_tie_lets_the_student_choose():
    probs = [0.30, 0.29, 0.21, 0.20]
    assert set(brain.near_tie(probs, random.Random(5))) == {"Gryffindor", "Hufflepuff"}
    assert brain.decide(probs)[0] == "Gryffindor"
    assert brain.decide(probs, prefer="Hufflepuff")[0] == "Hufflepuff"


def test_the_hat_stops_when_the_questions_run_out():
    assert brain.done_asking([0.25] * 4, n_asked=2, n_questions=2)
    assert not brain.done_asking([0.25] * 4, n_asked=2)
