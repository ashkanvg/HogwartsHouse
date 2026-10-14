"""The Sorting Hat's brain: questions, scores and the final decision.

There is no printing or input() in here, so the terminal game (sorting_hat.py) and the
graphical game (app.py) share the same brain.

How the hat thinks:
  * The face reading gives a probability for every house (or nothing, if there's no photo).
  * Every yes/no answer gives points to a house (see questions.json).
  * combine() blends the two: image_weight x face + (1 - image_weight) x answers.
  * next_question() runs a little tournament: two "semi-final" questions cover all four houses,
    then the leading house is tested against its strongest rivals - every house gets a fair chance.
  * The hat stops once it is sure enough. On a near tie it lets you choose, like it did for Harry.
"""
import json
import math
from pathlib import Path

from common import HOUSES, QUESTIONS_PATH

IMAGE_WEIGHT = 0.4         # how much the face counts compared to the answers (0 to 1)
MIN_QUESTIONS = 3          # always ask at least this many questions
MAX_QUESTIONS = 5          # never ask more than this many
SURE = 0.15                # stop asking once the leading house is this far ahead
ANSWER_TEMPERATURE = 2.5   # each answer (2 points) makes its house about 2.2 times more likely

WHISPERS = {  # what the hat mutters when a house takes the lead
    "Gryffindor": "Hmm... I see courage. A taste for adventure...",
    "Hufflepuff": "Hmm... a loyal heart, and not afraid of hard work...",
    "Ravenclaw": "Hmm... a sharp, curious mind...",
    "Slytherin": "Hmm... ambition. A real thirst to prove yourself...",
}
MURMURS = ["Interesting...", "I see...", "Yes, yes...", "Oh, that's telling...", "Hmm... hmm..."]


def load_questions(path=QUESTIONS_PATH):
    questions = json.loads(Path(path).read_text())["questions"]
    for q in questions:
        for answer in ("yes", "no"):
            if not set(q.get(answer, {})) <= set(HOUSES):
                raise SystemExit(f"{path}: the '{answer}' of question '{q['id']}' names an unknown house")
    if len({q["id"] for q in questions}) != len(questions):
        raise SystemExit(f"{path}: two questions have the same id")
    return questions


def houses_in(question):
    """The houses a question is about, e.g. {'Gryffindor', 'Slytherin'}."""
    return set(question["yes"]) | set(question["no"])


def add_answer(points, question, yes):
    scores = question["yes" if yes else "no"]
    return [p + scores.get(house, 0) for p, house in zip(points, HOUSES)]


def combine(face, points, image_weight=IMAGE_WEIGHT):
    """Blend what the hat saw (face probabilities, or None) with what it heard (answer points)."""
    exps = [math.exp(p / ANSWER_TEMPERATURE) for p in points]
    answers = [e / sum(exps) for e in exps]
    if face is None:
        return answers
    return [image_weight * f + (1 - image_weight) * a for f, a in zip(face, answers)]


def ranking(probs, rng):
    """House names from most to least likely (ties broken at random)."""
    order = sorted(range(len(HOUSES)), key=lambda h: (probs[h], rng.random()), reverse=True)
    return [HOUSES[h] for h in order]


def next_question(questions, asked, probs, rng):
    """Pick the next question like a tournament.

    1. Semi-finals: houses that haven't been asked about yet go first (likeliest first),
       so after two questions every house has had its chance.
    2. Then king of the hill: the leading house against the strongest rival it hasn't faced yet,
       and once it has faced everyone, against the runner-up again.
    """
    remaining = [q for q in questions if q["id"] not in asked]
    order = ranking(probs, rng)
    tested = set().union(*(houses_in(q) for q in questions if q["id"] in asked))
    untested = [house for house in order if house not in tested]
    if len(untested) >= 2:
        pairs = [untested[:2]]
    else:
        leader = order[0]
        faced = set().union(*(houses_in(q) for q in questions if q["id"] in asked and leader in houses_in(q)))
        rivals = untested + [r for r in order[1:] if r not in faced] + order[1:]
        pairs = [[leader, rival] for rival in rivals if rival != leader]
    for pair in pairs:
        matches = [q for q in remaining if set(pair) <= houses_in(q)]
        if matches:
            return rng.choice(matches)
    return rng.choice(remaining)


def done_asking(probs, n_asked, min_questions=MIN_QUESTIONS, max_questions=MAX_QUESTIONS, sure=SURE):
    """Stop at the maximum, or earlier once the leading house is far enough ahead."""
    first, second = sorted(probs, reverse=True)[:2]
    return n_asked >= max_questions or (n_asked >= min_questions and first - second >= sure)


def whisper(old_probs, new_probs, rng):
    """What the hat mutters after an answer: about the leader if it just took the lead, else a murmur."""
    leader = ranking(new_probs, rng)[0]
    best_before = max(old_probs)
    if old_probs[HOUSES.index(leader)] < best_before or old_probs.count(best_before) > 1:
        return WHISPERS[leader]
    return rng.choice(MURMURS)


def near_tie(probs, rng, sure=SURE):
    """The two houses the hat can't choose between, or None if it has made up its mind."""
    first, second = ranking(probs, rng)[:2]
    if probs[HOUSES.index(first)] - probs[HOUSES.index(second)] < sure / 3:
        return [first, second]
    return None


def decide(probs, prefer=None, sure=SURE):
    """The final house, plus the final scores. The hat takes your preferred house into account."""
    probs = [p + (sure if house == prefer else 0) for p, house in zip(probs, HOUSES)]
    final = {house: p / sum(probs) for house, p in zip(HOUSES, probs)}
    return max(final, key=final.get), final
