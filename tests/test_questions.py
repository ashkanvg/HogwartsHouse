"""The question bank must be fair: no house may get more chances than another."""
import json
from collections import Counter

import pytest

import hat_brain as brain
from common import HOUSES

QUESTIONS = brain.load_questions()


def test_every_question_pits_two_houses_against_each_other():
    for q in QUESTIONS:
        assert len(brain.houses_in(q)) == 2, q["id"]
        assert not set(q["yes"]) & set(q["no"]), q["id"]


def test_every_house_gets_the_same_number_of_yes_and_no_chances():
    yes = Counter(house for q in QUESTIONS for house in q["yes"])
    no = Counter(house for q in QUESTIONS for house in q["no"])
    assert len({yes[house] for house in HOUSES}) == 1, yes
    assert len({no[house] for house in HOUSES}) == 1, no


def test_nicer_answers_help_and_hurt_every_house_equally():
    """If nicer-sounding answers kept helping the same houses, people who like to sound nice would skew the hat."""
    balance = Counter({house: 0 for house in HOUSES})
    for q in QUESTIONS:
        if "nicer" in q:
            nicer, other = (q["yes"], q["no"]) if q["nicer"] == "yes" else (q["no"], q["yes"])
            balance.update(dict.fromkeys(nicer, 1))
            balance.subtract(dict.fromkeys(other, 1))
    assert all(balance[house] == 0 for house in HOUSES), dict(balance)


GOOD = {"id": "q", "text": "Brave?", "yes": {"Gryffindor": 2}, "no": {"Slytherin": 2}}


@pytest.mark.parametrize("questions, message", [
    ([{key: value for key, value in GOOD.items() if key != "no"}], 'has no "no"'),
    ([{**GOOD, "yes": {"Gryffindore": 2}}], "must give points to houses"),
    ([{**GOOD, "yes": {"Gryffindor": 0}}], "must give points to houses"),
    ([{**GOOD, "no": {}}], "must give points to houses"),
    ([{**GOOD, "nicer": "maybe"}], '"nicer"'),
    ([GOOD, GOOD], "same id"),
    ([], "no questions"),
])
def test_broken_question_banks_give_a_clear_message(tmp_path, questions, message):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps({"questions": questions}))
    with pytest.raises(SystemExit, match=message):
        brain.load_questions(path)
