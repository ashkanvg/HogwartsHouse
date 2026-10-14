"""The Sorting Hat - terminal game.

1. The hat looks at your photo (the neural network trained by train.py).
2. It asks you 3 to 5 questions from questions.json. Each question is picked to separate the two
   houses the hat is torn between, and it stops early once it is sure.
3. It blends what it saw with what it heard, and announces your house.

Usage:
    python sorting_hat.py                    # asks for a photo (you can skip it)
    python sorting_hat.py --image me.jpg
    python sorting_hat.py --image me.jpg --image-weight 0.3 --min-questions 3 --max-questions 5
"""
import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

from common import HOUSES, MODEL_PATH, QUESTIONS_PATH, color, print_house_scores

ANSWER_TEMPERATURE = 3.0  # a 3-point lead in the answers makes a house e (~2.7) times more likely
WHISPERS = {
    "Gryffindor": "Hmm... I see courage. A taste for adventure...",
    "Hufflepuff": "Hmm... a loyal heart, and not afraid of hard work...",
    "Ravenclaw": "Hmm... a sharp, curious mind...",
    "Slytherin": "Hmm... ambition. A real thirst to prove yourself...",
}
MURMURS = ["Interesting...", "I see...", "Yes, yes...", "Oh, that's telling...", "Not what I expected..."]

_classifier = None  # the neural network, loaded on first use


def hat(text, pause=0.8):
    print(f"\n🎩 {text}")
    if sys.stdout.isatty():
        time.sleep(pause)


def load_questions(path):
    questions = json.loads(Path(path).read_text())["questions"]
    for q in questions:
        for option in q["options"]:
            if set(option["scores"]) - set(HOUSES):
                raise SystemExit(f"{path}: question '{q['id']}' gives points to an unknown house: {option['scores']}")
    if len({q["id"] for q in questions}) != len(questions):
        raise SystemExit(f"{path}: two questions have the same id")
    return questions


def combine(face, points, image_weight):
    """Blend what the hat saw (face probabilities) with what it heard (answer points)."""
    exps = [math.exp(p / ANSWER_TEMPERATURE) for p in points]
    answers = [e / sum(exps) for e in exps]
    if face is None:
        return answers
    return [image_weight * f + (1 - image_weight) * a for f, a in zip(face, answers)]


def top_two(probs, rng):
    """Indexes of the two leading houses (ties broken at random)."""
    order = sorted(range(len(HOUSES)), key=lambda h: (probs[h], rng.random()), reverse=True)
    return order[0], order[1]


def next_question(questions, asked, probs, rng):
    """The unasked question whose answers best separate the two houses the hat is torn between."""
    first, second = (HOUSES[h] for h in top_two(probs, rng))

    def separation(question):
        return sum(abs(o["scores"].get(first, 0) - o["scores"].get(second, 0)) for o in question["options"])

    remaining = [q for q in questions if q["id"] not in asked]
    best = max(map(separation, remaining))
    return rng.choice([q for q in remaining if separation(q) == best])


def ask(question, number, rng):
    """Show the question with its answers shuffled; return the points of the chosen answer."""
    options = rng.sample(question["options"], len(question["options"]))
    print(f"\n  Question {number}: {question['text']}")
    for i, option in enumerate(options, 1):
        print(f"    {i}) {option['text']}")
    while True:
        answer = input(f"  Your answer (1-{len(options)}): ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]["scores"]
        print("  The hat didn't understand. Type the number of your answer.")


def ask_for_photo():
    path = input("\n   Drag a photo into this window and press Enter (or just press Enter to skip): ").strip()
    return re.sub(r"\\(.)", r"\1", path.strip("'\""))  # terminals add quotes or backslashes to dragged paths


def look_at_face(image, model_path):
    """Run the neural network on the photo. Returns house probabilities (in HOUSES order) or None."""
    global _classifier
    if not Path(model_path).exists():
        hat("(I have no eyes yet: run python train.py first. Your answers will have to do!)")
        return None
    if _classifier is None:
        from predict import HouseClassifier  # imported here because loading PyTorch takes a moment
        _classifier = HouseClassifier(model_path)
    probs, lookalikes, face_found = _classifier.predict(image)
    if not face_found:
        hat("I can't quite find your face in this picture... I'll look at all of it.")
    hat("Let me see that face...")
    print_house_scores(probs, indent="     ")
    hat(f"You remind me a little of {' and '.join(name for name, _, _ in lookalikes[:2])}...")
    return [probs[house] for house in HOUSES]


def sort_student(args, questions, rng, image=None):
    hat("Ah... a new student! Let me take a look at you.")
    if image is None:
        image = ask_for_photo()
    face = None
    if image and Path(image).is_file():
        try:
            face = look_at_face(image, args.model)
        except OSError:  # not a picture, or a format Pillow can't open
            hat("Hmm, I can't read that picture. Never mind!")
    elif image:
        hat(f"I can't find a picture at '{image}'. Never mind!")

    hat("But a face doesn't tell me everything. Let me look inside your mind...")
    points, asked, leader = [0] * len(HOUSES), [], None
    probs = combine(face, points, args.image_weight)
    while len(asked) < args.max_questions:
        question = next_question(questions, asked, probs, rng)
        asked.append(question["id"])
        scores = ask(question, len(asked), rng)
        points = [p + scores.get(house, 0) for p, house in zip(points, HOUSES)]
        probs = combine(face, points, args.image_weight)
        first, second = top_two(probs, rng)
        if len(asked) >= args.min_questions and probs[first] - probs[second] >= args.sure:
            break  # the hat is sure enough
        if len(asked) < args.max_questions:
            hat(WHISPERS[HOUSES[first]] if HOUSES[first] != leader else rng.choice(MURMURS), 0.5)
            leader = HOUSES[first]

    first, second = top_two(probs, rng)
    if probs[first] - probs[second] < args.sure / 3:  # a near tie: the hat lets you choose, like Harry did
        hat(f"Difficult. VERY difficult. {HOUSES[first]}... or {HOUSES[second]}?")
        choice = input("   Is there a house you'd prefer? (type its name, or just press Enter): ").strip().lower()
        for h in (first, second):
            if choice and HOUSES[h].lower().startswith(choice):
                hat("Are you sure? Well... the hat does take your choice into account.")
                probs[h] += args.sure
        first, second = top_two(probs, rng)

    hat("Right then. I know just where to put you...", 1.5)
    winner = HOUSES[first]
    print(f"\n     ✨  {color(winner.upper() + '!', winner)}  ✨\n")
    print_house_scores({house: p / sum(probs) for house, p in zip(HOUSES, probs)}, indent="     ")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", help="photo of the person to sort (asked for if missing)")
    parser.add_argument("--questions", default=QUESTIONS_PATH, help="question bank (default questions.json)")
    parser.add_argument("--model", default=MODEL_PATH, help="trained network (default models/sorting_hat.pt)")
    parser.add_argument("--min-questions", type=int, default=3, help="always ask at least this many (default 3)")
    parser.add_argument("--max-questions", type=int, default=5, help="never ask more than this many (default 5)")
    parser.add_argument("--image-weight", type=float, default=0.4,
                        help="how much the face counts compared to the answers, 0 to 1 (default 0.4)")
    parser.add_argument("--sure", type=float, default=0.15,
                        help="stop asking once the best house leads by this much (default 0.15)")
    parser.add_argument("--seed", type=int, help="random seed, for a repeatable game")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    args.max_questions = min(args.max_questions, len(questions))
    args.min_questions = min(args.min_questions, args.max_questions)
    rng = random.Random(args.seed)

    print("\n" + "=" * 60 + "\n" + "THE SORTING HAT".center(60) + "\n" + "=" * 60)
    image = args.image
    try:
        while True:
            sort_student(args, questions, rng, image)
            image = None
            if input("\n   Sort another student? (y/N): ").strip().lower() not in ("y", "yes"):
                break
    except (KeyboardInterrupt, EOFError):
        print()
    hat("The hat goes back to sleep. Zzz...", 0)


if __name__ == "__main__":
    main()
