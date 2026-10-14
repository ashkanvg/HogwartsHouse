"""The Sorting Hat - terminal game.

1. The hat looks at your photo (the neural network trained by train.py).
2. It asks you 3 to 5 yes/no questions from questions.json, picked so every house gets a fair
   chance, and stops early once it is sure.
3. It blends what it saw with what it heard, and announces your house.

The thinking lives in hat_brain.py (shared with the graphical version, app.py).

Usage:
    python sorting_hat.py                    # asks for a photo (you can skip it)
    python sorting_hat.py --image me.jpg
    python sorting_hat.py --image me.jpg --image-weight 0.3 --min-questions 3 --max-questions 5
"""
import argparse
import random
import re
import sys
import time
from pathlib import Path

import hat_brain as brain
from common import HOUSES, MODEL_PATH, QUESTIONS_PATH, color, print_house_scores

_classifier = None  # the neural network, loaded on first use


def hat(text, pause=0.8):
    print(f"\n🎩 {text}")
    if sys.stdout.isatty():
        time.sleep(pause)


def ask(question, number):
    """Show a yes/no question; return True for yes."""
    print(f"\n  Question {number}: {question['text']}")
    while True:
        answer = input("  Your answer (y/n): ").strip().lower()
        if answer in ("y", "yes", "n", "no"):
            return answer.startswith("y")
        print("  The hat didn't understand. Type y for yes or n for no.")


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
    result = _classifier.predict(image)
    if not result["face_found"]:
        hat("I can't quite find your face in this picture... I'll look at all of it.")
    hat("Let me see that face...")
    print_house_scores(result["probs"], indent="     ")
    hat(f"You remind me a little of {' and '.join(name for name, _, _ in result['lookalikes'][:2])}...")
    return [result["probs"][house] for house in HOUSES]


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
    points, asked = [0] * len(HOUSES), []
    probs = brain.combine(face, points, args.image_weight)
    for number in range(1, args.max_questions + 1):
        question = brain.next_question(questions, asked, probs, rng)
        asked.append(question["id"])
        points = brain.add_answer(points, question, ask(question, number))
        old_probs, probs = probs, brain.combine(face, points, args.image_weight)
        if brain.done_asking(probs, number, args.min_questions, args.max_questions, args.sure):
            break
        hat(brain.whisper(old_probs, probs, rng), 0.5)

    prefer = None
    tie = brain.near_tie(probs, rng, args.sure)
    if tie:  # a near tie: the hat lets you choose, like Harry did
        hat(f"Difficult. VERY difficult. {tie[0]}... or {tie[1]}?")
        choice = input("   Is there a house you'd prefer? (type its name, or just press Enter): ").strip().lower()
        prefer = next((house for house in tie if choice and house.lower().startswith(choice)), None)
        if prefer:
            hat("Are you sure? Well... the hat does take your choice into account.")

    house, final = brain.decide(probs, prefer, args.sure)
    hat("Right then. I know just where to put you...", 1.5)
    print(f"\n     ✨  {color(house.upper() + '!', house)}  ✨\n")
    print_house_scores(final, indent="     ")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", help="photo of the person to sort (asked for if missing)")
    parser.add_argument("--questions", default=QUESTIONS_PATH, help="question bank (default questions.json)")
    parser.add_argument("--model", default=MODEL_PATH, help="trained network (default models/sorting_hat.pt)")
    parser.add_argument("--min-questions", type=int, default=brain.MIN_QUESTIONS,
                        help=f"always ask at least this many (default {brain.MIN_QUESTIONS})")
    parser.add_argument("--max-questions", type=int, default=brain.MAX_QUESTIONS,
                        help=f"never ask more than this many (default {brain.MAX_QUESTIONS})")
    parser.add_argument("--image-weight", type=float, default=brain.IMAGE_WEIGHT,
                        help=f"how much the face counts compared to the answers, 0 to 1 (default {brain.IMAGE_WEIGHT})")
    parser.add_argument("--sure", type=float, default=brain.SURE,
                        help=f"stop asking once the best house leads by this much (default {brain.SURE})")
    parser.add_argument("--seed", type=int, help="random seed, for a repeatable game")
    args = parser.parse_args()

    questions = brain.load_questions(args.questions)
    args.max_questions = max(1, min(args.max_questions, len(questions)))
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
