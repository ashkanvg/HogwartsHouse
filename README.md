# 🎩 Hogwarts House – a Sorting Hat that looks at your face

A fun terminal game. Show the Sorting Hat a photo of a person; it reads the face with a neural
network trained on Harry Potter movie characters, asks 3–5 Sorting-Hat questions, and announces
the house: **Gryffindor, Hufflepuff, Ravenclaw or Slytherin**.

```
photo ──► find the face ──► neural network ──► house guess from the face ─┐
                                                                           ├─► 🎩 "RAVENCLAW!"
questions.json ──► 3-5 questions (picked to break the tie) ──► answers ───┘
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run it, step by step

| Step | Command | What it does |
|------|---------|--------------|
| 1 | `python scrape.py` | Gets every movie character with a house from [HP-API](https://hp-api.onrender.com), then downloads up to 50 pictures each from the [Harry Potter fandom wiki](https://harrypotter.fandom.com) into `data/raw/<House>/<Character>/`. Takes about 10 minutes. |
| 2 | `python prepare.py` | Finds the face in each picture and saves a square crop to `data/faces/`. Group photos and pictures with no face are skipped. |
| 3 | `python train.py` | Trains the neural network and saves it to `models/sorting_hat.pt`. Takes about 1–2 minutes on an Apple Silicon Mac. |
| 4 | `python predict.py photo.jpg` | Photo only: shows the house scores and which characters the face looks like. |
| 5 | `python sorting_hat.py` | **The game**: photo + 3–5 questions → your house. |

Tip: look through `data/faces/` after step 2 and delete wrong crops (a different person, a
cartoon, ...). Cleaner pictures give a better model.

## The game

```bash
python sorting_hat.py                       # it asks for a photo (drag one into the terminal, or skip)
python sorting_hat.py --image me.jpg
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--image` | asks | Photo of the person to sort |
| `--min-questions` | 3 | Always ask at least this many questions |
| `--max-questions` | 5 | Never ask more than this many |
| `--sure` | 0.15 | Stop asking once the best house leads the runner-up by this much |
| `--image-weight` | 0.4 | How much the face counts compared to the answers (0 = answers only, 1 = face only) |
| `--questions` | `questions.json` | Question bank to use |
| `--seed` | random | Fixes the random choices, for a repeatable game |

How the hat decides:

1. **Face:** the network gives a probability for each house, and the hat tells you which movie
   characters you look like.
2. **Questions:** each answer gives points to one or two houses. The hat picks the next question
   to separate the two houses it is torn between (e.g. Gryffindor vs Slytherin), and stops early
   once it is sure.
3. **Blend:** `final = image_weight × face + (1 − image_weight) × answers`.
4. **Near tie:** the hat asks which house *you* would prefer. It takes your choice into account,
   just like it did for Harry.

### Your own questions

`questions.json` has 18 questions inspired by the books and movies (the troll in the dungeon,
the midnight duel, the Mirror of Erised, the Goblet of Fire, boggarts, the Deathly Hallows...).
Add, remove or change them freely:

```json
{
  "id": "goblet",
  "text": "Would you put your name into the Goblet of Fire?",
  "options": [
    {"text": "Instantly. Eternal glory!", "scores": {"Gryffindor": 2, "Slytherin": 1}},
    {"text": "Only to make my house proud...", "scores": {"Hufflepuff": 3}}
  ]
}
```

## How the neural network works

- **Face finding:** [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet),
  a small face-detection network included with OpenCV. Its model file (230 KB) downloads
  automatically into `models/` the first time.
- **Classifier:** transfer learning with a ResNet-18 pretrained on ImageNet. Its last layer is
  replaced with a 4-house layer, then the last block and the new layer are fine-tuned on the face
  crops. Rarer houses get extra weight in the loss, because there are far more Gryffindor pictures.
- **Honest validation:** `train.py` holds out whole *characters*, not just some of their
  pictures. The validation score therefore measures faces the model has never seen, which is
  the real game. Random guessing scores 25%.
- **Fair to new faces:** on faces it has never seen, the network leans towards some houses
  (mostly the ones with extra weight in the loss). `train.py` measures this lean on the
  held-out characters and saves it, and `predict.py` divides it out. That way a new face isn't
  automatically sent to Ravenclaw.
- **Look-alikes:** the average face features of each character are saved with the model, so
  `predict.py` can tell you who you resemble.

Useful options: `python train.py --epochs 20`, `--train-all` (fine-tune the whole network), and
`--val-split 0` (train on every character once you're happy with the settings).

## Be honest with your friends 😉

Your face doesn't decide your house. The network learns what the *movie characters* look like:
hair colour, lighting, costume colours, the actors themselves. With the current data (about
1,000 face crops of 81 characters), it gets 96% right on characters it trained on, but only
about 31% on characters it has never seen. Random guessing gets 25%. That's why the questions
get 60% of the vote by default. It's a game!

## Files

```
common.py        shared code: houses, paths, face cropping, the network
scrape.py        step 1 – download character pictures
prepare.py       step 2 – crop faces
train.py         step 3 – train the network
predict.py       step 4 – house from a photo only
sorting_hat.py   the game
questions.json   the Sorting Hat's question bank
data/            downloaded pictures and face crops (not in git)
models/          trained network and face detector (not in git)
```

The pictures are © their owners (Warner Bros. and others). They are only used locally to train
this toy model. `data/` and `models/` are in `.gitignore` so they are never uploaded; please
don't publish them.

## Next ideas

- A graphical version: webcam photo, animated hat, house colours
- More questions, and multiple languages in `questions.json`
