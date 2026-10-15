# 🎩 Hogwarts House – a Sorting Hat that looks at your face

A fun game. Show the Sorting Hat a photo of a person (or use the webcam). It reads the face with a
neural network trained on Harry Potter movie characters, asks 3–5 yes/no Sorting-Hat questions
out loud, and announces the house: **Gryffindor, Hufflepuff, Ravenclaw or Slytherin**.

Play it in a window in your browser (`python app.py`) or in the terminal (`python sorting_hat.py`).

```
photo ──► find the face ──► neural network ──► a hint of a house ──────┐
                                                                        ├─► 🎩 "RAVENCLAW!"
questions.json ──► 3-5 yes/no questions (a little tournament) ──► answers ──┘
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` allows a range of versions. To install exactly the versions this was tested
with (macOS, Python 3.13), use `pip install -r requirements-lock.txt` instead.

## Run it, step by step

| Step | Command | What it does |
|------|---------|--------------|
| 1 | `python scrape.py` | Gets every movie character with a house from [HP-API](https://hp-api.onrender.com), then downloads up to 50 pictures each from the [Harry Potter fandom wiki](https://harrypotter.fandom.com) into `data/raw/<House>/<Character>/`. Takes about 10 minutes. |
| 2 | `python prepare.py` | Finds the face in each picture and saves a square crop to `data/faces/`. Group photos and pictures with no face are skipped. |
| 3 | `python train.py` | Trains the neural network and saves it to `models/sorting_hat.pt`. Takes about 1–2 minutes on an Apple Silicon Mac. |
| 4 | `python predict.py photo.jpg` | Photo only: shows the house scores and which characters the face looks like. |
| 5 | `python app.py` | **The game, graphical version**: opens in your browser. |
| 5b | `python sorting_hat.py` | The same game in the terminal. |

Tip: look through `data/faces/` after step 2 and delete wrong crops (a different person, a
cartoon, ...). Cleaner pictures give a better model.

## The graphical game

```bash
python app.py        # opens http://127.0.0.1:5050 in your browser
```

- Click **Wake the Sorting Hat**, then use the **webcam**, choose a photo, drop one on the page,
  or skip the photo. iPhone `.HEIC` photos work too.
- The hat shows what it sees in your face and which characters you look like, then asks its
  questions out loud. Answer with the **Yes / No** buttons, or the <kbd>Y</kbd> / <kbd>N</kbd> keys.
- A shield in your house colours, with sparkles, announces the result.
- If the server hiccups, just answer again, or press **Try again**.
- It works with the keyboard (<kbd>Tab</kbd>, <kbd>Enter</kbd>, <kbd>Y</kbd>/<kbd>N</kbd>) and with
  screen readers. The animations stop if your computer is set to reduce motion.

### The hat's voice

The page uses your browser's built-in voices. It picks a **British male voice** (on a Mac that's
"Daniel"), then lowers it and slows it down so it sounds like a wise old hat. You can choose
another voice from the menu at the top right, or mute it with 🔊.

- **Can it sound like a real Harry Potter actor?** No. Copying a real person's voice without
  their permission isn't okay. So the hat has its own *character-style* voice instead.
- **Want a nicer voice?** On a Mac, go to System Settings → Accessibility → Spoken Content →
  System Voice → Manage Voices… → English (United Kingdom). Download a voice such as
  "Daniel (Enhanced)" or "Jamie (Premium)", then restart the browser and pick it from the menu.

## The terminal game

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

The same defaults are used by the graphical game; you can change them at the top of `hat_brain.py`.

## How the hat decides

All the thinking is in `hat_brain.py`, shared by both versions.

1. **Face:** the network gives a probability for each house. It's calibrated, so it's only a
   hint (see below). The hat also tells you which movie characters you look like.
2. **Questions:** every yes/no question pits two houses against each other. *Yes* gives 2 points
   to one house and *no* gives 2 points to the other. The hat runs a little tournament:
   - **Semi-finals:** the first two questions cover all four houses, starting with the two the
     face suggests.
   - **Final:** the leading house then faces its strongest rival.
   - It stops after 3 questions if it's sure, and asks up to 5 if it isn't.
3. **Blend:** `final = image_weight × face + (1 − image_weight) × answers`.
4. **Near tie:** the hat asks which house *you* would prefer. It takes your choice into account,
   just like it did for Harry.

Simulations show the hat is fair:
- Students who always answer in favour of their true house: 100% sorted correctly, even with no
  photo.
- Students who answer at random: spread evenly, 25% in each house.
- Students who sometimes give the answer that sounds nicer (like handing back lost Galleons): the
  questions are written so this doesn't push anyone away from a house. If they give the nicer
  answer 30% of the time, every house is still recognised 93–96% of the time. (An earlier
  version of the questions found only 67% of true Slytherins.)

### Your own questions

`questions.json` has 20 yes/no questions inspired by the books and movies: the troll, the
Triwizard dragon, Mandrakes, the Ravenclaw riddle door, Chocolate Frog cards, and more. Add or
change them freely:

```json
{"id": "troll", "text": "A troll is loose in the castle and a classmate is missing. Would you go looking for them yourself?",
 "yes": {"Gryffindor": 2}, "no": {"Slytherin": 2}, "nicer": "yes"}
```

Keep the bank fair:
- Every house should get the same number of *yes* and *no* chances. Right now that's 5 of each.
- Both answers should be easy to say out loud. If one answer sounds nicer anyway, mark it with
  `"nicer"`, and balance it with another question whose nicer answer helps the house this one hurts.
- Run `pytest` afterwards: it checks all of this (see [Tests](#tests)).

## How the neural network works

- **Face finding:** [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet),
  a small face-detection network included with OpenCV. Its model file (230 KB) downloads
  automatically into `models/` the first time. It comes from one fixed version, and is only used
  if its fingerprint (SHA-256) matches.
- **Classifier:** transfer learning with a ResNet-18 pretrained on ImageNet. Its last layer is
  replaced with a 4-house layer, then the last block and the new layer are fine-tuned on the face
  crops. Rarer houses get extra weight in the loss, because there are far more Gryffindor pictures.
- **Honest validation:** `train.py` holds out whole *characters*, not just some of their
  pictures. The validation score therefore measures faces the model has never seen, which is
  the real game. Random guessing scores 25%.
- **Calibration:** on faces it has never seen, the network was far too sure of itself. It would
  say "82% Ravenclaw" while being right only about 31% of the time. `train.py` uses the held-out
  characters for two fixes, and `predict.py` applies both:
  - It learns a *temperature* (about 5): the scores are divided by it, which softens the
    probabilities until they match how often the network is really right.
  - It measures which houses the network still leans towards on new faces, and divides that out.
- **Look-alikes:** the average face features of each character are saved with the model, so the
  hat can tell you who you resemble.

Useful options: `python train.py --epochs 20`, `--train-all` (fine-tune the whole network), and
`--val-split 0` (train on every character; this also skips calibration).

## Be honest with your friends 😉

Your face doesn't decide your house. The network learns what the *movie characters* look like:
hair colour, lighting, costume colours, the actors themselves. With the current data (about
1,000 face crops of 81 characters), it gets 96% right on characters it trained on, but only
about 31% on characters it has never seen. Random guessing gets 25%. So the face gives a hint and
breaks ties, and your answers decide. It's a game!

## Tests

```bash
pytest
```

The tests take a few seconds and don't need a trained model. They check three things:
- The question bank is fair, and broken questions give a clear message.
- Simulated students are sorted correctly and fairly: honest ones, random ones, ones who like
  nicer answers, and ones whose face points to the wrong house.
- The web server handles both good and bad requests.

## Files

```
app.py           the graphical game (a small local website)
web/             its page: index.html, style.css, app.js
sorting_hat.py   the terminal game
hat_brain.py     the hat's thinking: questions, scores, tournament, decision
questions.json   the Sorting Hat's yes/no questions
common.py        shared code: houses, paths, face cropping, the network
scrape.py        step 1 – download character pictures
prepare.py       step 2 – crop faces
train.py         step 3 – train and calibrate the network
predict.py       step 4 – house from a photo only
tests/           the tests (run with: pytest)
requirements.txt / requirements-lock.txt   allowed versions / exact tested versions
data/            downloaded pictures and face crops (not in git)
models/          trained network and face detector (not in git)
```

The pictures are © their owners (Warner Bros. and others). They are only used locally to train
this toy model and to show look-alikes. `data/` and `models/` are in `.gitignore` so they are
never uploaded; please don't publish them.
