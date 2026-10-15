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
| 1 | `python scrape.py --commons` | Gets every movie character with a house from [HP-API](https://hp-api.onrender.com), then downloads up to 50 pictures each from the [Harry Potter fandom wiki](https://harrypotter.fandom.com) into `data/raw/<House>/<Character>/`. With `--commons` it also adds up to 25 freely licensed real-life photos of each actor from Wikimedia Commons. Takes about 10 minutes, or about an hour with `--commons`, because Wikimedia only allows slow downloads. Run it again later to top up. |
| 1b | `python import_roboflow.py <zip>` | Optional: adds film stills from a Roboflow dataset (see below). |
| 2 | `python prepare.py` | Finds the face in each picture and saves a square crop to `data/faces/`. Group photos and pictures with no face are skipped. Then it cleans: faces that don't look like the rest of their character (cartoons, LEGO, the wrong person) are moved to `data/rejected/`. |
| 3 | `python train.py` | Makes a face fingerprint for every face, tests and trains the house network, and saves everything to `models/sorting_hat.pt`. Takes a few seconds (about 30 the very first time). |
| 4 | `python predict.py photo.jpg` | Photo only: shows the house scores and which characters the face looks most like. |
| 5 | `python app.py` | **The game, graphical version**: opens in your browser. |
| 5b | `python sorting_hat.py` | The same game in the terminal. |

Tip: have a look at `data/rejected/` after step 2. If a good picture was moved there by mistake,
move it back to `data/faces/` and run `python prepare.py --no-clean`.

### Where the pictures come from

- **Harry Potter fandom wiki:** film stills and promo pictures of each character. This is the
  main source.
- **Wikimedia Commons** (`--commons`): real-life photos of the actors, the kind of picture players
  upload. Only photos under CC BY, CC BY-SA, CC0 or public domain are used (no "non-commercial"
  or "no derivatives" licences). Wikidata tells the scraper which Commons category belongs to
  which actor, so "Mark Williams" finds the right Mark Williams. Each folder's `sources.json`
  credits the photographer and the licence, written straight after each photo. If a credit ever
  goes missing, the next `--commons` run writes it down again. 46 of the 88 actors have photos
  there: about 600 photos, which gave 435 face crops (429 after cleaning).
  Wikimedia limits how fast photos can be downloaded, so the scraper goes slowly. If it gets
  "too busy" answers, it pauses and leaves the rest for a later run, which tops up.
- **Roboflow Universe** (optional): [Harry potter character detection](https://universe.roboflow.com/leanse-sangeerth-6o0b2/harry-potter-character-detection)
  by Leanse Sangeerth, listed under CC BY 4.0: 675 pictures, mostly film stills and promo
  pictures, with a box around each of 13 main characters (9 Gryffindors, 3 Slytherins, 1
  Ravenclaw). Downloading it needs a free Roboflow account: on that page click **Download
  Dataset**, choose **COCO JSON** and **Download zip to computer**. Then
  `python import_roboflow.py <the zip>` cuts out every labelled character into their folder
  (`roboflow_000.jpg`, ...), and `sources.json` records the still and the licence. Run
  `python prepare.py` and `python train.py` again afterwards.
  Roboflow squeezed every picture to 640×640 and added "mosaic" copies glued together from four
  pictures. The importer skips the mosaics and keeps one copy of each picture: 605 cut-outs,
  which gave 588 faces after cleaning.
- **Other datasets I looked for:**
  - Kaggle and Hugging Face have no Harry Potter character picture sets, only text and tables.
  - The other Roboflow sets are smaller (237 stills) and show the same main characters.
  - Big celebrity-face collections such as IMDB-WIKI are for research use only.

### Did the extra pictures help?

Faces per house after `python prepare.py` (with cleaning):

| Pictures | Gryffindor | Hufflepuff | Ravenclaw | Slytherin | All |
|---|---|---|---|---|---|
| Wiki only | 479 | 112 | 120 | 231 | 942 |
| + Commons actor photos | 692 | 179 | 171 | 329 | 1,371 |
| + Roboflow stills | 1,103 | 179 | 209 | 456 | 1,947 |

Each source was tested with the same 5-group cross-validation as `train.py`, repeated with 10
different groupings. Each time the house network is trained with and without the new pictures,
and both are tested on exactly the same faces of characters it has never seen. For the
look-alikes, every face is matched against all characters, with that face itself left out.

| Adding... | House guess on unseen characters: film stills | ... real-life photos | Look-alike: a real-life photo finds its character | ... a film still |
|---|---|---|---|---|
| Commons actor photos (to the wiki pictures) | 27% → 28% | 25% → 24% | 87% → 96% | 86% → 85% |
| Roboflow stills (to wiki + Commons) | 28% → 29% | 24% → 26% | 97% → 96% | 86% → 86% |

Neither source makes the house guess better. The changes are within the normal variation between
runs (about ±2 points), and the guess stays close to random guessing (25%). The actor photos make
the look-alikes much better for real photos, which is what players upload. The Roboflow stills
don't: for their 13 characters, a real photo finds the right character a little less often
(98% → 96%) and a film still a little more often (93% → 95%). They also make Gryffindor even
bigger. And cleaning with them moved out 12 older faces, 9 of them Dumbledore's: the stills
mostly show the later films' Dumbledore, while his Commons photos show Richard Harris (the actor
HP-API lists) decades earlier. So the Roboflow import stays optional, and the game doesn't need it.

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

The hat speaks with a **natural-sounding neural voice** that runs on your own computer: no
internet or account needed once it's downloaded. It uses
[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M), an open text-to-speech model
(Apache-2.0 licence), through [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx).

- **Voices:** four British men: George (the default), Fable, Lewis and Daniel. George and Fable
  are Kokoro's best-rated ones. Each is played 8% slower, which also makes it a little deeper,
  like a wise old hat.
- **First start:** `python app.py` downloads the voice model once (about 350 MB, checked against
  its fingerprint) into `models/voice/`.
- **Speed:** every line the hat says is saved in `models/voice/cache/`. All the questions are
  prepared in the background when the app starts, so they play without waiting. A brand-new line,
  like your look-alikes' names, takes about 2 seconds the first time.
- **Backup:** if the natural voice isn't available, the page falls back to your browser's
  built-in voices, also listed in the menu at the top right. Mute the hat with 🔊.
- **Try a voice from the terminal:** `python voice.py "Hmm... difficult." --voice bm_fable`
  writes `hat_voice.wav`.
- **Can it sound like a real Harry Potter actor?** No. Copying a real person's voice without their
  permission isn't okay, so the hat has its own *character-style* voice instead.

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

Three neural networks work together:

1. **Finding the face:** [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet),
   a small face-detection network included with OpenCV. It also finds the eyes, nose and mouth.
2. **The face fingerprint:** [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface),
   a face-recognition network trained on millions of faces. It straightens the face using the
   eyes, nose and mouth, then describes it with 128 numbers. Photos of the same person get similar
   numbers, whatever the lighting or clothes.
3. **The house network:** a small neural network that `train.py` trains to guess the house from a
   fingerprint. It takes a few seconds. Rarer houses count extra, because there are far more
   Gryffindor pictures.

Both OpenCV models download once into `models/` (230 KB and 39 MB). Each comes from one fixed
version and is only used if its fingerprint (SHA-256) matches.

- **Look-alikes:** each character's typical fingerprint is the average of its faces'
  fingerprints. Your look-alikes are the characters whose fingerprint is closest to yours
  ("67% alike").
- **Cleaning:** `prepare.py` compares every face with the rest of its character. Faces that don't
  match (cartoons, LEGO figures, drawings, the wrong person) go to `data/rejected/`. On the
  current pictures it caught 44, nearly all of them real mistakes.
- **Honest testing (cross-validation):** `train.py` splits the characters into 5 groups. It
  guesses each group with a network trained only on the other 4, so every guess is about
  characters the network has never seen. Random guessing scores 25%.
- **Calibration:** on new faces the network is far too sure of itself, so `train.py` uses those
  honest guesses for two fixes that `predict.py` applies:
  - A *temperature* (about 7) softens the probabilities until they match how often it's really
    right.
  - A correction for the houses it still leans towards.

### Before and after the face fingerprints

The look-alikes used to come from a ResNet-18 fine-tuned to guess houses. Its features described
houses, not people. Tested on all 1,415 faces of 82 characters (random guessing: about 1%):

| Look-alike test | Before (ResNet) | After (SFace) |
|---|---|---|
| A real-life actor photo finds the right movie character | 14% | **86%** (top 3: 93%) |
| A film still finds its own character (with that still left out) | 57% | **83%** (top 3: 88%) |

For the house hint, three methods were compared with the same cross-validation:

| House guess on unseen characters | All faces | Film stills | Real-life photos |
|---|---|---|---|
| ResNet fine-tuned on face pixels (before) | 30% | 30% | 30% |
| Small network on SFace fingerprints (now) | 30% | 32% | 25% |
| Vote by your closest look-alikes | 25% | 28% | 19% |

These scores include the calibration's house correction. The house hint is a tie, so the game uses
the simpler option: the small network trains in seconds instead of minutes and uses the same
fingerprint as the look-alikes.

Useful options: `python train.py --folds 10` (more, smaller groups) and `--epochs 500`.

## Be honest with your friends 😉

Your face doesn't decide your house. On characters it has never seen, every method tried here
scores close to random guessing (25%): the ResNet on pixels, the network on face fingerprints,
and the look-alike vote (see the table above).

Adding 435 real-life photos of the actors didn't change that either. I trained models with and
without them, three times each, and tested them on the same 16 unseen characters. The scores
stayed between 21% and 30%, within the normal variation between runs.

So the face gives a hint and breaks ties, and your answers decide. The look-alikes are the part
of the face that really works: a photo of Tom Felton finds Draco. It's a game!

## Tests

```bash
pytest
```

The tests take a few seconds and don't need a trained model or the internet. They check:
- The question bank is fair, and broken questions give a clear message.
- Simulated students are sorted correctly and fairly: honest ones, random ones, ones who like
  nicer answers, and ones whose face points to the wrong house.
- The web server handles both good and bad requests.
- Cleaning, training and predicting work: strangers among a character's photos are caught,
  every character stays in one cross-validation group, the network learns houses it can tell
  apart, calibration tones down an overconfident network, and look-alikes come out in the right
  order.
- The hat's voice makes slower, deeper, evenly loud lines, each only once. This uses a fake
  speech engine, so the 350 MB model isn't needed.
- The picture sources, with made-up answers: only free licences and photos of just one person
  are kept, no photo is downloaded twice, every credit survives a stopped run, and the Roboflow
  import skips mosaics and cuts out the right characters from a made-up dataset zip.

## Files

```
app.py           the graphical game (a small local website)
web/             its page: index.html, style.css, app.js
voice.py         the hat's natural voice (Kokoro text-to-speech)
sorting_hat.py   the terminal game
hat_brain.py     the hat's thinking: questions, scores, tournament, decision
questions.json   the Sorting Hat's yes/no questions
common.py        shared code: houses, paths, face cropping, the network
scrape.py        step 1 – download character pictures
import_roboflow.py  optional – add the Roboflow film stills
prepare.py       step 2 – crop faces and clean out wrong ones
train.py         step 3 – face fingerprints, look-alikes, the house network
predict.py       step 4 – house from a photo only
tests/           the tests (run with: pytest)
requirements.txt / requirements-lock.txt   allowed versions / exact tested versions
data/            downloaded pictures, face crops and rejected crops (not in git)
models/          house network, face detector, face recognizer, voice model (not in git)
```

The pictures are © their owners (Warner Bros. and others). They are only used locally to train
this toy model and to show look-alikes. The actor photos from Wikimedia Commons are under the free
licences recorded in each folder's `sources.json`. The Roboflow dataset is listed under CC BY 4.0
(credit: Leanse Sangeerth), but its stills are frames from the films. `data/` and `models/` are in `.gitignore` so they are
never uploaded; please don't publish them.
