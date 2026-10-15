"""The Sorting Hat - graphical version. It runs as a small website on your own computer.

    python app.py                 # opens http://127.0.0.1:5050 in your browser
    python app.py --port 8000 --no-browser

The page (web/index.html, web/style.css, web/app.js) shows the hat and talks to this server,
which does the thinking and the talking:
    POST /api/face    photo -> house probabilities, look-alike characters and the face crop
    POST /api/turn    the answers so far -> the hat's next question (or "done")
    POST /api/result  the final house
    GET  /api/voice   a line of text -> the hat saying it (a WAV file, see voice.py)
The thinking itself is shared with the terminal game: see hat_brain.py and predict.py.
"""
import argparse
import base64
import io
import random
import threading
import webbrowser

from flask import Flask, Response, jsonify, request, send_from_directory

import hat_brain as brain
import voice
from common import FACES_DIR, HOUSES, MODEL_PATH, ROOT

app = Flask(__name__, static_folder=str(ROOT / "web"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # photos up to 25 MB

QUESTIONS = brain.load_questions()
QUESTIONS_BY_ID = {q["id"]: q for q in QUESTIONS}
rng = random.Random()
classifier = None             # the neural network (None if it hasn't been trained yet)
hat_voice = None              # the neural voice (None if it couldn't load: the page uses the browser's voices)
lock = threading.Lock()       # one face at a time: the face detector isn't thread-safe


def warm_up(lines, voice_id=voice.DEFAULT_VOICE):
    """Say these lines once in the background, so they're ready the moment the hat needs them."""
    def run():
        for line in lines:
            try:
                hat_voice.wav(line, voice_id)
            except Exception:
                app.logger.exception("Could not prepare a line of speech")
                return
    if hat_voice is not None:
        threading.Thread(target=run, daemon=True).start()


def data_url(img, size):
    """A PIL image as a small JPEG that the web page can show directly."""
    img = img.convert("RGB")
    img.thumbnail((size, size))
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def character_picture(name, house):
    """A face crop of a movie character from data/faces, for the look-alike cards (if it's there)."""
    pictures = sorted((FACES_DIR / house / name.replace(" ", "_")).glob("*.jpg"))
    if not pictures:
        return None
    from PIL import Image
    return data_url(Image.open(pictures[0]), 120)


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/info")
def info():
    voices = [{"id": key, "name": name} for key, name in voice.VOICES.items()] if hat_voice else []
    return jsonify(houses=HOUSES, can_see=classifier is not None, voices=voices, default_voice=voice.DEFAULT_VOICE)


@app.get("/api/voice")
def speak_line():
    text, voice_id = request.args.get("text", "").strip(), request.args.get("voice", voice.DEFAULT_VOICE)
    if hat_voice is None:
        return jsonify(error="The hat's voice isn't available."), 404
    if not text or len(text) > 400 or voice_id not in voice.VOICES:
        return jsonify(error="The hat can't say that."), 400
    return Response(hat_voice.wav(text, voice_id), mimetype="audio/wav", headers={"Cache-Control": "max-age=86400"})


@app.post("/api/warm")
def warm():
    """The page sends its own lines once, so they're ready before the hat needs them."""
    data = request.get_json(silent=True) or {}
    voice_id = data.get("voice", voice.DEFAULT_VOICE)
    lines = [line for line in data.get("lines", []) if isinstance(line, str) and 0 < len(line) <= 400][:50]
    if voice_id in voice.VOICES:
        warm_up(lines, voice_id)
    return jsonify(ok=True)


@app.post("/api/face")
def read_face():
    if classifier is None:
        return jsonify(error="The hat has no eyes yet: run python train.py first."), 400
    try:
        photo = base64.b64decode((request.get_json(silent=True) or {})["image"].split(",", 1)[-1])
        with lock:
            result = classifier.predict(io.BytesIO(photo))
    except Exception:
        app.logger.exception("Could not read the photo")  # the full error goes to the terminal
        return jsonify(error="That doesn't look like a picture."), 400
    return jsonify(
        probs=result["probs"],
        face_found=result["face_found"],
        face=data_url(result["face"], 320),
        lookalikes=[{"name": name, "house": house, "similarity": round(similarity, 3),
                     "picture": character_picture(name, house)}
                    for name, house, similarity in result["lookalikes"]],
    )


@app.post("/api/turn")
def turn():
    """Takes the game so far (face, asked, points, and maybe a new answer) and returns what's next."""
    data = request.get_json(silent=True) or {}
    face, asked, points = data.get("face"), list(data.get("asked", [])), list(data.get("points", [0] * len(HOUSES)))
    old_probs = brain.combine(face, points)
    answer = data.get("answer")
    if answer:
        question = QUESTIONS_BY_ID.get(answer.get("id"))
        if question is None or question["id"] in asked or not isinstance(answer.get("yes"), bool):
            return jsonify(error="The hat doesn't know that question, or it was already answered."), 400
        points = brain.add_answer(points, question, answer["yes"])
        asked.append(question["id"])
    probs = brain.combine(face, points)
    reply = {"asked": asked, "points": points, "probs": dict(zip(HOUSES, probs))}
    if brain.done_asking(probs, len(asked), n_questions=len(QUESTIONS)):
        reply["tie"] = brain.near_tie(probs, rng)
    else:
        question = brain.next_question(QUESTIONS, asked, probs, rng)
        reply["question"] = {"id": question["id"], "text": question["text"], "number": len(asked) + 1}
        if answer:
            reply["whisper"] = brain.whisper(old_probs, probs, rng)
    return jsonify(reply)


@app.post("/api/result")
def result():
    data = request.get_json(silent=True) or {}
    if len(data.get("points", [])) != len(HOUSES):
        return jsonify(error="The hat needs the points of all four houses."), 400
    probs = brain.combine(data.get("face"), data["points"])
    house, final = brain.decide(probs, data.get("prefer"))
    return jsonify(house=house, probs=final)


def main():
    global classifier, hat_voice
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")
    args = parser.parse_args()

    if MODEL_PATH.exists():
        print("Waking up the Sorting Hat (loading the neural network)...")
        from predict import HouseClassifier
        classifier = HouseClassifier()
    else:
        print("No trained model yet (run python train.py) - the hat will only ask questions.")

    try:
        hat_voice = voice.HatVoice()
        warm_up([q["text"] for q in QUESTIONS] + list(brain.WHISPERS.values()) + brain.MURMURS)
    except Exception as error:  # e.g. kokoro-onnx not installed, or no internet for the first download
        print(f"The hat's natural voice isn't available ({error}) - using the browser's voices instead.")

    url = f"http://127.0.0.1:{args.port}"
    print(f"\n  The Sorting Hat is ready: {url}\n  (press Ctrl+C to stop)\n")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    app.run(host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
