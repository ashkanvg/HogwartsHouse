"""The web game's server, with a fake neural network and voice so the tests run in seconds."""
import base64
import io
import time

import pytest
from PIL import Image

import app as web
import voice


class FakeClassifier:
    """Stands in for predict.HouseClassifier: no trained model needed."""

    def predict(self, image):
        Image.open(image).verify()  # a junk upload fails, just like with the real network
        return {"probs": {"Gryffindor": 0.4, "Hufflepuff": 0.2, "Ravenclaw": 0.2, "Slytherin": 0.2},
                "lookalikes": [("Luna Lovegood", "Ravenclaw", 0.9)], "face_found": True,
                "face": Image.new("RGB", (64, 64), "white")}


class FakeVoice:
    """Stands in for voice.HatVoice: no 350 MB model needed."""

    def __init__(self):
        self.said = []

    def wav(self, text, voice_id=voice.DEFAULT_VOICE):
        if voice_id not in voice.VOICES:
            raise ValueError(voice_id)
        self.said.append(text)
        return b"RIFF-fake-wav"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(web, "classifier", FakeClassifier())
    return web.app.test_client()


def photo():
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "blue").save(buffer, "JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_the_page_and_its_files_load(client):
    for url in ("/", "/static/app.js", "/static/style.css"):
        assert client.get(url).status_code == 200, url


def test_reading_a_face(client):
    reply = client.post("/api/face", json={"image": photo()}).get_json()
    assert reply["face_found"] and reply["face"].startswith("data:image/jpeg")
    assert reply["lookalikes"][0]["name"] == "Luna Lovegood"


def test_a_junk_upload_is_refused_politely(client):
    junk = "data:image/jpeg;base64," + base64.b64encode(b"not a picture").decode()
    reply = client.post("/api/face", json={"image": junk})
    assert reply.status_code == 400 and "error" in reply.get_json()


def test_a_whole_game(client):
    state = {"face": [0.4, 0.2, 0.2, 0.2], "asked": [], "points": [0, 0, 0, 0], "answer": None}
    for _ in range(10):
        reply = client.post("/api/turn", json=state).get_json()
        state.update(asked=reply["asked"], points=reply["points"])
        if "question" not in reply:
            break
        state["answer"] = {"id": reply["question"]["id"], "yes": True}
    assert 3 <= len(state["asked"]) <= 5
    result = client.post("/api/result", json={"face": state["face"], "points": state["points"]}).get_json()
    assert result["house"] in result["probs"]


@pytest.mark.parametrize("game", [
    {"asked": [], "points": [0, 0, 0, 0], "answer": {"id": "no-such-question", "yes": True}},
    {"asked": [], "points": [0, 0, 0, 0], "answer": {"id": "troll", "yes": "maybe"}},
    {"asked": ["troll"], "points": [2, 0, 0, 0], "answer": {"id": "troll", "yes": True}},  # answered twice
])
def test_bad_answers_get_a_clear_error_instead_of_a_crash(client, game):
    reply = client.post("/api/turn", json={"face": None, **game})
    assert reply.status_code == 400 and "error" in reply.get_json()


def test_a_result_needs_all_four_houses(client):
    assert client.post("/api/result", json={"points": [1, 2]}).status_code == 400


def test_the_hat_speaks_with_its_natural_voice(client, monkeypatch):
    monkeypatch.setattr(web, "hat_voice", FakeVoice())
    voices = client.get("/api/info").get_json()["voices"]
    assert voices[0]["id"] == voice.DEFAULT_VOICE
    reply = client.get("/api/voice", query_string={"text": "Hello", "voice": "bm_fable"})
    assert reply.status_code == 200 and reply.mimetype == "audio/wav"


@pytest.mark.parametrize("query", [{"text": ""}, {"text": "x" * 401}, {"text": "Hi", "voice": "bm_nobody"}])
def test_the_hat_refuses_strange_requests_to_speak(client, monkeypatch, query):
    monkeypatch.setattr(web, "hat_voice", FakeVoice())
    assert client.get("/api/voice", query_string=query).status_code == 400


def test_without_its_natural_voice_the_page_uses_the_browser_voices(client):
    assert client.get("/api/info").get_json()["voices"] == []
    assert client.get("/api/voice", query_string={"text": "Hello"}).status_code == 404


def test_the_page_can_ask_for_lines_to_be_prepared(client, monkeypatch):
    fake = FakeVoice()
    monkeypatch.setattr(web, "hat_voice", fake)
    lines = ["Hello", 42, "x" * 500]  # only real, short lines are prepared
    assert client.post("/api/warm", json={"voice": voice.DEFAULT_VOICE, "lines": lines}).get_json()["ok"]
    for _ in range(100):  # it happens in the background
        if fake.said:
            break
        time.sleep(0.01)
    assert fake.said == ["Hello"]
