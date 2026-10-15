"""The web game's server, with a fake neural network so the tests run in seconds."""
import base64
import io

import pytest
from PIL import Image

import app as web


class FakeClassifier:
    """Stands in for predict.HouseClassifier: no trained model needed."""

    def predict(self, image):
        Image.open(image).verify()  # a junk upload fails, just like with the real network
        return {"probs": {"Gryffindor": 0.4, "Hufflepuff": 0.2, "Ravenclaw": 0.2, "Slytherin": 0.2},
                "lookalikes": [("Luna Lovegood", "Ravenclaw", 0.9)], "face_found": True,
                "face": Image.new("RGB", (64, 64), "white")}


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
