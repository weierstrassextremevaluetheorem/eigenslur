from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert isinstance(payload["labeler_mode"], str)
    assert isinstance(payload["llm_configured"], bool)


def test_frontend_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "EigenSlur Frontend" in response.text


def test_static_css_served() -> None:
    response = client.get("/static/styles.css")
    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]


def test_score_term() -> None:
    response = client.post(
        "/score/term",
        json={
            "term": "example",
            "contexts": [
                "you are such an example",
                "they quoted 'example' in class",
                "we reclaimed example among ourselves",
            ],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["term"] == "example"
    assert 0.0 <= payload["score"] <= 1.0
    assert payload["band"] in {"monitor", "review", "block"}
    assert payload["warnings"] == []


def test_score_term_warns_when_term_absent_from_contexts() -> None:
    response = client.post(
        "/score/term",
        json={
            "term": "example",
            "contexts": [
                "This sentence has no target token.",
                "Completely unrelated discussion text.",
            ],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["term"] == "example"
    assert payload["warnings"]
    assert "not found" in payload["warnings"][0].lower()


def test_score_text_term_detection() -> None:
    response = client.post(
        "/score/text",
        json={
            "text": "You are an example. We reclaimed example in our group.",
            "candidate_terms": ["example", "missing"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["terms_found"] == 1
    assert payload["results"][0]["term"] == "example"


def test_score_text_hyphenated_term_detection() -> None:
    response = client.post(
        "/score/text",
        json={
            "text": "That history-term appears again in this context.",
            "candidate_terms": ["history-term"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["terms_found"] == 1
    assert payload["results"][0]["term"] == "history-term"


def test_term_history_after_scoring() -> None:
    scored = client.post(
        "/score/term",
        json={
            "term": "history-term",
            "contexts": [
                "you are a history-term",
                "we reclaimed history-term in-group",
            ],
        },
    )
    assert scored.status_code == 200

    history = client.get("/term/history-term/history?limit=10")
    assert history.status_code == 200
    payload = history.json()
    assert payload["term"] == "history-term"
    assert payload["count"] >= 1
    assert payload["history"][0]["term"] == "history-term"


def test_feedback_endpoint() -> None:
    response = client.post(
        "/feedback",
        json={
            "term": "example",
            "feedback_type": "false_positive",
            "proposed_band": "monitor",
            "proposed_score": 0.2,
            "notes": "This was quoted educational context.",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "accepted"
    assert payload["feedback_id"] >= 1
