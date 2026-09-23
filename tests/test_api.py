"""HTTP-level tests for the FastAPI repair service."""

from __future__ import annotations

import random

from fastapi.testclient import TestClient

from app.main import app
from app.repair import ALPHABET, MATCH_OPEN_TO_CLOSE

client = TestClient(app)


def token(ch: str, locked: bool = False) -> dict:
    return {"char": ch, "locked": locked}


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------

def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_repair_basic():
    response = client.post("/repair", json={"tokens": [token("("), token("]")]})
    assert response.status_code == 200
    assert response.json() == {
        "status": "OK",
        "repaired": "()",
        "pairs": [[0, 1]],
        "changes": [{"index": 1, "before": "]", "after": ")"}],
        "cost": 1,
    }


def test_repair_already_balanced_all_locked():
    payload = {
        "tokens": [
            token("(", True),
            token("[", True),
            token("]", True),
            token(")", True),
        ]
    }
    body = client.post("/repair", json=payload).json()
    assert body["status"] == "OK"
    assert body["repaired"] == "([])"
    assert body["changes"] == []
    assert body["cost"] == 0


def test_no_repair_response():
    response = client.post(
        "/repair",
        json={"tokens": [token("(", True), token("(", True)]},
    )
    assert response.status_code == 200
    assert response.json() == {
        "status": "NO_REPAIR",
        "repaired": None,
        "pairs": None,
        "changes": None,
        "cost": None,
    }


def test_length_bounds_accepted():
    for n in (2, 160):
        response = client.post(
            "/repair",
            json={"tokens": [token("(") for _ in range(n)]},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "OK"


# ---------------------------------------------------------------------------
# full response is verifiable position by position
# ---------------------------------------------------------------------------

def assert_body_verifiable(tokens, body):
    repaired = body["repaired"]
    n = len(tokens)
    assert len(repaired) == n

    # locked positions survive unchanged
    for i, t in enumerate(tokens):
        if t["locked"]:
            assert repaired[i] == t["char"]

    # changes describe exactly every differing position
    changed = {c["index"] for c in body["changes"]}
    expected_changed = {
        i for i, t in enumerate(tokens) if repaired[i] != t["char"]
    }
    assert changed == expected_changed
    assert body["cost"] == len(changed)
    for c in body["changes"]:
        assert tokens[c["index"]]["char"] == c["before"]
        assert repaired[c["index"]] == c["after"]
        assert tokens[c["index"]]["locked"] is False

    # pairs partition every index once, indices ordered and types match
    pairs = body["pairs"]
    flat = sorted(i for pair in pairs for i in pair)
    assert flat == list(range(n))
    for i, j in pairs:
        assert i < j
        assert repaired[j] == MATCH_OPEN_TO_CLOSE[repaired[i]]

    # the pair set really describes a balanced nesting
    stack = []
    for i, ch in enumerate(repaired):
        if ch in "([{":
            stack.append(i)
        else:
            opener = stack.pop()
            assert [opener, i] in pairs


def test_random_responses_are_verifiable():
    rng = random.Random(31415)
    for _ in range(60):
        n = rng.choice([2, 4, 6, 8, 16, 32])
        tokens = [
            token(rng.choice(ALPHABET), rng.random() < 0.4) for _ in range(n)
        ]
        body = client.post("/repair", json={"tokens": tokens}).json()
        if body["status"] == "NO_REPAIR":
            continue
        assert_body_verifiable(tokens, body)


# ---------------------------------------------------------------------------
# 422 validation
# ---------------------------------------------------------------------------

BAD_PAYLOADS = [
    # extra field inside a token
    {"tokens": [{"char": "(", "locked": False, "who": 1}, {"char": ")", "locked": False}]},
    # extra field on the request
    {"tokens": [token("("), token(")")], "extra": True},
    # illegal characters
    {"tokens": [token("x"), token(")")]},
    {"tokens": [token("(("), token(")")]},
    {"tokens": [token(""), token(")")]},
    # char of wrong type
    {"tokens": [{"char": 1, "locked": False}, token(")")]},
    # locked of wrong type / non-strict coercion must fail
    {"tokens": [{"char": "(", "locked": "true"}, {"char": ")", "locked": False}]},
    {"tokens": [{"char": "(", "locked": 1}, {"char": ")", "locked": False}]},
    {"tokens": [{"char": "(", "locked": 0}, {"char": ")", "locked": False}]},
    # missing fields
    {"tokens": [{"locked": False}, {"char": ")", "locked": False}]},
    {"tokens": [{"char": "("}, {"char": ")", "locked": False}]},
    # token count out of range or odd
    {"tokens": []},
    {"tokens": [token("(")]},
    {"tokens": [token("("), token(")"), token("[")]},
    {"tokens": [token("(") for _ in range(162)]},
    # tokens not a list / body not an object
    {"tokens": {"char": "(", "locked": False}},
]


def test_bad_payloads_rejected():
    for payload in BAD_PAYLOADS:
        response = client.post("/repair", json=payload)
        assert response.status_code == 422, payload


def test_non_json_body_rejected():
    response = client.post("/repair", content="not-json",
                           headers={"content-type": "application/json"})
    assert response.status_code == 422
