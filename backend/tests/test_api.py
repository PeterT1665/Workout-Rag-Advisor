import time

import pytest
from fastapi.testclient import TestClient

from app import auth
from app.config import Settings
from app.db import merge_history
from app.main import create_app

SECRET = "x" * 40


@pytest.fixture()
def client(tmp_path):
    s = Settings(password_hash=auth.hash_password("correct horse"), secret=SECRET, db_path=str(tmp_path / "t.db"), secure_cookie=False)
    return TestClient(create_app(s))


def login(c):
    return c.post("/api/auth/login", json={"password": "correct horse"})


def test_health_is_public(client):
    assert client.get("/api/health").json() == {"ok": True}


def test_everything_else_needs_login(client):
    for path in ["/api/auth/me", "/api/auth/check", "/api/state"]:
        assert client.get(path).status_code == 401
    assert client.put("/api/state/games", json={"value": [], "updated_at": 1}).status_code == 401


def test_wrong_password_then_right(client):
    assert client.post("/api/auth/login", json={"password": "nope"}).status_code == 401
    assert login(client).status_code == 200
    assert client.get("/api/auth/me").json() == {"user": "peter"}
    assert client.get("/api/auth/check").status_code == 204


def test_throttle_blocks_guessing(client):
    for _ in range(10):
        client.post("/api/auth/login", json={"password": "nope"})
    assert client.post("/api/auth/login", json={"password": "correct horse"}).status_code == 429


def test_tokens_expire_and_cannot_be_forged():
    tok = auth.make_token(SECRET, now=1000)
    assert auth.check_token(tok, SECRET, now=1001)
    assert not auth.check_token(tok, SECRET, now=1000 + auth.MAX_AGE + 1)
    assert not auth.check_token(tok, "y" * 40, now=1001)
    assert not auth.check_token(tok.replace("peter", "admin"), SECRET, now=1001)


def test_state_last_writer_wins(client):
    login(client)
    client.put("/api/state/games", json={"value": [{"day": 2}], "updated_at": 200})
    old = client.put("/api/state/games", json={"value": [], "updated_at": 100}).json()
    assert old["value"] == [{"day": 2}]  # older write ignored
    assert client.get("/api/state").json()["games"]["value"] == [{"day": 2}]
    assert client.put("/api/state/secrets", json={"value": 1, "updated_at": 1}).status_code == 404


def test_history_merges_sessions_from_both_devices(client):
    login(client)
    phone = [{"id": "a", "title": "A", "date": "2026-10-26", "minutes": 60, "rpe": 7, "load": 420}]
    laptop = [{"id": "b", "title": "B", "date": "2026-10-27", "minutes": 20}]
    client.put("/api/state/history", json={"value": phone, "updated_at": int(time.time() * 1000)})
    merged = client.put("/api/state/history", json={"value": laptop, "updated_at": 1}).json()["value"]
    assert [e["id"] for e in merged] == ["a", "b"]


def test_checked_in_copy_wins():
    plain = {"id": "a", "title": "A", "date": "d", "minutes": 60}
    checked = {**plain, "rpe": 7, "load": 420}
    assert merge_history([plain], [checked]) == [checked]
    assert merge_history([checked], [plain]) == [checked]


def test_refuses_to_start_without_config(tmp_path):
    with pytest.raises(RuntimeError):
        create_app(Settings(password_hash="", secret="", db_path=str(tmp_path / "x.db"), secure_cookie=False))
