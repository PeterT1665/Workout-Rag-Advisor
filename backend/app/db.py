"""Tiny key-value store for synced app state (SQLite, one file)."""
import json
import sqlite3
import time

SYNC_KEYS = {"history", "games", "settings", "quotes"}


def connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at INTEGER NOT NULL)")
    con.commit()
    return con


def all_state(con: sqlite3.Connection) -> dict:
    rows = con.execute("SELECT key, value, updated_at FROM state").fetchall()
    return {k: {"value": json.loads(v), "updated_at": u} for k, v, u in rows}


def _history_key(e: dict) -> str:
    return f"{e.get('date')}|{e.get('id')}|{e.get('minutes')}|{e.get('title')}"


def merge_history(server: list, incoming: list) -> list:
    """Union of sessions from both devices; for the same session, the copy with a check-in wins."""
    by_key: dict[str, dict] = {}
    for e in list(server) + list(incoming):
        k = _history_key(e)
        prev = by_key.get(k)
        if prev is None or (e.get("load") and not prev.get("load")) or len(e) > len(prev):
            by_key[k] = e
    return sorted(by_key.values(), key=lambda e: str(e.get("date")))[-500:]


def put(con: sqlite3.Connection, key: str, value, updated_at: int) -> dict:
    row = con.execute("SELECT value, updated_at FROM state WHERE key = ?", (key,)).fetchone()
    if key == "history" and row:
        value = merge_history(json.loads(row[0]), value)
        updated_at = max(updated_at, row[1], int(time.time() * 1000))
    elif row and row[1] >= updated_at:
        return {"value": json.loads(row[0]), "updated_at": row[1]}  # server copy is newer: keep it
    con.execute(
        "INSERT INTO state (key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        (key, json.dumps(value), updated_at),
    )
    con.commit()
    return {"value": value, "updated_at": updated_at}
