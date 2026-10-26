"""Settings from environment variables (set in /etc/training-agent.env on the server)."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    password_hash: str  # scrypt$salt_hex$hash_hex, made with `python -m app.hashpw`
    secret: str  # signs the session cookie
    db_path: str
    secure_cookie: bool  # True behind HTTPS


def load() -> Settings:
    return Settings(
        password_hash=os.environ.get("TA_PASSWORD_HASH", ""),
        secret=os.environ.get("TA_SECRET", ""),
        db_path=os.environ.get("TA_DB", "training.db"),
        secure_cookie=os.environ.get("TA_SECURE_COOKIE", "1") == "1",
    )
