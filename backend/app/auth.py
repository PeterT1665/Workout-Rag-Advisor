"""Single-user password login with a signed, expiring session cookie. Standard library only."""
import hashlib
import hmac
import secrets
import time

COOKIE = "ta_session"
MAX_AGE = 180 * 24 * 3600  # 6 months: stay logged in on the phone


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${h.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, salt_hex, hash_hex = stored.split("$")
    except ValueError:
        return False
    if algo != "scrypt":
        return False
    candidate = hash_password(password, bytes.fromhex(salt_hex)).split("$")[2]
    return hmac.compare_digest(candidate, hash_hex)


def make_token(secret: str, now: float | None = None) -> str:
    expires = int((now or time.time()) + MAX_AGE)
    payload = f"peter.{expires}"
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def check_token(token: str | None, secret: str, now: float | None = None) -> bool:
    if not token or not secret:
        return False
    try:
        user, expires, sig = token.rsplit(".", 2)
    except ValueError:
        return False
    good = hmac.new(secret.encode(), f"{user}.{expires}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, good):
        return False
    return int(expires) > (now or time.time())


class Throttle:
    """Slows down password guessing: max 10 failures per IP per 15 minutes."""

    def __init__(self, limit: int = 10, window: int = 900):
        self.limit, self.window = limit, window
        self.fails: dict[str, list[float]] = {}

    def blocked(self, ip: str) -> bool:
        now = time.time()
        recent = [t for t in self.fails.get(ip, []) if now - t < self.window]
        self.fails[ip] = recent
        return len(recent) >= self.limit

    def fail(self, ip: str) -> None:
        self.fails.setdefault(ip, []).append(time.time())
