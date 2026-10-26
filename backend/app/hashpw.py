"""Make a password hash for TA_PASSWORD_HASH:  python -m app.hashpw  (asks for the password, never echoes it)."""
import getpass

from .auth import hash_password

if __name__ == "__main__":
    pw = getpass.getpass("New app password: ")
    if len(pw) < 8:
        raise SystemExit("Use at least 8 characters.")
    if pw != getpass.getpass("Repeat: "):
        raise SystemExit("Passwords did not match.")
    print(hash_password(pw))
