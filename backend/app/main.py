"""Sport Personal Coach API: login, a gate for private photos, and phone/laptop sync."""
import asyncio
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

from . import auth, db
from .config import Settings, load


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or load()
    if not cfg.password_hash or len(cfg.secret) < 32:
        raise RuntimeError("Set TA_PASSWORD_HASH and TA_SECRET (32+ chars) before starting.")
    app = FastAPI(title="Sport Personal Coach", docs_url=None, redoc_url=None, openapi_url=None)
    con = db.connect(cfg.db_path)
    throttle = auth.Throttle()

    def require_login(request: Request) -> None:
        if not auth.check_token(request.cookies.get(auth.COOKIE), cfg.secret):
            raise HTTPException(401, "Not logged in")

    class Login(BaseModel):
        password: str

    class Put(BaseModel):
        value: Any
        updated_at: int

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.post("/api/auth/login")
    async def login(body: Login, request: Request, response: Response):
        ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0].strip()
        if throttle.blocked(ip):
            raise HTTPException(429, "Too many attempts. Try again in 15 minutes.")
        if not auth.verify_password(body.password, cfg.password_hash):
            throttle.fail(ip)
            await asyncio.sleep(0.5)
            raise HTTPException(401, "Wrong password")
        response.set_cookie(auth.COOKIE, auth.make_token(cfg.secret), max_age=auth.MAX_AGE,
                            httponly=True, secure=cfg.secure_cookie, samesite="lax", path="/")
        return {"ok": True}

    @app.post("/api/auth/logout")
    def logout(response: Response):
        response.delete_cookie(auth.COOKIE, path="/")
        return {"ok": True}

    @app.get("/api/auth/me", dependencies=[Depends(require_login)])
    def me():
        return {"user": "peter"}

    # Caddy asks this before serving /personal/* (private photos)
    @app.get("/api/auth/check", dependencies=[Depends(require_login)], status_code=204)
    def check():
        return Response(status_code=204)

    @app.get("/api/state", dependencies=[Depends(require_login)])
    def get_state():
        return db.all_state(con)

    @app.put("/api/state/{key}", dependencies=[Depends(require_login)])
    def put_state(key: str, body: Put):
        if key not in db.SYNC_KEYS:
            raise HTTPException(404, "Unknown key")
        return db.put(con, key, body.value, body.updated_at)

    return app


def app_factory() -> FastAPI:  # uvicorn --factory app.main:app_factory
    return create_app()
