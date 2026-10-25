"""ASGI app factories for the dfd-demo backend (imported inside Modal functions, fastapi present)."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

COOKIE_NAME = "dfd_session"
COOKIE_TTL_S = 12 * 3600
MEDIA_ROOT = Path("/demo/media")
FEED_ROOT = Path("/demo/feed")
STATIC = Path("/app/static")

MODELS_INFO = {
    "specialists": [
        {"name": "general", "title": "General AI-image detector",
         "checkpoint": "general_all / full7_dinov2_conformal",
         "trained_on": "Tiny-GenImage LOGO rotations (7 known generators)",
         "metrics": {"unseen_generator_auc": 0.915, "tpr@1%fpr": 0.42,
                     "weakest_generator": "Midjourney (AUC 0.741)"},
         "limitations": "Calibrated on still images only; social-media cascades degrade it."},
        {"name": "face", "title": "AI-generated face detector",
         "checkpoint": "face_t2i_b3",
         "trained_on": "SFHQ-T2I text-to-image faces (train split)",
         "metrics": {"held_out_flux_auc": 0.988, "fairface_real_fpr": 0.0058},
         "limitations": "Not a face-swap detector: WildDeepfake in-the-wild fakes score at chance."},
    ],
    "fusion": "fake if any specialist says fake; real only if all say real; else review",
    "notes": [
        "Fusion can only raise the false-positive rate (up to ~the sum of specialist FPRs).",
        "Video verdicts are a per-keyframe heuristic, not clip-calibrated.",
        "Research demo only; uncertain cases go to human review.",
    ],
}


def _sign(ts: int) -> str:
    return hmac.new(os.environ["DFD_API_KEY"].encode(), str(ts).encode(), hashlib.sha256).hexdigest()


def _session_ok(cookie: str | None) -> bool:
    if not cookie or "." not in cookie:
        return False
    ts, sig = cookie.rsplit(".", 1)
    return (ts.isdigit() and time.time() - int(ts) < COOKIE_TTL_S
            and hmac.compare_digest(sig, _sign(int(ts))))


def _validate_upload(data: bytes, filename: str):
    ext = Path(filename or "").suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".webp"}:
        if len(data) > 15 * 1024 * 1024:
            return "image", JSONResponse({"error": "image exceeds 15 MB"}, status_code=413)
        return "image", None
    if ext in {".mp4", ".mov", ".webm"}:
        if len(data) > 50 * 1024 * 1024:
            return "video", JSONResponse({"error": "video exceeds 50 MB"}, status_code=413)
        return "video", None
    return None, JSONResponse({"error": f"unsupported media type {ext!r}"}, status_code=415)


def _detect_call(data: bytes, filename: str, detector_cls):
    _, err = _validate_upload(data, filename)
    if err is not None:
        return err
    try:
        return detector_cls().detect.remote(data, filename)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"detector error: {type(e).__name__}: {e}"}, status_code=502)


def build_api(detector_cls) -> FastAPI:
    app = FastAPI()
    key = os.environ["DFD_API_KEY"]

    def ok(request: Request) -> bool:
        auth = request.headers.get("authorization", "")
        return auth.startswith("Bearer ") and hmac.compare_digest(auth[7:], key)

    @app.get("/v1/health")
    def health():
        return {"ok": True}

    @app.get("/v1/models")
    def models(request: Request):
        if not ok(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return MODELS_INFO

    @app.post("/v1/detect")
    async def detect(request: Request):
        if not ok(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        form = await request.form()
        f = form.get("file")
        if f is None:
            return JSONResponse({"error": "multipart field 'file' required"}, status_code=400)
        return _detect_call(await f.read(), f.filename or "", detector_cls)

    return app


def build_web(detector_cls, api_fn=None) -> FastAPI:
    app = FastAPI()

    def authed(request: Request) -> bool:
        return _session_ok(request.cookies.get(COOKIE_NAME))

    def need_auth(request: Request):
        return None if authed(request) else JSONResponse({"error": "unauthorized"}, status_code=401)

    @app.post("/api/login")
    async def login(request: Request):
        body = await request.json()
        if not hmac.compare_digest(str(body.get("passcode", "")), os.environ["DEMO_PASSCODE"]):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        ts = int(time.time())
        resp = JSONResponse({"ok": True})
        resp.set_cookie(COOKIE_NAME, f"{ts}.{_sign(ts)}", max_age=COOKIE_TTL_S,
                        httponly=True, secure=True, samesite="lax")
        return resp

    @app.get("/api/session")
    def session(request: Request):
        if not authed(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return {"ok": True}

    @app.post("/api/detect")
    async def api_detect(request: Request):
        if (err := need_auth(request)) is not None:
            return err
        form = await request.form()
        f = form.get("file")
        if f is None:
            return JSONResponse({"error": "multipart field 'file' required"}, status_code=400)
        return _detect_call(await f.read(), f.filename or "", detector_cls)

    @app.get("/api/models")
    def api_models(request: Request):
        if (err := need_auth(request)) is not None:
            return err
        out = dict(MODELS_INFO)
        try:
            out["api_url"] = api_fn.get_web_url() if api_fn is not None else None
        except Exception:
            pass
        return out

    @app.get("/api/feed/{name}")
    def feed(name: str, request: Request):
        if (err := need_auth(request)) is not None:
            return err
        if name not in ("instagram", "tiktok"):
            return JSONResponse({"error": "unknown feed"}, status_code=404)
        p = FEED_ROOT / f"{name}.json"
        if not p.exists():
            return JSONResponse({"error": "feed not seeded"}, status_code=404)
        return json.loads(p.read_text())

    @app.get("/media/{path:path}")
    def media(path: str, request: Request):
        if (err := need_auth(request)) is not None:
            return err
        target = (MEDIA_ROOT / path).resolve()
        if not str(target).startswith(str(MEDIA_ROOT.resolve())) or not target.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(target)

    @app.get("/{path:path}")
    def spa(path: str):
        if path.startswith("api/"):
            return JSONResponse({"error": "not found"}, status_code=404)
        target = (STATIC / path).resolve()
        if str(target).startswith(str(STATIC.resolve())) and target.is_file():
            return FileResponse(target)
        index = STATIC / "index.html"
        return FileResponse(index) if index.exists() else HTMLResponse("<h1>dfd-demo</h1>")

    return app
