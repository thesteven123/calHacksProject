"""Bounded ASGI upload buffering tests (FastAPI is provided by the serving image)."""

import asyncio

from serve.asgi_apps import MAX_IMAGE_BYTES, _read_upload


class _Upload:
    def __init__(self, data: bytes, size: int | None):
        self.data = data
        self.size = size
        self.read_sizes: list[int] = []

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return self.data[:size]


def test_known_oversize_is_rejected_without_reading_payload():
    upload = _Upload(b"not materialized", MAX_IMAGE_BYTES + 1)

    data, error = asyncio.run(_read_upload(upload, "large.jpg"))

    assert data == b""
    assert error is not None and error.status_code == 413
    assert upload.read_sizes == []


def test_unknown_size_read_is_bounded_and_oversize_rejected():
    upload = _Upload(b"x" * (MAX_IMAGE_BYTES + 1), None)

    data, error = asyncio.run(_read_upload(upload, "large.jpg"))

    assert data == b""
    assert error is not None and error.status_code == 413
    assert upload.read_sizes == [MAX_IMAGE_BYTES + 1]


def test_supported_upload_within_limit_is_returned():
    upload = _Upload(b"valid image bytes", 17)

    data, error = asyncio.run(_read_upload(upload, "sample.jpg"))

    assert data == b"valid image bytes"
    assert error is None
    assert upload.read_sizes == [MAX_IMAGE_BYTES + 1]


def test_media_route_rejects_sibling_traversal(monkeypatch, tmp_path):
    from starlette.testclient import TestClient

    import serve.asgi_apps as a

    media_root = tmp_path / "media"
    media_root.mkdir()
    (media_root / "ok.jpg").write_bytes(b"img")
    sibling = tmp_path / "media-private"
    sibling.mkdir()
    (sibling / "secret").write_bytes(b"x")
    monkeypatch.setattr(a, "MEDIA_ROOT", media_root)
    monkeypatch.setenv("DEMO_PASSCODE", "pw")
    monkeypatch.setenv("DFD_API_KEY", "k")
    app = a.build_web(None)
    import hashlib
    import hmac as _h
    import os
    import time

    ts = int(time.time())
    sig = _h.new(os.environ["DFD_API_KEY"].encode(), str(ts).encode(), hashlib.sha256).hexdigest()
    c = TestClient(app)
    c.cookies.set("dfd_session", f"{ts}.{sig}")
    # httpx normalizes literal '..' segments client-side, so exercise the traversal via an
    # encoded dot segment that reaches the route handler unresolved
    assert c.get("/media/..%2F..%2Fmedia-private%2Fsecret").status_code == 404
    assert c.get("/media/ok.jpg").status_code == 200
    # the resolve+is_relative_to guard itself must reject a sibling dir name sharing the prefix
    assert not (media_root.parent / "media-private" / "secret").resolve().is_relative_to(media_root.resolve())
