"""
A slow face analysis must not freeze the API (found by flaky E2E runs).

``analyze_face`` and the photo-registration route are ``async def``. Their
landmark detection is CPU-bound; run directly on the event loop it made every
other request wait, so a trivial language lookup took as long as the analysis.
Both requests below share ONE event loop (httpx's ASGI transport), which is the
only setup in which blocking is visible: Starlette's TestClient gives each call
its own loop and would never notice.
"""

import asyncio
import io
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import httpx
from PIL import Image

import app as app_module
import avatar_store
from avatar_store import AvatarStore
from vision_fixtures import FakeFaceEngine, gradient_image

SLOW_SECONDS = 1.0


class SlowEngine(FakeFaceEngine):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.entered_at = 0.0

    def check_quality(self, image):
        self.entered_at = time.perf_counter()
        self.entered.set()
        time.sleep(SLOW_SECONDS)  # stands in for a cold landmarker
        return super().check_quality(image)


class EventLoopTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.engine = engine = SlowEngine()
        store = AvatarStore(root=Path(self._tmp.name) / "faces", engine=engine)
        for patcher in (
            mock.patch.object(app_module, "faces", store),
            mock.patch.object(avatar_store, "FACES_DIR", Path(self._tmp.name) / "faces"),
            mock.patch("face_engine.shared_face_engine", return_value=engine),
            mock.patch.object(app_module.security_gate, "inspect", return_value=(True, 0, {})),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def png() -> bytes:
        buffer = io.BytesIO()
        Image.fromarray(gradient_image(256)).save(buffer, "PNG")
        return buffer.getvalue()

    async def other_request_latency(self, slow_request) -> float:
        """
        Seconds the quick request takes once the slow one is inside the engine.

        The quick request is fired by a watcher that waits (on a worker
        thread) for the engine to report it has started. If the engine's work
        were running on the event loop, the loop could not resume the watcher
        until the work finished, and the measured latency would be the full
        slow time.
        """
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            done_at = []

            async def quick():
                await asyncio.to_thread(self.engine.entered.wait)
                reply = await client.get("/api/v1/audio/languages/en")
                self.assertEqual(reply.status_code, 200)
                done_at.append(time.perf_counter())

            await asyncio.gather(slow_request(client), quick())
            return done_at[0] - self.engine.entered_at

    async def test_analysis_does_not_block_other_requests(self):
        async def analyse(client):
            return await client.post(
                "/api/v1/avatar/face/analyze", files={"file": ("a.png", self.png(), "image/png")}
            )

        self.assertLess(await self.other_request_latency(analyse), SLOW_SECONDS / 2)

    async def test_photo_registration_does_not_block_other_requests(self):
        async def register(client):
            return await client.post(
                "/api/v1/avatar/faces",
                files={"file": ("a.png", self.png(), "image/png")},
                data={"avatarId": "slowface", "subject": "t", "consentBasis": "subject-provided"},
            )

        self.assertLess(await self.other_request_latency(register), SLOW_SECONDS / 2)


if __name__ == "__main__":
    unittest.main()
