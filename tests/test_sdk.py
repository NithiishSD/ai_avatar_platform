"""The Python SDK, driven through httpx's MockTransport (no server, no network)."""

import json
import sys
import unittest
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sdk"))

from avatar_platform import ApiError, AvatarClient, JobFailed  # noqa: E402
from contracts import AvatarRenderJob  # noqa: E402

TIMESTAMPS = [{"phoneme": "AA", "viseme": "viseme_aa", "startMs": 0, "endMs": 400}]
SPEECH = {
    "taskId": "t1", "status": "SUCCESS", "modelUsed": "kokoro", "outputPath": "/srv/app/outputs/speech.wav",
    "durationSeconds": 1.0, "phonemeTimestamps": TIMESTAMPS, "alignmentMethod": "mms_fa",
}


class FakeServer:
    """Records requests and answers from a script of (method, path) -> responses."""

    def __init__(self, routes):
        self.routes = {k: list(v) for k, v in routes.items()}
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = (request.method, request.url.path)
        queue = self.routes[key]
        status, body, *headers = queue.pop(0) if len(queue) > 1 else queue[0]
        return httpx.Response(status, json=body, headers=headers[0] if headers else {})


def client_for(routes, **kwargs):
    server = FakeServer(routes)
    sleeps = []
    client = AvatarClient("http://api", transport=httpx.MockTransport(server), sleep=sleeps.append, **kwargs)
    return client, server, sleeps


class SynthesisTests(unittest.TestCase):
    def test_polls_until_success_and_addresses_audio_by_its_outputs_url(self):
        client, server, sleeps = client_for({
            ("POST", "/api/v1/audio/synthesize"): [(202, {"taskId": "t1", "status": "QUEUED"})],
            ("GET", "/api/v1/audio/synthesize/t1"): [(200, {"taskId": "t1", "status": "PROCESSING"}), (200, SPEECH)],
        })
        speech = client.synthesize("hi", emotion="joy")
        self.assertEqual(speech.audio_url, "http://api/outputs/speech.wav")
        self.assertEqual(speech.model_used, "kokoro")
        self.assertEqual(len(sleeps), 1)
        sent = json.loads(server.requests[0].content)
        self.assertEqual((sent["text"], sent["emotion"], sent["returnAlignment"]), ("hi", "joy", True))

    def test_answers_immediately_when_the_server_runs_the_task_eagerly(self):
        client, server, _ = client_for({("POST", "/api/v1/audio/synthesize"): [(202, SPEECH)]})
        client.synthesize("hi")
        self.assertEqual(len(server.requests), 1)

    def test_failure_and_missing_timing_are_errors_not_empty_results(self):
        client, *_ = client_for({("POST", "/api/v1/audio/synthesize"): [(202, {**SPEECH, "status": "FAILED", "error": "no GPU"})]})
        with self.assertRaisesRegex(JobFailed, "FAILED"):
            client.synthesize("hi")
        client, *_ = client_for({("POST", "/api/v1/audio/synthesize"): [(202, {**SPEECH, "phonemeTimestamps": []})]})
        with self.assertRaisesRegex(JobFailed, "no phoneme timestamps"):
            client.synthesize("hi")


class RenderTests(unittest.TestCase):
    def render_client(self, final):
        return client_for({
            ("POST", "/api/v1/audio/synthesize"): [(202, SPEECH)],
            ("POST", "/api/v1/avatar/render-job"): [(202, {"jobId": "x", "status": "QUEUED"})],
            ("GET", "/api/v1/avatar/render-job/job 1"): [(200, {"status": "PROCESSING"}), (200, final)],
        })

    def test_the_job_it_sends_is_valid_against_the_real_contract(self):
        client, server, _ = self.render_client({"status": "COMPLETED", "videoUrl": "/outputs/renders/job.mp4", "engine": "wav2lip", "result": {}})
        result = client.render(client.synthesize("hi"), "demo", engine="wav2lip", background={"color": "#112233"}, job_id="job 1")
        posted = next(r for r in server.requests if r.url.path == "/api/v1/avatar/render-job" and r.method == "POST")
        job = AvatarRenderJob.model_validate(json.loads(posted.content))  # raises if the SDK drifts from the contract
        self.assertEqual((job.avatar_id, job.background.color), ("demo", "#112233"))
        self.assertEqual(posted.url.params["engine"], "wav2lip")
        self.assertEqual(result.video_url, "http://api/outputs/renders/job.mp4")

    def test_a_failed_render_raises_with_the_servers_reason(self):
        client, *_ = self.render_client({"status": "FAILED", "error": "AvatarNotFound: gone"})
        with self.assertRaisesRegex(JobFailed, "AvatarNotFound"):
            client.render(client.synthesize("hi"), "demo", job_id="job 1")

    def test_polling_gives_up_after_the_timeout(self):
        client, *_ = client_for(
            {("GET", "/api/v1/avatar/render-job/j"): [(200, {"status": "PROCESSING"})]}, poll_timeout=3, poll_interval=1
        )
        with self.assertRaises(TimeoutError):
            client._poll("/api/v1/avatar/render-job/j", ("COMPLETED",), "render")


class NewerEndpointTests(unittest.TestCase):
    def test_render_batch_polls_until_done_and_keeps_per_item_refusals(self):
        refused = {"index": 1, "jobId": "b", "accepted": False, "httpStatus": 404, "detail": "not registered"}
        client, server, sleeps = client_for({
            ("POST", "/api/v1/avatar/render-batch"): [(202, {"batchId": "B1", "accepted": 1, "rejected": 1, "jobs": [refused]})],
            ("GET", "/api/v1/avatar/render-batch/B1"): [(200, {"done": False, "counts": {"QUEUED": 1}}),
                                                       (200, {"done": True, "counts": {"COMPLETED": 1, "REJECTED": 1}})],
        })
        state = client.render_batch([{"jobId": "a"}, {"jobId": "b"}], engine="wav2lip")
        self.assertEqual(state["counts"], {"COMPLETED": 1, "REJECTED": 1})
        self.assertEqual(json.loads(server.requests[0].content)["jobs"][1], {"jobId": "b"})
        self.assertEqual(server.requests[0].url.params["engine"], "wav2lip")
        self.assertEqual(len(sleeps), 1)

    def test_voice_to_avatar_uploads_the_file_with_its_consent_basis_and_returns_the_render(self):
        import tempfile

        client, server, _ = client_for({
            ("POST", "/api/v1/avatar/voice-to-avatar"): [(202, {"jobId": "v2a-1", "status": "QUEUED", "speech": {"transcript": "hi", "transcriptSource": "asr"}})],
            ("GET", "/api/v1/avatar/render-job/v2a-1"): [(200, {"status": "COMPLETED", "videoUrl": "/outputs/renders/v2a-1.mp4", "result": {"engine": "blendshape"}})],
        })
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
            audio.write(b"RIFFfake")
            audio.flush()
            result = client.voice_to_avatar(audio.name, "demo", "speaker-recorded")
        body = server.requests[0].content
        self.assertIn(b'name="consentBasis"', body)
        self.assertIn(b"speaker-recorded", body)
        self.assertIn(b"RIFFfake", body)
        self.assertNotIn(b'name="transcript"', body)              # left out: the server will recognise the words
        self.assertEqual((result.video_url, result.result["speech"]["transcriptSource"]), ("http://api/outputs/renders/v2a-1.mp4", "asr"))

    def test_stylize_waits_and_a_failed_task_raises_its_reason(self):
        client, _, _ = client_for({
            ("POST", "/api/v1/avatar/stylize"): [(202, {"taskId": "s1", "status": "QUEUED"})],
            ("GET", "/api/v1/avatar/generate/s1"): [(200, {"taskId": "s1", "status": "FAILED", "error": "InsufficientRAM: close other programs"})],
        })
        with self.assertRaisesRegex(JobFailed, "InsufficientRAM"):
            client.stylize("demo", "cartoon", "demo-cartoon")


class ErrorAndLimitTests(unittest.TestCase):
    def test_errors_carry_the_servers_detail_status_and_request_id(self):
        client, *_ = client_for({("GET", "/health"): [(403, {"detail": "no consent"}, {"X-Request-ID": "abc123"})]})
        with self.assertRaises(ApiError) as caught:
            client.health()
        self.assertEqual((caught.exception.status, caught.exception.detail, caught.exception.request_id), (403, "no consent", "abc123"))
        self.assertIn("abc123", str(caught.exception))

    def test_rate_limit_is_retried_after_retry_after_then_succeeds(self):
        client, server, sleeps = client_for({("GET", "/health"): [(429, {"detail": "slow"}, {"Retry-After": "7"}), (200, {"status": "ok"})]})
        self.assertEqual(client.health()["status"], "ok")
        self.assertEqual(sleeps, [7.0])
        self.assertEqual(len(server.requests), 2)

    def test_rate_limit_gives_up_instead_of_looping_forever(self):
        client, server, sleeps = client_for({("GET", "/health"): [(429, {"detail": "slow"}, {"Retry-After": "1"})]})
        with self.assertRaises(ApiError) as caught:
            client.health()
        self.assertEqual(caught.exception.status, 429)
        self.assertEqual(len(server.requests), 4)  # first try + 3 retries

    def test_api_key_and_a_request_id_are_sent_on_every_call(self):
        server = FakeServer({("GET", "/health"): [(200, {})]})
        AvatarClient("http://api", api_key="k1", transport=httpx.MockTransport(server)).health()
        self.assertEqual(server.requests[0].headers["X-API-Key"], "k1")
        self.assertRegex(server.requests[0].headers["X-Request-ID"], r"^[0-9a-f]{16}$")


if __name__ == "__main__":
    unittest.main()
