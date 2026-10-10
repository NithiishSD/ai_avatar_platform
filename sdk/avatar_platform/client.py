"""
The client: one class, one method per step of the pipeline.

The API is asynchronous for anything heavy: ``POST`` answers with an id, and
the client polls until the job is terminal. This module hides that loop
(``synthesize`` and ``render`` return finished results) and turns every HTTP
error into an ``ApiError`` that carries the server's own explanation and the
request id, so a failure can be traced in the server's logs.

Rate limiting: a 429 is retried after the server's ``Retry-After`` (a few
times, then it is raised), because a polling client cannot avoid brushing the
limit and the server tells it exactly how long to wait.

Typical use::

    with AvatarClient("http://localhost:8000") as client:
        speech = client.synthesize("Hello there.")
        video = client.render(speech, avatar_id="demo")
        print(video.video_url)

Concepts used here, explained once:

**httpx** is an HTTP client library with an API close to ``requests``. One
``httpx.Client`` keeps a pool of open connections (keep-alive), so the many
small polling requests reuse one TCP connection instead of opening a new one
each time.

**Polling.** A render can take minutes, longer than an HTTP request should
stay open. So the server answers a ``POST`` at once with a job id, and the
client asks "is it done yet?" with a ``GET`` every ``poll_interval`` seconds
until the status is terminal (finished, one way or the other).

**Request id.** Every request carries a random ``X-Request-ID``. The server
logs it, so the id in an error message finds the matching server log lines.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote

import httpx

# Header names, written once so they cannot drift. They match the server's
# own constants (backend/request_context.py and backend/security.py).
REQUEST_ID_HEADER = "X-Request-ID"
API_KEY_HEADER = "X-API-Key"
# After this many 429 retries the error is raised: a server that is still
# refusing is overloaded, and the caller should know rather than wait forever.
MAX_RATE_LIMIT_RETRIES = 3


class ApiError(RuntimeError):
    """The server refused or failed a request. ``detail`` is its own message."""

    def __init__(self, status: int, detail: Any, request_id: Optional[str] = None):
        """Keep the status, the server's detail and the request id as attributes."""
        self.status = status
        self.detail = detail
        self.request_id = request_id
        suffix = f" [request {request_id}]" if request_id else ""
        # The text passed to RuntimeError is what str(err) and a traceback show.
        super().__init__(f"HTTP {status}: {detail}{suffix}")


class JobFailed(RuntimeError):
    """A job was accepted but ended FAILED; ``args[0]`` is the server's reason."""


@dataclass
class SpeechResult:
    """A finished synthesis: what spoke, the audio, and the measured phoneme timeline."""

    task_id: str
    # Which engine actually spoke. Golden rule 1 (no silent fallbacks): this
    # can differ from what was asked for, so it is always reported.
    model_used: str
    # Path on the server's disk, as the server reported it.
    output_path: str
    # The same file as a URL the render worker is allowed to read.
    audio_url: str
    duration_seconds: float
    # When each phoneme starts and ends; the renderer moves the mouth to it.
    phoneme_timestamps: List[Dict[str, Any]]
    # How those timestamps were obtained, so their quality can be judged.
    alignment_method: Optional[str]
    # The whole server response, for fields the dataclass does not name.
    # repr=False keeps it out of print(), where it would be very long.
    # default_factory=dict gives each instance its own empty dict; a plain
    # default of {} would be shared by every instance.
    raw: Dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass
class RenderResult:
    """A finished render: where the MP4 is and how it was made."""

    job_id: str
    video_url: str
    # The render engine that ran, as the server reports it.
    engine: Optional[str]
    # The server's result object (timings, scores and so on), passed through.
    result: Dict[str, Any]


class AvatarClient:
    """
    A synchronous client for the avatar platform API.

    Use it as a context manager (``with AvatarClient() as client:``) so the
    connection pool is closed at the end, or call ``close()`` yourself.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        poll_interval: float = 1.0,
        poll_timeout: float = 1800.0,
        transport: Optional[httpx.BaseTransport] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """
        Configure the connection.

        ``timeout`` limits one HTTP request; ``poll_timeout`` limits how long
        a whole job may take (30 minutes by default). ``sleep`` is injectable
        so tests can skip the real waiting.
        """
        # The key header is only sent when a key was given, so a server with
        # auth off never sees an empty key.
        headers = {API_KEY_HEADER: api_key} if api_key else {}
        # `transport` is how the tests (and anyone behind a proxy) swap the network out.
        self._http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport)
        # rstrip("/") so joining with "/outputs/..." never gives "//outputs".
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self._sleep = sleep

    def close(self) -> None:
        """Close the underlying connection pool."""
        self._http.close()

    def __enter__(self) -> "AvatarClient":
        """Enter a ``with`` block; the client itself is the value bound by ``as``."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Leave a ``with`` block, even on an exception, by closing the client."""
        self.close()

    # -- plumbing ---------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """
        Send one request and return its decoded JSON body (None if empty).

        Retries on 429 after the server's ``Retry-After`` seconds, and raises
        ``ApiError`` for any status of 400 or above.
        """
        # 16 hex characters are plenty to tell requests apart in a log while
        # staying short enough to read.
        request_id = uuid.uuid4().hex[:16]
        # Caller headers are merged after the request id, so a caller may
        # supply its own id.
        headers = {REQUEST_ID_HEADER: request_id, **kwargs.pop("headers", {})}
        # range(N + 1): one first try plus up to N retries.
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            response = self._http.request(method, path, headers=headers, **kwargs)
            # 429 Too Many Requests: the server's rate limiter refused this
            # call. Retry-After says how many seconds until a token is free;
            # one second is assumed if the header is missing.
            if response.status_code == 429 and attempt < MAX_RATE_LIMIT_RETRIES:
                self._sleep(float(response.headers.get("Retry-After", "1")))
                continue
            break
        # Prefer the id the server echoes back, which is the one in its logs.
        served_id = response.headers.get(REQUEST_ID_HEADER, request_id)
        if response.status_code >= 400:
            # FastAPI puts its explanation under "detail". A proxy or a crash
            # may answer with non-JSON text, which json() rejects with
            # ValueError; then the raw text is the best explanation there is.
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(response.status_code, detail, served_id)
        return response.json() if response.content else None

    def _poll(self, path: str, done: tuple, label: str) -> Dict[str, Any]:
        """
        GET ``path`` until its ``status`` is one of ``done``; return that state.

        Raises ``TimeoutError`` after ``poll_timeout`` seconds. ``label`` names
        the job in that error.
        """
        # Time waited is summed from the sleeps rather than read from a clock,
        # so a test with a fake sleep times out at once instead of hanging.
        waited = 0.0
        while True:
            state = self._request("GET", path)
            if state.get("status") in done:
                return state
            if waited >= self.poll_timeout:
                raise TimeoutError(f"{label} still {state.get('status')} after {self.poll_timeout:.0f}s")
            self._sleep(self.poll_interval)
            waited += self.poll_interval

    # -- pipeline ---------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """The server's /health report: status, model weights, queue backend."""
        return self._request("GET", "/health")

    def synthesize(self, text: str, mode: str = "fast", language: str = "en", **options: Any) -> SpeechResult:
        """
        Speak ``text`` and return once the audio and its phoneme timing exist.

        ``options`` are the API's own camelCase fields: ``speakerWav`` and
        ``cloneEngine`` for cloning, ``emotion``, ``speed``, ``pitch``...
        """
        # returnAlignment asks for the phoneme timeline, which render() needs.
        body = {"text": text, "mode": mode, "language": language, "returnAlignment": True, **options}
        state = self._request("POST", "/api/v1/audio/synthesize", json=body)
        # The POST may already carry a terminal status; only poll when it
        # does not. These four are the terminal states of a synthesis task.
        if state.get("status") not in ("SUCCESS", "FAILED", "CANCELLED", "UNKNOWN"):
            state = self._poll(
                f"/api/v1/audio/synthesize/{state['taskId']}", ("SUCCESS", "FAILED", "CANCELLED", "UNKNOWN"), "synthesis"
            )
        if state.get("status") != "SUCCESS":
            raise JobFailed(f"synthesis ended {state.get('status')}: {state.get('error') or state}")
        # Fail here, not later inside the render, so the error points at the
        # step that actually went wrong.
        timestamps = state.get("phonemeTimestamps") or []
        if not timestamps:
            raise JobFailed("synthesis returned no phoneme timestamps, so there is nothing to lip-sync")
        output_path = state["outputPath"]
        # The render worker only reads this server's own /outputs/ URLs, so the
        # audio is addressed by the part of its path under outputs/.
        # Backslashes become slashes first so a Windows path splits the same way.
        relative = output_path.replace("\\", "/").rsplit("/outputs/", 1)[-1]
        return SpeechResult(
            task_id=state["taskId"],
            model_used=state.get("modelUsed", ""),
            output_path=output_path,
            # quote() percent-encodes characters such as spaces so the URL is valid.
            audio_url=f"{self.base_url}/outputs/{quote(relative)}",
            duration_seconds=float(state["durationSeconds"]),
            phoneme_timestamps=timestamps,
            alignment_method=state.get("alignmentMethod"),
            raw=state,
        )

    def render(
        self,
        speech: SpeechResult,
        avatar_id: str,
        engine: Optional[str] = None,
        quality: str = "PREVIEW",
        fps: int = 25,
        emotion_vector: Optional[Dict[str, float]] = None,
        background: Optional[Dict[str, str]] = None,
        job_id: Optional[str] = None,
    ) -> RenderResult:
        """Render ``speech`` onto a registered face and wait for the MP4."""
        # A caller-chosen id makes a job easy to find later; otherwise a
        # random one with an "sdk-" prefix marks where it came from.
        job_id = job_id or f"sdk-{uuid.uuid4().hex[:12]}"
        # This dict is an AvatarRenderJob (backend/contracts.py), the frozen
        # audio-to-vision contract, written with its camelCase wire names.
        job: Dict[str, Any] = {
            "jobId": job_id,
            "avatarId": avatar_id,
            "audioUrl": speech.audio_url,
            # A fixed value: the SDK does not read the rate from the
            # synthesis result.
            "sampleRate": 24000,
            "durationSeconds": speech.duration_seconds,
            "phonemeTimestamps": speech.phoneme_timestamps,
            # Default expression: fully neutral, normal blink rate.
            "emotionVector": emotion_vector or {"happy": 0.0, "neutral": 1.0, "eyeblinkRate": 1.0},
            "renderQuality": quality,
            "targetFps": fps,
        }
        if background:
            job["background"] = background
        # The engine is a query parameter, not part of the contract body;
        # params=None sends no query string so the server picks its default.
        self._request("POST", "/api/v1/avatar/render-job", json=job, params={"engine": engine} if engine else None)
        # safe='' also encodes "/", so an id containing one stays one path segment.
        state = self._poll(f"/api/v1/avatar/render-job/{quote(job_id, safe='')}", ("COMPLETED", "FAILED"), "render")
        if state["status"] != "COMPLETED":
            raise JobFailed(state.get("error") or "render failed")
        # videoUrl is a server-relative path; prefixing base_url makes it absolute.
        return RenderResult(job_id=job_id, video_url=f"{self.base_url}{state['videoUrl']}", engine=state.get("engine"), result=state.get("result") or {})

    def score_lipsync(self, job_id: str) -> Dict[str, Any]:
        """SyncNet LSE-C / LSE-D / offset for a finished render, with its method string."""
        return self._request("POST", f"/api/v1/avatar/render-job/{quote(job_id, safe='')}/lipsync-score")["score"]

    def voice_similarity(self, reference_path: str, generated_path: str) -> Dict[str, Any]:
        """ECAPA-TDNN similarity between a reference voice and a generated clip."""
        return self._request(
            "POST", "/api/v1/audio/voice-similarity", json={"referencePath": reference_path, "generatedPath": generated_path}
        )["report"]

    def faces(self) -> List[Dict[str, Any]]:
        """Every registered avatar face the server knows about."""
        return self._request("GET", "/api/v1/avatar/faces")["avatars"]

    def generate_face(self, avatar_id: str, **choices: Any) -> Dict[str, Any]:
        """Generate a synthetic face (``age``, ``presentation``, ``hair``, ``glasses``...) and wait for it."""
        task = self._request("POST", "/api/v1/avatar/generate", json={"avatarId": avatar_id, **choices})
        state = self._poll(f"/api/v1/avatar/generate/{task['taskId']}", ("COMPLETED", "FAILED"), "face generation")
        if state["status"] != "COMPLETED":
            raise JobFailed(state.get("error") or "face generation failed")
        return state["result"]

    # -- added with milestone 8 -------------------------------------------

    def render_batch(self, jobs: List[Dict[str, Any]], engine: Optional[str] = None, wait: bool = True) -> Dict[str, Any]:
        """
        Queue up to 50 ``AvatarRenderJob`` payloads at once. Items the server refuses are reported by
        index (``accepted: false`` with ``httpStatus`` and ``detail``) and do not stop the others.
        With ``wait`` the call returns once no job is still queued or running.
        """
        batch = self._request("POST", "/api/v1/avatar/render-batch", json={"jobs": jobs}, params={"engine": engine} if engine else None)
        if not wait:
            return batch
        # Same loop as _poll, but a batch reports a "done" flag and per-status
        # counts instead of one status field.
        waited = 0.0
        while True:
            state = self._request("GET", f"/api/v1/avatar/render-batch/{batch['batchId']}")
            if state["done"]:
                return state
            if waited >= self.poll_timeout:
                raise TimeoutError(f"batch {batch['batchId']} not finished after {self.poll_timeout:.0f}s: {state['counts']}")
            self._sleep(self.poll_interval)
            waited += self.poll_interval

    def voice_to_avatar(
        self, audio_file: str, avatar_id: str, consent_basis: str, transcript: Optional[str] = None,
        language: Optional[str] = None, engine: Optional[str] = None, quality: str = "PREVIEW",
    ) -> RenderResult:
        """
        Your own recording drives a face. ``consent_basis`` is whose voice it is (speaker-recorded,
        written-consent, open-licence). Without a ``transcript`` the server recognises the words.
        Returns the finished render; ``result["speech"]`` holds the transcript and its source.
        """
        form = {"avatarId": avatar_id, "consentBasis": consent_basis, "renderQuality": quality}
        # Only the optional fields that were actually given are sent.
        form.update({k: v for k, v in (("transcript", transcript), ("language", language), ("engine", engine)) if v})
        # data= and files= together make a multipart/form-data upload, the
        # format HTML forms use to send a file next to text fields. Each file
        # is (filename, open file, content type). The file stays open only for
        # the upload, thanks to the with block.
        with open(audio_file, "rb") as handle:
            accepted = self._request("POST", "/api/v1/avatar/voice-to-avatar", data=form,
                                     files={"file": (audio_file.rsplit("/", 1)[-1], handle, "application/octet-stream")})
        state = self._poll(f"/api/v1/avatar/render-job/{quote(accepted['jobId'], safe='')}", ("COMPLETED", "FAILED"), "render")
        if state["status"] != "COMPLETED":
            raise JobFailed(state.get("error") or "render failed")
        # The transcript arrives with the upload's answer, not the render
        # state, so it is merged into the result here.
        return RenderResult(job_id=accepted["jobId"], video_url=f"{self.base_url}{state['videoUrl']}", engine=state.get("engine"),
                            result={**(state.get("result") or {}), "speech": accepted.get("speech")})

    def stylize(self, avatar_id: str, style: str, new_avatar_id: str, seed: int = 0, steps: int = 30) -> Dict[str, Any]:
        """Restyle a face (realistic, cartoon, painting, sketch) as a new avatar; the result carries its identity score."""
        task = self._request("POST", "/api/v1/avatar/stylize",
                             json={"avatarId": avatar_id, "style": style, "newAvatarId": new_avatar_id, "seed": seed, "steps": steps})
        # Style transfer shares the face-generation task endpoint for polling.
        state = self._poll(f"/api/v1/avatar/generate/{task['taskId']}", ("COMPLETED", "FAILED"), "style transfer")
        if state["status"] != "COMPLETED":
            raise JobFailed(state.get("error") or "style transfer failed")
        return state["result"]

    def metrics(self) -> Dict[str, Any]:
        """Queue depth, render timings and lip-sync scores so far."""
        return self._request("GET", "/api/v1/metrics")

    def parameters(self) -> Dict[str, Any]:
        """Every customisation parameter with its range and measured status, and the count against the target."""
        return self._request("GET", "/api/v1/parameters")

    def protect_voice(self, recording: str) -> str:
        """Put a voice on the protected list (only its speaker embedding is kept); returns its opaque id."""
        with open(recording, "rb") as handle:
            return self._request("POST", "/api/v1/abuse/protected-voices", files={"file": ("voice", handle, "application/octet-stream")})["id"]
