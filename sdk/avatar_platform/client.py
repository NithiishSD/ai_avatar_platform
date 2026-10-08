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
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote

import httpx

REQUEST_ID_HEADER = "X-Request-ID"
API_KEY_HEADER = "X-API-Key"
MAX_RATE_LIMIT_RETRIES = 3


class ApiError(RuntimeError):
    """The server refused or failed a request. ``detail`` is its own message."""

    def __init__(self, status: int, detail: Any, request_id: Optional[str] = None):
        self.status = status
        self.detail = detail
        self.request_id = request_id
        suffix = f" [request {request_id}]" if request_id else ""
        super().__init__(f"HTTP {status}: {detail}{suffix}")


class JobFailed(RuntimeError):
    """A job was accepted but ended FAILED; ``args[0]`` is the server's reason."""


@dataclass
class SpeechResult:
    """A finished synthesis: what spoke, the audio, and the measured phoneme timeline."""

    task_id: str
    model_used: str
    output_path: str
    audio_url: str
    duration_seconds: float
    phoneme_timestamps: List[Dict[str, Any]]
    alignment_method: Optional[str]
    raw: Dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass
class RenderResult:
    """A finished render: where the MP4 is and how it was made."""

    job_id: str
    video_url: str
    engine: Optional[str]
    result: Dict[str, Any]


class AvatarClient:
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
        headers = {API_KEY_HEADER: api_key} if api_key else {}
        # `transport` is how the tests (and anyone behind a proxy) swap the network out.
        self._http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport)
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self._sleep = sleep

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "AvatarClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- plumbing ---------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        request_id = uuid.uuid4().hex[:16]
        headers = {REQUEST_ID_HEADER: request_id, **kwargs.pop("headers", {})}
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            response = self._http.request(method, path, headers=headers, **kwargs)
            if response.status_code == 429 and attempt < MAX_RATE_LIMIT_RETRIES:
                self._sleep(float(response.headers.get("Retry-After", "1")))
                continue
            break
        served_id = response.headers.get(REQUEST_ID_HEADER, request_id)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(response.status_code, detail, served_id)
        return response.json() if response.content else None

    def _poll(self, path: str, done: tuple, label: str) -> Dict[str, Any]:
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
        return self._request("GET", "/health")

    def synthesize(self, text: str, mode: str = "fast", language: str = "en", **options: Any) -> SpeechResult:
        """
        Speak ``text`` and return once the audio and its phoneme timing exist.

        ``options`` are the API's own camelCase fields: ``speakerWav`` and
        ``cloneEngine`` for cloning, ``emotion``, ``speed``, ``pitch``...
        """
        body = {"text": text, "mode": mode, "language": language, "returnAlignment": True, **options}
        state = self._request("POST", "/api/v1/audio/synthesize", json=body)
        if state.get("status") not in ("SUCCESS", "FAILED", "CANCELLED", "UNKNOWN"):
            state = self._poll(
                f"/api/v1/audio/synthesize/{state['taskId']}", ("SUCCESS", "FAILED", "CANCELLED", "UNKNOWN"), "synthesis"
            )
        if state.get("status") != "SUCCESS":
            raise JobFailed(f"synthesis ended {state.get('status')}: {state.get('error') or state}")
        timestamps = state.get("phonemeTimestamps") or []
        if not timestamps:
            raise JobFailed("synthesis returned no phoneme timestamps, so there is nothing to lip-sync")
        output_path = state["outputPath"]
        # The render worker only reads this server's own /outputs/ URLs, so the
        # audio is addressed by the part of its path under outputs/.
        relative = output_path.replace("\\", "/").rsplit("/outputs/", 1)[-1]
        return SpeechResult(
            task_id=state["taskId"],
            model_used=state.get("modelUsed", ""),
            output_path=output_path,
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
        job_id = job_id or f"sdk-{uuid.uuid4().hex[:12]}"
        job: Dict[str, Any] = {
            "jobId": job_id,
            "avatarId": avatar_id,
            "audioUrl": speech.audio_url,
            "sampleRate": 24000,
            "durationSeconds": speech.duration_seconds,
            "phonemeTimestamps": speech.phoneme_timestamps,
            "emotionVector": emotion_vector or {"happy": 0.0, "neutral": 1.0, "eyeblinkRate": 1.0},
            "renderQuality": quality,
            "targetFps": fps,
        }
        if background:
            job["background"] = background
        self._request("POST", "/api/v1/avatar/render-job", json=job, params={"engine": engine} if engine else None)
        state = self._poll(f"/api/v1/avatar/render-job/{quote(job_id, safe='')}", ("COMPLETED", "FAILED"), "render")
        if state["status"] != "COMPLETED":
            raise JobFailed(state.get("error") or "render failed")
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
        form.update({k: v for k, v in (("transcript", transcript), ("language", language), ("engine", engine)) if v})
        with open(audio_file, "rb") as handle:
            accepted = self._request("POST", "/api/v1/avatar/voice-to-avatar", data=form,
                                     files={"file": (audio_file.rsplit("/", 1)[-1], handle, "application/octet-stream")})
        state = self._poll(f"/api/v1/avatar/render-job/{quote(accepted['jobId'], safe='')}", ("COMPLETED", "FAILED"), "render")
        if state["status"] != "COMPLETED":
            raise JobFailed(state.get("error") or "render failed")
        return RenderResult(job_id=accepted["jobId"], video_url=f"{self.base_url}{state['videoUrl']}", engine=state.get("engine"),
                            result={**(state.get("result") or {}), "speech": accepted.get("speech")})

    def stylize(self, avatar_id: str, style: str, new_avatar_id: str, seed: int = 0, steps: int = 30) -> Dict[str, Any]:
        """Restyle a face (realistic, cartoon, painting, sketch) as a new avatar; the result carries its identity score."""
        task = self._request("POST", "/api/v1/avatar/stylize",
                             json={"avatarId": avatar_id, "style": style, "newAvatarId": new_avatar_id, "seed": seed, "steps": steps})
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
