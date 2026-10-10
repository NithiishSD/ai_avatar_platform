"""Python client for the AI Avatar Platform API.

    from avatar_platform import AvatarClient

    client = AvatarClient("http://localhost:8000")
    speech = client.synthesize("Hello there.")
    video = client.render(speech, avatar_id="demo")
    print(video.video_url, client.score_lipsync(video.job_id)["lseC"])
"""

from .client import ApiError, AvatarClient, JobFailed, RenderResult, SpeechResult

__all__ = ["ApiError", "AvatarClient", "JobFailed", "RenderResult", "SpeechResult"]
