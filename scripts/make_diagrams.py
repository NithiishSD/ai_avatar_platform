#!/usr/bin/env python
"""
Draw the engineering flowcharts in ``docs/images/``.

    backend/.conda/bin/python scripts/make_diagrams.py

The diagrams are code so they can be corrected when the pipeline changes:
edit the function for the chart, re-run, commit the PNG. Only matplotlib is
needed. Each chart describes what the code does today; module names in the
boxes are the files under ``backend/`` that own that step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "images"

INK = "#1F1F1C"
MUTED = "#5F5F57"
LINE = "#9C9C8E"
STYLES = {
    "normal": {"face": "#F4F4EF", "edge": "#BDBDAE", "lw": 1.2},
    "key": {"face": "#EAF3FC", "edge": "#2D7DD2", "lw": 2.0},
    "ok": {"face": "#E8F5EC", "edge": "#3C9A5F", "lw": 1.6},
    "stop": {"face": "#FCEDEA", "edge": "#C8553D", "lw": 1.6},
    "decision": {"face": "#FFF6DD", "edge": "#C9A227", "lw": 1.5},
    "data": {"face": "#FFFFFF", "edge": "#BDBDAE", "lw": 1.2},
}

Point = Tuple[float, float]


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def l(self) -> Point:  # noqa: E743
        return (self.x, self.y + self.h / 2)

    @property
    def r(self) -> Point:
        return (self.x + self.w, self.y + self.h / 2)

    @property
    def t(self) -> Point:
        return (self.x + self.w / 2, self.y)

    @property
    def b(self) -> Point:
        return (self.x + self.w / 2, self.y + self.h)


class Diagram:
    """A canvas in inches with y growing downward, like reading a page."""

    def __init__(self, width: float, height: float, title: str, subtitle: str = "") -> None:
        self.fig = plt.figure(figsize=(width, height), dpi=160)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        self.ax.axis("off")
        self.ax.text(0.45, 0.5, title, fontsize=15, fontweight="bold", color=INK, va="center")
        if subtitle:
            self.ax.text(0.45, 0.88, subtitle, fontsize=9.5, color=MUTED, va="center")

    def panel(self, x: float, y: float, w: float, h: float, label: str) -> None:
        self.ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                           facecolor="#ECECE8", edgecolor="#D8D8D0", linewidth=1.0, zorder=0)
        )
        self.ax.text(x + 0.22, y + 0.27, label, fontsize=9.5, fontweight="bold", color=MUTED, va="center")

    def box(self, x: float, y: float, w: float, h: float, title: str, sub: str = "", kind: str = "normal") -> Box:
        style = STYLES[kind]
        self.ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.10",
                           facecolor=style["face"], edgecolor=style["edge"], linewidth=style["lw"], zorder=2)
        )
        cx = x + w / 2
        if sub:
            lines = sub.count("\n") + 1
            self.ax.text(cx, y + h / 2 - 0.085 * lines - 0.04, title, fontsize=9.6, fontweight="bold",
                         color=INK, ha="center", va="center", zorder=3)
            self.ax.text(cx, y + h / 2 + 0.15, sub, fontsize=7.9, color=MUTED, ha="center", va="center",
                         zorder=3, linespacing=1.25)
        else:
            self.ax.text(cx, y + h / 2, title, fontsize=9.6, fontweight="bold", color=INK,
                         ha="center", va="center", zorder=3, linespacing=1.25)
        return Box(x, y, w, h)

    def decision(self, x: float, y: float, w: float, h: float, text: str) -> Box:
        style = STYLES["decision"]
        cx, cy = x + w / 2, y + h / 2
        self.ax.add_patch(
            Polygon([(cx, y), (x + w, cy), (cx, y + h), (x, cy)], closed=True,
                    facecolor=style["face"], edgecolor=style["edge"], linewidth=style["lw"], zorder=2)
        )
        self.ax.text(cx, cy, text, fontsize=8.6, fontweight="bold", color=INK, ha="center", va="center",
                     zorder=3, linespacing=1.2)
        return Box(x, y, w, h)

    def arrow(self, points: Sequence[Point], label: str = "", label_at: Optional[Point] = None,
              color: str = LINE, dashed: bool = False) -> None:
        pts: List[Point] = list(points)
        style = (0, (4, 3)) if dashed else "solid"
        for a, b in zip(pts[:-2], pts[1:-1], strict=True):
            self.ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=1.4, linestyle=style,
                         solid_capstyle="round", zorder=1)
        self.ax.annotate(
            "", xy=pts[-1], xytext=pts[-2],
            arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.4, linestyle=style,
                            mutation_scale=11, shrinkA=0, shrinkB=0),
            zorder=1,
        )
        if label:
            if label_at is None:
                a, b = pts[0], pts[1]
                label_at = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            self.ax.text(label_at[0], label_at[1], label, fontsize=7.8, color=MUTED, ha="center", va="center",
                         zorder=4, bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none"))

    def note(self, x: float, y: float, text: str, size: float = 8.2, color: str = MUTED, ha: str = "left") -> None:
        self.ax.text(x, y, text, fontsize=size, color=color, ha=ha, va="center", linespacing=1.3)

    def save(self, name: str) -> Path:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / name
        self.fig.savefig(path, facecolor="white")
        plt.close(self.fig)
        return path


def row(d: Diagram, y: float, x0: float, w: float, h: float, gap: float, items, connect: bool = True) -> List[Box]:
    """Lay boxes out left to right and join them with arrows."""
    boxes: List[Box] = []
    for index, item in enumerate(items):
        title, sub, kind = (list(item) + ["", "normal"])[:3] if not isinstance(item, str) else (item, "", "normal")
        boxes.append(d.box(x0 + index * (w + gap), y, w, h, title, sub, kind or "normal"))
    if connect:
        # Consecutive pairs: the second list is one shorter by design.
        for a, b in zip(boxes, boxes[1:], strict=False):
            d.arrow([a.r, b.l])
    return boxes


# ---------------------------------------------------------------------------
# 1. The whole system
# ---------------------------------------------------------------------------
def system_overview() -> Path:
    d = Diagram(15.2, 8.3, "How a script becomes a talking avatar",
                "Two pipelines that only meet at one frozen data contract. Box footers name the file in backend/ that owns the step.")
    w, h, gap = 2.42, 0.95, 0.38

    d.panel(0.4, 1.3, 14.4, 1.85, "Audio pipeline")
    audio = row(d, 1.85, 0.75, w, h, gap, [
        ("Script + voice", "text, language, emotion,\noptional reference clip", "data"),
        ("Speech router", "picks 1 of 5 TTS models\nvoice_engine.py"),
        ("Emotion prosody", "pitch, rate, energy\nemotion_engine.py"),
        ("Forced aligner", "MMS_FA: phonemes -> 15 visemes\nalignment_engine.py"),
        ("Speech + timings", "24 kHz WAV and\n(viseme, startMs, endMs)", "data"),
    ])

    contract = d.box(11.75, 3.62, 2.75, 1.0, "AvatarRenderJob",
                     "the only interface between\naudio and vision - contracts.py", "key")
    d.arrow([audio[-1].b, (audio[-1].b[0], 3.62)])

    d.panel(0.4, 5.05, 14.4, 1.85, "Vision pipeline")
    vision = row(d, 5.6, 0.75, w, h, gap, [
        ("Face image", "photo with consent, or\nStable Diffusion face", "data"),
        ("Avatar store", "consent + quality gate\navatar_store.py"),
        ("Face analysis", "478 landmarks, 52 blendshapes\nface_engine.py"),
        ("Render worker", "timeline -> frames -> lip sync\nrender_engine.py", "key"),
        ("Talking video", "MP4: H.264 + AAC,\n'AI-generated' label", "ok"),
    ])
    # contract -> render worker
    d.arrow([contract.l, (vision[3].t[0], contract.l[1]), vision[3].t], "job", (10.9, contract.l[1]))
    metric = d.box(11.75, 7.15, 2.75, 0.62, "SyncNet LSE-C / LSE-D", "", "normal")
    d.arrow([vision[4].b, (vision[4].b[0], 7.02), (metric.t[0], 7.02), metric.t])
    d.note(11.55, 7.46, "measured lip sync, with its method  ", ha="right")

    d.box(0.4, 7.15, 8.6, 0.62,
          "FastAPI accepts every step as a job  ·  a worker runs it  ·  one heavy model on the 6 GB GPU at a time")
    return d.save("system_overview.png")


# ---------------------------------------------------------------------------
# 2. Inside the render worker
# ---------------------------------------------------------------------------
def render_pipeline() -> Path:
    d = Diagram(15.2, 11.2, "Inside the render worker",
                "render_engine.render_job(): one AvatarRenderJob in, one MP4 out. No step falls back silently.")
    w, h, gap = 2.42, 0.95, 0.38

    job = d.box(0.75, 1.35, w, h, "AvatarRenderJob", "avatarId, audioUrl, timestamps,\nemotionVector, quality, fps", "key")
    pre = d.decision(3.95, 1.2, 2.3, 1.25, "preflight\nrenderable?")
    d.arrow([job.r, pre.l])
    reject = d.box(7.0, 1.35, 3.6, h, "Rejected with the reason",
                   "unknown avatar · no consent record\nunreadable audioUrl · engine has no weights", "stop")
    d.arrow([pre.r, reject.l], "no")

    d.panel(0.4, 2.95, 14.4, 1.85, "Face: done once per job")
    face = row(d, 3.5, 0.75, w, h, gap, [
        ("Load avatar", "consent checked again\navatar_store.py"),
        ("Fit to quality tier", "PREVIEW <= 512 px, HQ <= 1080p\nnever upscaled"),
        ("Find landmarks", "MediaPipe, 478 points\nface_engine.py"),
        ("Rig the photo", "triangulate; build mouth,\neye, brow fields - face_warp.py"),
    ])
    d.arrow([pre.b, (pre.b[0], 2.7), (face[0].t[0], 2.7), face[0].t], "yes", (pre.b[0], 2.62))

    d.panel(0.4, 5.15, 14.4, 1.85, "Motion: phoneme timestamps -> one row of blendshape weights per frame (face_animation.py)")
    motion = row(d, 5.7, 0.75, w, h, gap, [
        ("Hold + rasterise", "hold each phoneme to the next;\n5 ms grid per blendshape"),
        ("Smooth", "Gaussian = coarticulation;\nlips never teleport"),
        ("Re-close + gate", "p/b/m shut the lips; loudness\nscales the jaw, silence closes it"),
        ("Sample per frame", "frame centres at targetFps"),
        ("Emotion + blinks", "brows, cheeks, smile; blinks\nseeded by jobId (repeatable)"),
    ])

    d.note(11.95, 3.98, "the rig renders any set of\nweights; it is reused for\nevery frame below", size=8.0)

    engine = d.decision(0.9, 7.95, 2.3, 1.25, "engine?")
    d.arrow([motion[-1].b, (motion[-1].b[0], 7.45), (engine.t[0], 7.45), engine.t], "weights per frame", (7.6, 7.45))

    blend = d.box(3.95, 7.75, 3.3, 0.9, "blendshape (CPU, default)",
                  "warp the mesh, paint mouth interior\nand eyelids - face_warp.py")
    wav = d.box(3.95, 8.95, 3.3, 0.9, "wav2lip (GPU, opt-in)",
                "warp gives blinks + brows; network\nrepaints the mouth - wav2lip_engine.py")
    d.arrow([engine.r, (3.55, engine.r[1]), (3.55, blend.l[1]), blend.l])
    d.arrow([(3.55, engine.r[1]), (3.55, wav.l[1]), wav.l])

    label = d.box(7.95, 8.35, 2.1, 0.9, "Stamp label", "'AI-generated' burned\ninto every frame")
    d.arrow([blend.r, (7.6, blend.r[1]), (7.6, label.l[1]), label.l])
    d.arrow([wav.r, (7.6, wav.r[1]), (7.6, label.l[1] + 0.02)])
    mux = d.box(10.4, 8.35, 2.2, 0.9, "Encode + mux", "raw frames piped to ffmpeg\nwith the audio - video_io.py")
    d.arrow([label.r, mux.l])
    out = d.box(12.95, 8.35, 1.85, 0.9, "RenderResult", "MP4 + engine name,\ntimings, warnings", "ok")
    d.arrow([mux.r, out.l])

    d.note(0.75, 10.45,
           "What the result always reports: which engine ran, frame count, render time vs real time, ffprobe stream durations,\n"
           "visemes it did not recognise, a duration mismatch between contract and audio, and peak VRAM for the GPU engine.")
    return d.save("render_pipeline.png")


# ---------------------------------------------------------------------------
# 3. Consent and quality gate
# ---------------------------------------------------------------------------
def avatar_gate() -> Path:
    d = Diagram(13.6, 10.0, "How a face is allowed in, and checked again before every render",
                "avatar_store.py + provenance.py. A face with no consent record can be listed, but never animated.")
    cx = 1.0
    start = d.box(cx, 1.35, 2.6, 0.8, "New face image", "upload, CLI, or generated", "data")
    src = d.decision(cx + 0.1, 2.55, 2.4, 1.2, "real person?")
    d.arrow([start.b, src.t])

    consent = d.decision(4.6, 2.55, 2.6, 1.2, "consent basis\n+ subject named?")
    d.arrow([src.r, consent.l], "yes")
    no_consent = d.box(8.2, 2.75, 4.6, 0.8, "Refused: nothing is written",
                       "needs subject-provided, written-consent or open-licence", "stop")
    d.arrow([consent.r, no_consent.l], "no")

    synth = d.box(cx, 4.35, 2.6, 0.85, "Synthetic face", "SD 1.5; prompt + seed recorded\navatar_generator.py")
    d.arrow([src.b, synth.t], "no")

    decode = d.box(4.6, 4.35, 2.6, 0.85, "Decode + read EXIF", "upright RGB, <= 2048 px;\nrights notices are surfaced")
    d.arrow([consent.b, decode.t], "yes")
    d.arrow([synth.r, decode.l])

    gate = d.decision(4.7, 5.75, 2.4, 1.3, "quality gate\npasses?")
    d.arrow([decode.b, gate.t])
    bad = d.box(8.2, 5.75, 4.6, 1.3, "Rejected, with what to fix",
                "no face  ·  more than one face\nhead turned > 30 degrees  ·  face < 96 px tall\n(tilt, open mouth, closed eyes only warn)", "stop")
    d.arrow([gate.r, bad.l], "no")

    store = d.box(4.35, 7.95, 3.1, 0.9, "Stored in inputs/faces/",
                  "<avatarId>.png (EXIF stripped)\n+ .provenance.json sidecar", "ok")
    d.arrow([gate.b, store.t], "yes")

    use = d.decision(8.6, 7.75, 2.5, 1.3, "at render time:\nrecord permits use?")
    d.arrow([store.r, use.l])
    go = d.box(11.5, 7.55, 1.7, 0.7, "Render", "", "ok")
    stop = d.box(11.5, 8.55, 1.7, 0.7, "403 refused", "", "stop")
    d.arrow([use.r, (11.3, use.r[1]), (11.3, go.l[1]), go.l], "yes", (11.3, 8.1))
    d.arrow([(11.3, use.r[1]), (11.3, stop.l[1]), stop.l], "no", (11.3, 8.7))

    d.note(1.0, 6.6, "Generated faces are held to a\nstricter gate: closed eyes or an\nopen mouth reject the seed and\nthe next seed is tried.")
    d.note(1.0, 8.4, "Face images are never committed\nto git (.gitignore); the sidecar\nis the consent record.")
    return d.save("avatar_consent_gate.png")


# ---------------------------------------------------------------------------
# 4. Job lifecycle through the API
# ---------------------------------------------------------------------------
def job_lifecycle() -> Path:
    d = Diagram(14.4, 8.2, "Life of a render job through the API",
                "app.py + job_queue.py + celery_app.py. The request returns in milliseconds; the render happens on a worker.")
    w, h = 2.5, 0.9

    client = d.box(0.6, 1.4, w, h, "POST /avatar/render-job", "?engine=blendshape | wav2lip", "data")
    sec = d.box(3.6, 1.4, w, h, "Security gate", "API key, token-bucket\nrate limit - security.py")
    val = d.box(6.6, 1.4, w, h, "Validate contract", "Pydantic: AvatarRenderJob\n422 if malformed")
    pre = d.box(9.6, 1.4, w, h, "Preflight", "avatar, consent, audio, engine\n400 / 403 / 404 with reason")
    d.arrow([client.r, sec.l]); d.arrow([sec.r, val.l]); d.arrow([val.r, pre.l])
    acc = d.box(12.4 - 0.3, 3.0, 2.0, 0.8, "202 QUEUED", "returned at once", "ok")
    d.arrow([pre.r, (acc.t[0], pre.r[1]), acc.t])

    d.panel(0.4, 4.25, 13.6, 2.0, "Worker: one render at a time")
    q = d.box(0.75, 4.85, 2.7, h, "Queue", "in_memory: 1 worker thread\ncelery: Redis broker + worker")
    p = d.box(4.05, 4.85, 2.4, h, "PROCESSING", "progress = frames written", "key")
    ok = d.box(7.2, 4.5, 2.9, 0.75, "COMPLETED", "videoUrl + full RenderResult", "ok")
    fail = d.box(7.2, 5.4, 2.9, 0.75, "FAILED", "error says why and how to fix", "stop")
    d.arrow([(pre.b[0], pre.b[1]), (pre.b[0], 3.85), (q.t[0], 3.85), q.t], "enqueue", (6.0, 3.85))
    d.arrow([q.r, p.l])
    d.arrow([p.r, (6.85, p.r[1]), (6.85, ok.l[1]), ok.l])
    d.arrow([(6.85, p.r[1]), (6.85, fail.l[1]), fail.l])
    mp4 = d.box(10.9, 4.5, 2.8, 0.75, "outputs/renders/<jobId>.mp4", "", "data")
    d.arrow([ok.r, mp4.l])

    poll = d.box(0.75, 6.85, 3.6, 0.8, "GET /avatar/render-job/{id}", "status, progress, videoUrl, error", "data")
    play = d.box(5.0, 6.85, 3.0, 0.8, "GET /outputs/renders/...mp4", "static file, plays in <video>", "data")
    score = d.box(8.65, 6.85, 5.05, 0.8, "POST /avatar/render-job/{id}/lipsync-score",
                  "SyncNet LSE-C / LSE-D of the finished video", "data")
    d.arrow([poll.r, play.l]); d.arrow([play.r, score.l])
    d.arrow([(p.b[0], p.b[1]), (p.b[0], 6.55), (poll.t[0], 6.55), poll.t], "polled by the UI", (3.9, 6.55), dashed=True)
    return d.save("render_job_lifecycle.png")


# ---------------------------------------------------------------------------
# 5. Speech routing
# ---------------------------------------------------------------------------
def speech_routing() -> Path:
    d = Diagram(14.0, 8.4, "How the speech router picks a model",
                "voice_engine.py. Five routes; a route whose weights are missing is reported, not hidden.")
    req = d.box(0.6, 3.55, 2.5, 0.95, "Synthesis request", "text, mode, language,\nquality, style, emotion", "data")
    pick = d.decision(3.7, 3.3, 2.5, 1.45, "select_model()")
    d.arrow([req.r, pick.l])

    models = [
        ("Kokoro 82M", "mode fast, English\n~74 ms warm, MOS 4.33", "ok", "weights present"),
        ("MMS-TTS", "mode multilingual / non-English\n1,000+ languages (VITS per language)", "ok", "hin, tam, swh, spa cached"),
        ("XTTS-v2", "mode clone: zero-shot voice cloning\nfrom a 30-60 s reference", "stop", "weights missing (CPML licence)"),
        ("Higgs TTS 2 (3B)", "mode high_quality / quality high", "stop", "weights missing"),
        ("Dia 1.6B", "mode dialogue, [S1]/[S2] tags", "stop", "weights missing"),
    ]
    boxes = []
    for index, (title, sub, kind, state) in enumerate(models):
        y = 1.35 + index * 1.18
        b = d.box(7.2, y, 3.5, 0.95, title, sub, kind)
        boxes.append(b)
        d.arrow([(6.75, pick.r[1]), (6.75, b.l[1]), b.l])
        d.note(10.9, y + 0.47, state, color="#3C9A5F" if kind == "ok" else "#C8553D")
    d.arrow([pick.r, (6.75, pick.r[1])])

    post = d.box(3.3, 7.35, 3.0, 0.75, "Emotion prosody", "one combined stretch + pitch shift")
    align = d.box(6.9, 7.35, 3.0, 0.75, "Forced alignment", "phoneme / viseme timestamps")
    audit = d.box(10.5, 7.35, 3.0, 0.75, "Quality audit (optional)", "SQUIM MOS, ECAPA similarity")
    d.arrow([(boxes[-1].b[0] - 1.2, boxes[-1].b[1]), (boxes[-1].b[0] - 1.2, 7.15), (post.t[0], 7.15), post.t],
            "audio from whichever model ran", (6.6, 7.15))
    d.arrow([post.r, align.l]); d.arrow([align.r, audit.l])
    d.note(0.6, 5.7, "Status shown is from the on-disk weight\naudit on 30 Sep 2026 (model_registry.py).\nThe same audit is printed at server start\nand returned by GET /health; the response\nfield model_used names what actually ran.")
    return d.save("speech_routing.png")


# ---------------------------------------------------------------------------
# 6. Lip-sync metric
# ---------------------------------------------------------------------------
def lipsync_metric() -> Path:
    d = Diagram(14.4, 6.6, "How lip sync is measured",
                "lipsync_metric.py. SyncNet v2, same procedure as the lip-sync literature; the score carries its method string.")
    w, h, gap = 2.3, 0.95, 0.35
    top = row(d, 1.45, 0.6, w, h, gap, [
        ("Rendered MP4", "video + audio streams", "data"),
        ("Frames at 25 fps", "decoded by ffmpeg\nvideo_io.py"),
        ("Face crop 224 px", "box from first frame,\nmouth in the lower middle"),
        ("5-frame windows", "0.2 s of mouth motion"),
        ("Lip embedding", "SyncNet video branch\n1024 numbers per window", "key"),
    ])
    bottom = row(d, 3.0, 0.6 + (w + gap), w, h, gap, [
        ("Audio at 16 kHz", "mono"),
        ("13 MFCCs, 100 Hz", "python_speech_features"),
        ("20-step windows", "the same 0.2 s"),
        ("Audio embedding", "SyncNet audio branch\n1024 numbers per window", "key"),
    ])
    d.arrow([top[0].b, (top[0].b[0], bottom[0].l[1]), bottom[0].l])

    cmp_box = d.box(4.2, 4.75, 3.6, 1.0, "Compare at offsets -15 .. +15 frames",
                    "distance between lip and audio\nembeddings, averaged over the clip")
    d.arrow([top[-1].r, (14.0, top[-1].r[1]), (14.0, 5.25), cmp_box.r])
    d.arrow([bottom[-1].r, (14.0, bottom[-1].r[1])])
    out = d.box(0.6, 4.75, 3.0, 1.0, "LSE-D  ·  LSE-C  ·  offset",
                "min distance (lower = better)\nmedian - min (higher = better)", "ok")
    d.arrow([cmp_box.l, out.r])
    d.note(8.2, 4.4, "Controls run on 30 Sep 2026: audio delayed 200 / 400 ms was reported as\n-5 / -10 frames; unrelated audio dropped LSE-C to 0.7.", size=7.9)
    d.note(8.2, 6.15, "Not a percentage: the roadmap's '> 95% lip sync' has no defined measurement.", size=7.9)
    return d.save("lipsync_metric.png")


CHARTS = [system_overview, render_pipeline, avatar_gate, job_lifecycle, speech_routing, lipsync_metric]


def main() -> int:
    for chart in CHARTS:
        print(f"wrote {chart().relative_to(OUT_DIR.parents[1])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
