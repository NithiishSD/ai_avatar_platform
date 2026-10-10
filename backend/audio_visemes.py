"""
Mouth shapes estimated from the sound alone, for audio that arrives without its words (I-02).

The live avatar's microphone input has no transcript, so there are no phonemes to align. It used to
hold one open-mouth shape and only vary how far it opened with loudness; the owner saw "the same mouth
for every word". This module looks at each 40 ms of audio and picks the viseme family the sound most
resembles, from two cheap measurements:

* **spectral centroid** - the "centre of mass" of the spectrum, in Hz. Rounded vowels (o, u) keep their
  energy low; open vowels (a) sit in the middle; front vowels (e, i) push energy higher; hissing
  consonants (s, f, sh) are mostly above 3 kHz.
* **zero-crossing rate** - how often the waveform changes sign; noise-like consonants cross far more
  often than voiced sounds, which separates "s" from a bright vowel.

    silence            -> viseme_sil  (closed)
    hiss / fricative   -> viseme_SS   (teeth together)
    rounded vowel      -> viseme_O    (rounded)
    open vowel         -> viseme_aa   (open)
    front vowel        -> viseme_E    (wide)

It is an estimate, not lip reading: it cannot tell "p" from "b" or see a tongue. The live panel and
every chunk message say so (``live_engine.AUDIO_DRIVE_NOTE``). How often it agrees with real phoneme
timing on synthesised speech is measured in ``tests/test_audio_visemes.py`` and the progress log.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

WINDOW_MS = 40.0
SILENCE_RMS = 0.01        # same -40 dBFS gate as the live loudness envelope
FRICATIVE_CENTROID = 3000.0
FRICATIVE_ZCR = 0.25      # sign changes per sample
ROUNDED_CENTROID = 650.0
FRONT_CENTROID = 1600.0


def classify(window: np.ndarray, sample_rate: int) -> str:
    """The viseme for one window of mono float audio in [-1, 1]."""
    if window.size == 0 or float(np.sqrt(np.mean(window ** 2))) < SILENCE_RMS:
        return "viseme_sil"
    zcr = float(np.mean(np.abs(np.diff(np.signbit(window).astype(np.int8)))))
    spectrum = np.abs(np.fft.rfft(window * np.hanning(window.size)))
    freqs = np.fft.rfftfreq(window.size, 1.0 / sample_rate)
    centroid = float((freqs * spectrum).sum() / (spectrum.sum() + 1e-12))
    if centroid > FRICATIVE_CENTROID and zcr > FRICATIVE_ZCR:
        return "viseme_SS"
    if centroid < ROUNDED_CENTROID:
        return "viseme_O"
    if centroid > FRONT_CENTROID:
        return "viseme_E"
    return "viseme_aa"


def estimate_timestamps(mono: np.ndarray, sample_rate: int) -> List[Dict[str, object]]:
    """
    Viseme segments for a chunk of audio, in the same shape the aligner produces
    (``phoneme``, ``viseme``, ``startMs``, ``endMs``), with runs of the same viseme merged.
    """
    hop = max(1, int(sample_rate * WINDOW_MS / 1000.0))
    segments: List[Dict[str, object]] = []
    for start in range(0, len(mono), hop):
        viseme = classify(mono[start:start + hop], sample_rate)
        start_ms = start * 1000.0 / sample_rate
        end_ms = min(len(mono), start + hop) * 1000.0 / sample_rate
        if segments and segments[-1]["viseme"] == viseme:
            segments[-1]["endMs"] = end_ms
        else:
            segments.append({"phoneme": "EST", "viseme": viseme, "startMs": start_ms, "endMs": end_ms})
    return segments
