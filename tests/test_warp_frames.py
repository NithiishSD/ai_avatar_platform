"""
The parallel frame warper (``render_engine._warp_frames``, T9.3).

Frames are warped by a small thread pool but must reach the encoder in order and identical to the
one-at-a-time loop. These tests use a stand-in animator whose frames take random, uneven times, so
a pool that handed frames out in completion order would be caught. No model is loaded.
"""

import os
import random
import time
import unittest
from unittest import mock

import numpy as np

import render_engine


class FakeTrack:
    """The three things ``_warp_frames`` asks a track for."""

    def __init__(self, frames):
        self.frame_count = frames

    def frame(self, index):
        return {"jawOpen": 0.1 * (index % 7), "mouthSmileLeft": 0.2, "eyeBlinkLeft": 0.05 * (index % 3)}

    def pose(self, index):
        return (float(index), 0.0, 0.0, 0.0, 0.0)


class FakeAnimator:
    """Returns a 2x2 'frame' encoding which index and weights it was given, after a random delay."""

    def __init__(self, fail_at=None):
        self.fail_at = fail_at
        self.calls = 0

    def render(self, weights, pose):
        self.calls += 1
        if pose[0] == self.fail_at:
            raise RuntimeError("warp failed")
        time.sleep(random.random() * 0.01)
        value = pose[0] + sum(weights.values())
        return np.full((2, 2, 3), value, dtype=np.float32)


def collect(frames=40, include_mouth=True, animator=None, threads="4"):
    animator = animator or FakeAnimator()
    with mock.patch.dict(os.environ, {"RENDER_THREADS": threads}):
        return [f.copy() for f in render_engine._warp_frames(animator, FakeTrack(frames), include_mouth)], animator


class WarpFramesTests(unittest.TestCase):
    def test_frames_arrive_in_order_whatever_order_they_finish(self):
        random.seed(3)
        frames, _ = collect(frames=60)
        # pose[0] is the index, so the first element of each frame is >= its index and strictly rises with it
        firsts = [float(f[0, 0, 0]) for f in frames]
        self.assertEqual(len(frames), 60)
        self.assertEqual(firsts, sorted(firsts))

    def test_parallel_output_equals_the_serial_loop(self):
        random.seed(5)
        serial, _ = collect(frames=50, threads="1")
        parallel, _ = collect(frames=50, threads="4")
        for a, b in zip(serial, parallel, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_mouth_weights_are_dropped_when_the_mouth_is_repainted_elsewhere(self):
        # Wav2Lip repaints the mouth: only the non-mouth weights (here the blink) may reach the renderer.
        with_mouth, _ = collect(frames=6, include_mouth=True)
        without, _ = collect(frames=6, include_mouth=False)
        self.assertGreater(float(with_mouth[3][0, 0, 0]), float(without[3][0, 0, 0]))

    def test_a_failing_frame_raises_instead_of_being_skipped(self):
        with self.assertRaises(RuntimeError):
            collect(frames=30, animator=FakeAnimator(fail_at=12.0))

    def test_every_frame_is_rendered_exactly_once(self):
        _, animator = collect(frames=37)
        self.assertEqual(animator.calls, 37)

    def test_thread_count_setting(self):
        for raw, expected in (("4", 4), ("1", 1), ("0", 1), ("-3", 1), ("lots", 4), ("", 4)):
            with mock.patch.dict(os.environ, {"RENDER_THREADS": raw}):
                self.assertEqual(render_engine.render_threads(), expected, raw)


if __name__ == "__main__":
    unittest.main()


class PrefetchTests(unittest.TestCase):
    """``_prefetched``: the front of the render chain on its own thread (T9.3)."""

    def test_items_come_out_in_order_and_complete(self):
        out = list(render_engine._prefetched(iter(range(500)), depth=4))
        self.assertEqual(out, list(range(500)))

    def test_an_error_in_the_source_is_raised_for_the_consumer(self):
        def source():
            yield 1
            yield 2
            raise ValueError("encoder source broke")

        seen = []
        with self.assertRaises(ValueError):
            for item in render_engine._prefetched(source(), depth=2):
                seen.append(item)
        self.assertEqual(seen, [1, 2])

    def test_the_producer_never_runs_far_ahead_of_the_consumer(self):
        produced = []

        def source():
            for i in range(1000):
                produced.append(i)
                yield i

        stream = render_engine._prefetched(source(), depth=3)
        next(stream)
        time.sleep(0.3)
        # depth 3 queued + the one taken + the one the producer holds while waiting for room
        self.assertLessEqual(len(produced), 3 + 3)
        stream.close()

    def test_closing_early_stops_the_producer_and_closes_the_source(self):
        closed = []

        def source():
            try:
                for i in range(10_000):
                    yield i
            finally:
                closed.append(True)

        stream = render_engine._prefetched(source(), depth=2)
        next(stream)
        stream.close()
        time.sleep(0.2)
        self.assertEqual(closed, [True])
        self.assertFalse(any(t.name == "frame-prefetch" and t.is_alive() for t in __import__("threading").enumerate()))
