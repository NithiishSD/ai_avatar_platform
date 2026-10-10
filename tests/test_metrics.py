"""Tests for the metrics summary (R-51): the arithmetic, the empty case, and the endpoint."""

import unittest
from types import SimpleNamespace

import metrics
from contracts import JobStatus


def job(status, result=None):
    return SimpleNamespace(status=JobStatus(status), result=result)


def result(render_s, realtime, engine="blendshape", marked=True, lipsync=None, vram=None, warnings=()):
    body = {"engine": engine, "renderSeconds": render_s, "realtimeFactor": realtime, "peakVramMb": vram,
            "watermark": {"applied": marked}, "warnings": list(warnings)}
    if lipsync:
        body["lipsync"] = lipsync
    return body


class SpreadTests(unittest.TestCase):
    def test_empty_list_reports_nothing_not_zero(self):
        self.assertEqual(metrics.spread([]), {"n": 0, "mean": None, "p50": None, "p95": None, "max": None})

    def test_nearest_rank_percentiles_are_observed_values(self):
        stats = metrics.spread([float(i) for i in range(1, 21)])  # 1..20
        self.assertEqual((stats["n"], stats["mean"], stats["p50"], stats["p95"], stats["max"]), (20, 10.5, 10.0, 19.0, 20.0))

    def test_single_value(self):
        self.assertEqual(metrics.spread([4.0])["p95"], 4.0)


class SummariseTests(unittest.TestCase):
    def test_no_jobs(self):
        out = metrics.summarise([])
        self.assertEqual(out["queue"], {"jobsByState": {}, "waiting": 0, "running": 0})
        self.assertEqual(out["renders"]["finished"], 0)
        self.assertIsNone(out["renders"]["renderSeconds"]["mean"])
        self.assertEqual(out["quality"]["lipSyncScored"], 0)

    def test_counts_timings_marks_and_scores(self):
        out = metrics.summarise([
            job("QUEUED"), job("PROCESSING"), job("FAILED"),
            job("COMPLETED", result(2.0, 0.5, lipsync={"lseC": 4.0, "lseD": 10.0, "offsetFrames": 0})),
            job("COMPLETED", result(6.0, 1.5, engine="wav2lip", marked=False, warnings=["w"],
                                    lipsync={"lseC": 8.0, "lseD": 6.0, "offsetFrames": -4}, vram=3000.0)),
        ])
        self.assertEqual(out["queue"]["jobsByState"], {"QUEUED": 1, "PROCESSING": 1, "FAILED": 1, "COMPLETED": 2})
        self.assertEqual((out["queue"]["waiting"], out["queue"]["running"]), (1, 1))
        renders = out["renders"]
        self.assertEqual((renders["finished"], renders["byEngine"]), (2, {"blendshape": 1, "wav2lip": 1}))
        self.assertEqual((renders["renderSeconds"]["mean"], renders["renderSeconds"]["max"]), (4.0, 6.0))
        self.assertEqual((renders["watermarked"], renders["notWatermarked"], renders["withWarnings"]), (1, 1, 1))
        self.assertEqual(renders["peakVramMb"]["n"], 1)
        quality = out["quality"]
        self.assertEqual((quality["lipSyncScored"], quality["lseC"]["mean"], quality["offsetWithinOneFrame"]), (2, 6.0, 1))

    def test_a_result_missing_fields_does_not_break_the_report(self):
        out = metrics.summarise([job("COMPLETED", {"engine": "blendshape"})])
        self.assertEqual(out["renders"]["finished"], 1)
        self.assertEqual(out["renders"]["renderSeconds"]["n"], 0)


if __name__ == "__main__":
    unittest.main()
