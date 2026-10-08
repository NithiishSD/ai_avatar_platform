"""The parameter catalogue must cover every customisation field and name only fields that exist (N-15)."""

import unittest

import parameters


class CatalogueTests(unittest.TestCase):
    def test_every_customisation_field_has_a_row(self):
        self.assertEqual(parameters.unlisted_fields(), [], "add a row to parameters._ROWS (or declare it plumbing)")

    def test_rows_read_their_schema_from_the_models(self):
        by_name = {r["name"]: r for r in parameters.catalogue()["parameters"]}
        self.assertEqual((by_name["speed"]["minimum"], by_name["speed"]["maximum"]), (0.5, 2.0))
        self.assertEqual(by_name["hair"]["enum"][0], "short-dark")
        self.assertEqual(by_name["background.color"]["pattern"], "^#[0-9a-fA-F]{6}$")
        self.assertEqual(by_name["emotionVector.eyeblinkRate"]["maximum"], 10.0)
        self.assertEqual(by_name["targetFps"]["maximum"], 120)

    def test_names_are_unique_and_statuses_are_known(self):
        rows = parameters.catalogue()["parameters"]
        self.assertEqual(len({r["name"] for r in rows}), len(rows))
        known = {parameters.M, parameters.NO, parameters.LIM, parameters.UNAV, parameters.NM}
        self.assertTrue({r["status"] for r in rows} <= known)

    def test_target_is_reported_as_not_met_with_the_counts_behind_it(self):
        out = parameters.catalogue()
        self.assertFalse(out["target"]["met"])
        self.assertLess(out["counts"]["visualMeasuredWorking"], parameters.TARGET)
        self.assertEqual(out["counts"]["listed"], len(out["parameters"]))
        self.assertGreaterEqual(out["counts"]["noEffect"], 1)  # `neutral` is reported, not hidden

    def test_a_new_unlisted_field_is_caught(self):
        original = parameters._ROWS
        try:
            parameters._ROWS = [row for row in original if row[1] != "glasses"]
            self.assertEqual(parameters.unlisted_fields(), ["AvatarGenerateRequest.glasses"])
        finally:
            parameters._ROWS = original


class EndpointTests(unittest.TestCase):
    def test_endpoint_serves_the_catalogue(self):
        from unittest import mock

        from fastapi.testclient import TestClient

        import app as app_module

        with mock.patch.object(app_module.security_gate, "inspect", return_value=(True, 0, {})):
            body = TestClient(app_module.app).get("/api/v1/parameters").json()
        self.assertEqual(body["counts"]["listed"], len(body["parameters"]))
        self.assertFalse(body["target"]["met"])


if __name__ == "__main__":
    unittest.main()
