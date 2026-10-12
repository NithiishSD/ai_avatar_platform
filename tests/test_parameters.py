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

    def test_target_is_derived_from_the_measured_count_not_set_by_hand(self):
        out = parameters.catalogue()
        measured = out["counts"]["visualMeasuredWorking"]
        self.assertEqual(out["target"]["met"], measured >= parameters.TARGET)
        self.assertEqual(out["counts"]["listed"], len(out["parameters"]))
        self.assertGreaterEqual(out["counts"]["noEffect"], 1)  # `neutral` is reported, not hidden

    def test_the_target_falls_back_to_not_met_when_the_evidence_is_removed(self):
        # Every appearance / expression row downgraded to "not measured": nothing is verified, so 50+ is not met.
        original = parameters._ROWS
        try:
            parameters._ROWS = [
                (m, f, g, parameters.NM if g in ("appearance", "expression") else s, n) for m, f, g, s, n in original
            ]
            out = parameters.catalogue()
            self.assertFalse(out["target"]["met"])
            self.assertEqual(out["counts"]["visualMeasuredWorking"], 0)
        finally:
            parameters._ROWS = original

    def test_appearance_rows_carry_their_ranges_from_the_contract(self):
        by_name = {r["name"]: r for r in parameters.catalogue()["parameters"]}
        self.assertEqual((by_name["appearance.brightness"]["minimum"], by_name["appearance.brightness"]["maximum"]), (-1.0, 1.0))
        self.assertEqual((by_name["appearance.gamma"]["minimum"], by_name["appearance.gamma"]["maximum"]), (0.5, 2.0))
        self.assertEqual(by_name["appearance.autoLighting"]["type"], "boolean")
        self.assertEqual((by_name["appearance.expression.jawOpen"]["minimum"], by_name["appearance.expression.jawOpen"]["maximum"]), (0.0, 1.0))
        self.assertEqual(len([n for n in by_name if n.startswith("appearance.expression.")]), 25)

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
        self.assertEqual(body["target"]["met"], body["counts"]["visualMeasuredWorking"] >= body["target"]["appearanceParameters"])


if __name__ == "__main__":
    unittest.main()
