"""Request ids and log stamping (R-27)."""

import logging
import time
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

import request_context
from generation_jobs import GenerationJobs
from request_context import REQUEST_ID_HEADER

request_context.install_logging()
probe_logger = logging.getLogger("probe")


def make_app() -> FastAPI:
    app = FastAPI()
    app.middleware("http")(request_context.request_id_middleware)

    @app.get("/work")
    def work():
        probe_logger.warning("inside the endpoint")
        return {"id": request_context.current_request_id()}

    @app.get("/boom")
    def boom():
        raise RuntimeError("unhandled")

    return app


class RecordingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


class RequestIdTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(make_app(), raise_server_exceptions=False)
        self.handler = RecordingHandler()
        probe_logger.addHandler(self.handler)
        self.addCleanup(probe_logger.removeHandler, self.handler)

    def test_every_response_has_an_id_and_the_endpoint_log_line_carries_it(self):
        response = self.client.get("/work")
        request_id = response.headers[REQUEST_ID_HEADER]
        self.assertEqual(len(request_id), 16)
        self.assertEqual(response.json()["id"], request_id)
        self.assertEqual([r.request_id for r in self.handler.records], [request_id])

    def test_a_safe_incoming_id_is_kept_so_a_caller_can_trace_its_own_request(self):
        response = self.client.get("/work", headers={REQUEST_ID_HEADER: "trace-42.a_b"})
        self.assertEqual(response.headers[REQUEST_ID_HEADER], "trace-42.a_b")
        self.assertEqual(self.handler.records[0].request_id, "trace-42.a_b")

    def test_an_unsafe_incoming_id_is_replaced_not_logged(self):
        for bad in ("has space", "x" * 65, "a\\nforged line", "<script>"):
            response = self.client.get("/work", headers={REQUEST_ID_HEADER: bad})
            self.assertNotEqual(response.headers[REQUEST_ID_HEADER], bad)
            self.assertRegex(response.headers[REQUEST_ID_HEADER], r"^[0-9a-f]{16}$")

    def test_concurrent_requests_do_not_share_an_id(self):
        ids = {self.client.get("/work").headers[REQUEST_ID_HEADER] for _ in range(5)}
        self.assertEqual(len(ids), 5)

    def test_the_id_does_not_leak_past_the_request(self):
        self.client.get("/work")
        self.assertEqual(request_context.current_request_id(), request_context.NO_REQUEST)

    def test_an_unhandled_error_response_still_carries_an_id(self):
        response = self.client.get("/boom")
        self.assertEqual(response.status_code, 500)
        request_id = response.headers[REQUEST_ID_HEADER]
        self.assertEqual(response.json()["requestId"], request_id)
        self.assertIn(request_id, response.json()["detail"])


class BackgroundWorkTests(unittest.TestCase):
    def test_a_queued_job_logs_under_the_request_that_queued_it(self):
        handler = RecordingHandler()
        probe_logger.addHandler(handler)
        self.addCleanup(probe_logger.removeHandler, handler)
        jobs = GenerationJobs()

        def work():
            probe_logger.warning("in the worker thread")
            return {}

        token = request_context._request_id.set("req-from-http")
        try:
            task = jobs.submit(work)
        finally:
            request_context._request_id.reset(token)
        for _ in range(100):
            if jobs.get(task)["status"] == "COMPLETED":
                break
            time.sleep(0.02)
        self.assertEqual([r.request_id for r in handler.records], ["req-from-http"])


class RealAppTests(unittest.TestCase):
    def test_the_real_api_returns_an_id(self):
        import app as app_module

        response = TestClient(app_module.app).get("/health")
        self.assertRegex(response.headers[REQUEST_ID_HEADER], r"^[0-9a-f]{16}$")


if __name__ == "__main__":
    unittest.main()
