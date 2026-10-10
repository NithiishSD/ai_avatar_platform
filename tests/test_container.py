"""The Docker image's rules (T7.2): not root, no weights baked in, and the studio served without hiding the API."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DockerfileTests(unittest.TestCase):
    def setUp(self):
        self.dockerfile = (ROOT / "Dockerfile").read_text()

    def test_the_api_runs_as_a_non_root_user_with_a_health_check(self):
        users = [line.split()[1] for line in self.dockerfile.splitlines() if line.startswith("USER ")]
        self.assertEqual(users, ["app"])                 # S-16
        self.assertIn("HEALTHCHECK", self.dockerfile)

    def test_weights_and_media_stay_out_of_the_image(self):
        ignored = (ROOT / ".dockerignore").read_text().split()
        for path in (".models", "backend/.conda", "inputs", "outputs", "docs", ".git"):
            self.assertIn(path, ignored)
        self.assertNotIn("COPY .models", self.dockerfile)


class StudioMountTests(unittest.TestCase):
    def test_the_built_studio_is_served_at_the_root_and_the_api_still_answers(self):
        with tempfile.TemporaryDirectory() as dist:
            Path(dist, "index.html").write_text("<html>studio</html>")
            probe = (
                "from fastapi.testclient import TestClient\n"
                "import app\n"
                "c = TestClient(app.app)\n"
                "print(c.get('/').text.strip(), c.get('/health').headers['content-type'].split(';')[0], c.get('/api/v1/parameters').status_code)\n"
            )
            env = {**os.environ, "FRONTEND_DIST": dist, "PYTHONPATH": f"{ROOT / 'backend'}{os.pathsep}{ROOT / 'tests'}", "JOBS_DB": ":memory:"}
            out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env, cwd=ROOT / "backend", timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertEqual(out.stdout.strip().splitlines()[-1], "<html>studio</html> application/json 200")


if __name__ == "__main__":
    unittest.main()
