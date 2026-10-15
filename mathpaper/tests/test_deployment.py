"""Public hosting must not expose corpus writes or enable paid chat by default."""
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from mathpaper.app import app


class PublicDeploymentTests(unittest.TestCase):
    def test_upload_blocked_and_not_advertised_in_public_mode(self):
        with patch.dict(os.environ, {"MATHPAPER_PUBLIC_DEMO": "1"}), TestClient(app) as client:
            result = client.post("/api/upload", files={"file": ("sample.pdf", b"%PDF-placeholder", "application/pdf")})
            self.assertEqual(result.status_code, 403)
            self.assertFalse(client.get("/api/status").json()["can_upload"])

    def test_model_configuration_does_not_enable_public_chat(self):
        with patch.dict(os.environ, {"MATHPAPER_PUBLIC_DEMO": "1", "MATHPAPER_PUBLIC_CHAT": "0", "MATHPAPER_CHAT_MODEL": "configured-but-not-enabled"}), TestClient(app) as client:
            result = client.post("/api/chat", json={"question": "A question", "thread_id": "public-test"})
            self.assertEqual(result.status_code, 403)
            self.assertFalse(client.get("/api/status").json()["chat_configured"])


if __name__ == "__main__":
    unittest.main()
