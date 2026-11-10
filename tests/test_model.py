import io
import json
import unittest
from http.client import RemoteDisconnected
from unittest.mock import patch

from propellab.model import LanguageModel, object_schema


class ModelRetryTest(unittest.TestCase):
    def test_retries_a_disconnected_inference_request(self):
        body = {
            "choices": [{"finish_reason": "stop", "message": {"content": '{"value":"ok"}'}}],
            "usage": {"prompt_tokens": 13, "completion_tokens": 5},
        }
        model = LanguageModel("http://127.0.0.1:8157/v1", "local", "chat", 5)
        with patch("propellab.model.urlopen", side_effect=[
            RemoteDisconnected("connection reset"), io.BytesIO(json.dumps(body).encode()),
        ]) as request, patch("propellab.model.time.sleep"):
            result = model.complete("Return the value", {}, object_schema({"value": {"type": "string"}}), 40)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(result.data, {"value": "ok"})
        self.assertEqual((result.input_tokens, result.output_tokens), (13, 5))
