import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from agent import InferenceAgent


class TestInferenceAgent(unittest.TestCase):
    def setUp(self):
        self.agent = InferenceAgent()

    def test_preprocess_shape_and_dtype(self):
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.png"

            image = Image.new(
                "RGB",
                size=(32, 32),
                color=(128, 64, 32),
            )
            image.save(image_path)

            tensor = self.agent.preprocess(image_path)

        self.assertEqual(
            tuple(tensor.shape),
            (1, 3, 224, 224),
        )
        self.assertEqual(str(tensor.dtype), "torch.float32")

    def test_missing_image_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            self.agent.preprocess("/tmp/does_not_exist.png")

    def test_parse_logits(self):
        response = {
            "outputs": [
                {
                    "name": "output",
                    "data": [1.0, 2.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                }
            ]
        }

        logits = self.agent.parse_logits(response)

        self.assertEqual(len(logits), 10)
        self.assertEqual(logits[2], 8.0)

    def test_parse_logits_rejects_missing_outputs(self):
        with self.assertRaises(ValueError):
            self.agent.parse_logits({})

    def test_predict_parses_response(self):
        response = {
            "outputs": [
                {
                    "name": "output",
                    "data": [1.0, 2.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                }
            ]
        }

        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.png"

            image = Image.new(
                "RGB",
                size=(32, 32),
                color=(128, 64, 32),
            )
            image.save(image_path)

            with patch.object(
                self.agent,
                "send_request",
                return_value=response,
            ):
                result = self.agent.predict(image_path)

        self.assertEqual(result.predicted_class, 2)
        self.assertEqual(result.class_name, "bird")
        self.assertGreater(result.confidence, 0.0)
        self.assertGreaterEqual(result.inference_time_ms, 0.0)
        self.assertEqual(len(result.logits), 10)


if __name__ == "__main__":
    unittest.main()
