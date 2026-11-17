from __future__ import annotations

import unittest

import numpy as np

from evaluation.model_adapters import ModelNotAvailableError
from models.daniel_bert.adapter import DanielBertAdapter


class FakeTokenizer:
    def __call__(self, texts, padding=True, truncation=True, max_length=512, return_tensors=None):
        self.last_batch = list(texts)
        return {"input_ids": np.ones((len(texts), 2), dtype=np.int64), "attention_mask": np.ones((len(texts), 2), dtype=np.int64)}


class FakeModel:
    def __call__(self, **kwargs):
        logits = np.array([
            [1.0, 3.0],
            [3.0, 1.0],
        ], dtype=float)
        return type("Outputs", (), {"logits": logits})()


class DanielBertAdapterTests(unittest.TestCase):
    def test_empty_input_is_empty_probability_vector(self):
        self.assertEqual(DanielBertAdapter("missing_test_artifacts").predict_proba([]).shape, (0,))

    def test_missing_artifacts_raise_clear_error(self):
        adapter = DanielBertAdapter("missing_test_artifacts")
        with self.assertRaisesRegex(ModelNotAvailableError, "Missing fitted model checkpoint"):
            adapter.predict_proba(["some text"])

    def test_valid_output_shape_range_and_ai_orientation(self):
        adapter = DanielBertAdapter.__new__(DanielBertAdapter)
        adapter.model_ready = True
        adapter.model = FakeModel()
        adapter.tokenizer = FakeTokenizer()
        result = adapter.predict_proba(["stub essay", "another essay"])
        self.assertEqual(result.shape, (2,))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.all((result >= 0) & (result <= 1)))
        self.assertGreater(result[0], 0.5)
        self.assertLess(result[1], 0.5)


if __name__ == "__main__":
    unittest.main()
