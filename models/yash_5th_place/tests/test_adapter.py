from __future__ import annotations

import numpy as np
import unittest

from models.yash_5th_place.adapter import YashFifthPlaceAdapter
from evaluation.model_adapters import ModelNotAvailableError


class FakeVectorizer:
    def transform(self, texts):
        return np.asarray([[len(text)] for text in texts])


class FakeClassifier:
    classes_ = np.asarray([0, 1])

    def __init__(self, value):
        self.value = value

    def predict_proba(self, features):
        return np.column_stack([np.full(len(features), 1 - self.value), np.full(len(features), self.value)])


class FakeTokenizer:
    def encode_batch(self, texts):
        return [type("Encoded", (), {"tokens": text.split()}) for text in texts]


class FakeBooster:
    def predict(self, features, num_threads=1):
        return np.full(len(features), .8)


class AdapterTests(unittest.TestCase):
  def test_empty_input_is_empty_probability_vector(self):
    adapter = YashFifthPlaceAdapter("missing_test_artifacts")
    result = adapter.predict_proba([])
    self.assertEqual(result.shape, (0,))


  def test_missing_artifacts_raise_clear_error(self):
    adapter = YashFifthPlaceAdapter("missing_test_artifacts")
    with self.assertRaisesRegex(ModelNotAvailableError, "Missing fitted model artifacts"):
      adapter.predict_proba(["some text"])


  def test_valid_output_shape_range_and_ai_orientation(self):
    adapter = YashFifthPlaceAdapter.__new__(YashFifthPlaceAdapter)
    adapter.model_ready = True
    adapter.tokenizer = FakeTokenizer()
    adapter.vectorizer = FakeVectorizer()
    adapter.classifiers = [FakeClassifier(.6) for _ in range(6)]
    adapter.booster = FakeBooster()
    result = adapter.predict_proba(["human example", "AI generated sample"])
    self.assertEqual(result.shape, (2,))
    self.assertTrue(np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all())
    self.assertTrue(np.allclose(result, [.68, .68]))


if __name__ == "__main__":
    unittest.main()
