from __future__ import annotations

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from evaluation.model_adapters import ModelNotAvailableError
from models.aakash_7th_place.adapter import AakashSeventhPlaceAdapter


class FakeTokenizer:
    def encode_batch(self, texts, add_special_tokens=False):
        return [type("Encoded", (), {"tokens": text.split()}) for text in texts]


class FakeVectorizer:
    def transform(self, tokens):
        return csr_matrix([[1.0, 0.0] if "AI" in t else [0.0, 1.0] for t in tokens])


class AdapterTests(unittest.TestCase):
    def test_empty_input_is_empty_probability_vector(self):
        self.assertEqual(AakashSeventhPlaceAdapter("missing_test_artifacts").predict_proba([]).shape, (0,))

    def test_missing_artifacts_raise_clear_error(self):
        adapter = AakashSeventhPlaceAdapter("missing_test_artifacts")
        with self.assertRaisesRegex(ModelNotAvailableError, "Missing fitted model artifacts"):
            adapter.predict_proba(["some text"])

    def test_valid_output_shape_range_and_ai_orientation(self):
        adapter = AakashSeventhPlaceAdapter.__new__(AakashSeventhPlaceAdapter)
        adapter.model_ready = True
        adapter.tokenizer, adapter.vectorizer = FakeTokenizer(), FakeVectorizer()
        # feature 0 votes AI (class 1), feature 1 votes human (class 0)
        adapter.nb_log_prob = np.log([[0.1, 0.9], [0.9, 0.1]])
        adapter.nb_log_prior = np.log([0.5, 0.5])
        adapter.sgd_coef, adapter.sgd_intercept = np.array([[2.0, -2.0]]), np.array([0.0])
        adapter.weights = [0.43, 0.57]
        result = adapter.predict_proba(["human example", "AI generated sample"])
        self.assertEqual(result.shape, (2,))
        self.assertTrue(np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all())
        self.assertTrue(np.allclose(result, [0.043, 0.957]))


if __name__ == "__main__":
    unittest.main()
