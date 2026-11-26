from __future__ import annotations

import unittest

import numpy as np

from evaluation.model_adapters import ModelNotAvailableError
from models.custom_model.adapter import CustomEnsembleAdapter


class FakeMember:
    def __init__(self, ai, human):
        self.ai, self.human = ai, human

    def predict_proba(self, texts):
        return np.asarray([self.ai if "AI" in t else self.human for t in texts])


class AdapterTests(unittest.TestCase):
    def test_empty_input_is_empty_probability_vector(self):
        self.assertEqual(CustomEnsembleAdapter("missing_test_artifacts").predict_proba([]).shape, (0,))

    def test_missing_artifacts_raise_clear_error(self):
        adapter = CustomEnsembleAdapter("missing_test_artifacts")
        with self.assertRaisesRegex(ModelNotAvailableError, "combiner.json"):
            adapter.predict_proba(["some text"])

    def test_valid_output_shape_range_and_ai_orientation(self):
        adapter = CustomEnsembleAdapter.__new__(CustomEnsembleAdapter)
        adapter.model_ready = True
        adapter.members = [FakeMember(.9, .2), FakeMember(.8, .1)]
        adapter.weights = np.asarray([1.0, 0.5])
        adapter.offset = 0.0
        result = adapter.predict_proba(["human essay", "AI essay", "another human one"])
        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all())
        self.assertGreater(result[1], 0.5)
        self.assertLess(result[0], 0.5)
        logit = lambda p: np.log(p / (1 - p))  # noqa: E731
        self.assertAlmostEqual(result[1], 1 / (1 + np.exp(-(logit(.9) + .5 * logit(.8)))))


if __name__ == "__main__":
    unittest.main()
