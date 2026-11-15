import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.aditya_3rd_place.adapter import AdityaSubmission3Adapter


def test_missing_artifacts_raise_clear_error(tmp_path):
    adapter = AdityaSubmission3Adapter(artifact_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing fitted adaptation artifacts"):
        adapter.predict_proba(['This is a sample essay for testing.'])


def test_zero_length_input_is_safe(tmp_path):
    adapter = AdityaSubmission3Adapter(artifact_dir=tmp_path)
    result = adapter.predict_proba([])
    assert result.shape == (0,)


def test_packaged_adaptation_predicts_ai_probability():
    adapter = AdityaSubmission3Adapter()
    assert adapter.model_ready
    probabilities = adapter.predict_proba(
        [
            "A detailed essay presents its argument, supports each claim with evidence, "
            "and explains how the conclusion follows from the preceding analysis.",
            "This text discusses a topic using examples and connected paragraphs to "
            "develop its main idea in a clear and organized way.",
        ]
    )
    assert probabilities.shape == (2,)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
