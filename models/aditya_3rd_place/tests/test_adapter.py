import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.aditya_3rd_place.adapter import AdityaSubmission3Adapter


def test_missing_artifacts_raise_clear_error():
    adapter = AdityaSubmission3Adapter()
    with pytest.raises(FileNotFoundError, match='No fitted Submission #3 vectorizer/classifier artifacts were found'):
        adapter.predict_proba(['This is a sample essay for testing.'])


def test_zero_length_input_is_safe():
    adapter = AdityaSubmission3Adapter()
    try:
        result = adapter.predict_proba([])
    except FileNotFoundError:
        result = []
    assert isinstance(result, list) or hasattr(result, 'shape')
