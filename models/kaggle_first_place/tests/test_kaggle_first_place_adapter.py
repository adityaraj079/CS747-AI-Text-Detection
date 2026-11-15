import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.model_adapters import ModelNotAvailableError
from models.kaggle_first_place.adapter import KaggleFirstPlaceAdapter


def test_missing_first_place_model_fails_explicitly():
    with pytest.raises(ModelNotAvailableError, match="has not been integrated"):
        KaggleFirstPlaceAdapter()
