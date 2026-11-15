from __future__ import annotations

import json

import pandas as pd

from scripts.evaluate_registered_model import _load_partition


def test_standard_test_alias_loads_test_partition_and_manifest(tmp_path):
    processed = tmp_path / "processed"
    manifests = tmp_path / "manifests"
    processed.mkdir()
    manifests.mkdir()
    pd.DataFrame(
        {
            "essay_id": ["essay-a", "essay-b"],
            "text": ["A human essay.", "An AI essay."],
            "labels": [0, 1],
        }
    ).to_csv(processed / "test.csv", index=False)
    (manifests / "split_manifest.json").write_text(
        json.dumps(
            {
                "dataset_hash": "dataset-hash",
                "split_fingerprints": {"test": "test-fingerprint"},
                "split_counts": {"test": 2},
            }
        ),
        encoding="utf-8",
    )

    frame, manifest, split_id = _load_partition(tmp_path, "standard_test")

    assert len(frame) == 2
    assert manifest["dataset_hash"] == "dataset-hash"
    assert split_id == "test-fingerprint"
