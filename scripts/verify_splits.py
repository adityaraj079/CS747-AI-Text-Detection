#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PROCESSED = REPO / 'data' / 'processed'
MANIFESTS = REPO / 'data' / 'manifests'


def verify():
    required = ['train.csv', 'validation.csv', 'test.csv']
    for name in required:
        path = PROCESSED / name
        if not path.exists():
            raise FileNotFoundError(f'Missing split file: {path}')
    train = pd.read_csv(PROCESSED / 'train.csv')
    val = pd.read_csv(PROCESSED / 'validation.csv')
    test = pd.read_csv(PROCESSED / 'test.csv')
    unseen = pd.read_csv(PROCESSED / 'unseen_test.csv') if (PROCESSED / 'unseen_test.csv').exists() else pd.DataFrame()

    all_frames = [train, val, test, unseen] if not unseen.empty else [train, val, test]
    ids = [df['essay_id'] for df in all_frames]
    merged = pd.concat(ids, ignore_index=True)
    if merged.duplicated().any():
        raise AssertionError('An essay_id appears in multiple splits.')

    manifest = json.loads((MANIFESTS / 'split_manifest.json').read_text())
    split_counts = {
        'train': len(train),
        'validation': len(val),
        'test': len(test),
        'unseen_test': len(unseen),
    }
    for key, value in split_counts.items():
        if key in manifest['split_counts'] and value != manifest['split_counts'][key]:
            raise AssertionError(f'Count mismatch for {key}: {value} vs {manifest["split_counts"][key]}')

    print('Verified split counts and disjoint essay IDs.')


if __name__ == '__main__':
    verify()
