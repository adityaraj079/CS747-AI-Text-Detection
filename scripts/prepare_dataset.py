#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.dataset_utils import (
    DEFAULT_RANDOM_SEED,
    build_holdout_sources,
    compute_dataset_hash,
    deduplicate_records,
    ensure_split_manifest,
    make_standard_splits,
    validate_dataset,
    write_split_assignments,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CSV = REPO_ROOT.parent / 'Progress' / 'data' / 'CS747data.csv'


def main():
    parser = argparse.ArgumentParser(description='Validate and split the CS747 source dataset.')
    parser.add_argument('--source-csv', type=str, default=str(DEFAULT_SOURCE_CSV))
    parser.add_argument('--rebuild', action='store_true', help='Require explicit rebuild confirmation before overwriting existing processed datasets.')
    args = parser.parse_args()
    source_csv = Path(args.source_csv).expanduser().resolve()
    if not source_csv.exists():
        print(f'Source CSV not found: {source_csv}')
        print('To proceed after supplying the dataset, run:')
        print(f'python {REPO_ROOT / "scripts" / "prepare_dataset.py"} --source-csv "/absolute/path/to/CS747data.csv" --rebuild')
        return 0

    output_dir = REPO_ROOT / 'data'
    processed_dir = output_dir / 'processed'
    manifests_dir = output_dir / 'manifests'
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source_csv)
    quality = validate_dataset(df)
    quality['source_csv'] = str(source_csv)
    quality['dataset_hash'] = compute_dataset_hash(df)
    (manifests_dir / 'data_quality_report.json').write_text(json.dumps(quality, indent=2, sort_keys=True))

    accepted = deduplicate_records(df)
    accepted = accepted.reset_index(drop=True)
    if accepted.empty:
        raise ValueError('No accepted essays remain after validation and deduplication.')

    holdout_sources = build_holdout_sources(accepted)
    unseen_df = accepted[accepted['source'].isin(holdout_sources)].copy().reset_index(drop=True)
    standard_pool = accepted[~accepted['source'].isin(holdout_sources)].copy().reset_index(drop=True)

    train_df, val_df, test_df = make_standard_splits(standard_pool, random_seed=DEFAULT_RANDOM_SEED)

    train_path = processed_dir / 'train.csv'
    val_path = processed_dir / 'validation.csv'
    test_path = processed_dir / 'test.csv'
    unseen_path = processed_dir / 'unseen_test.csv'

    for path, frame in [(train_path, train_df), (val_path, val_df), (test_path, test_df)]:
        if path.exists() and not args.rebuild:
            raise FileExistsError(f'Refusing to overwrite existing split: {path}. Re-run with --rebuild to confirm.')
        frame = frame[['essay_id', 'text', 'labels', 'source'] + [c for c in frame.columns if c not in ['essay_id', 'text', 'labels', 'source']]].copy()
        frame.to_csv(path, index=False)

    if unseen_path.exists() and not args.rebuild:
        raise FileExistsError(f'Refusing to overwrite existing unseen test set: {unseen_path}. Re-run with --rebuild to confirm.')
    unseen_df = unseen_df[['essay_id', 'text', 'labels', 'source'] + [c for c in unseen_df.columns if c not in ['essay_id', 'text', 'labels', 'source']]].copy()
    unseen_df.to_csv(unseen_path, index=False)

    split_manifest = ensure_split_manifest(
        source_hash=quality['dataset_hash'],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        unseen_df=unseen_df,
        random_seed=DEFAULT_RANDOM_SEED,
        holdout_sources=sorted(holdout_sources),
    )
    (manifests_dir / 'split_manifest.json').write_text(json.dumps(split_manifest, indent=2, sort_keys=True))
    write_split_assignments(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        unseen_df=unseen_df,
        output_path=manifests_dir / 'split_assignments.csv',
    )

    unseen_manifest = {
        'selection_strategy': 'source_family_holdout',
        'held_out_sources': sorted(holdout_sources),
        'class_distribution': unseen_df['labels'].value_counts().sort_index().to_dict(),
        'record_count': int(len(unseen_df)),
        'human_count': int((unseen_df['labels'] == 0).sum()),
        'ai_count': int((unseen_df['labels'] == 1).sum()),
        'exclusion_rules': [
            'Hold out all source families explicitly identified as Mistral7B-CME variants.',
            'Hold out the human source persuade_2.0_human_scores_demo_id_github to preserve a class-balanced OOD evaluation set.',
            'Remove all selected holdout sources from the standard 70/15/15 split.',
        ],
    }
    (manifests_dir / 'unseen_test_manifest.json').write_text(json.dumps(unseen_manifest, indent=2, sort_keys=True))
    print(json.dumps({
        'dataset_hash': quality['dataset_hash'],
        'accepted_rows': int(len(accepted)),
        'training_rows': int(len(train_df)),
        'validation_rows': int(len(val_df)),
        'test_rows': int(len(test_df)),
        'unseen_rows': int(len(unseen_df)),
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    raise SystemExit(main())
