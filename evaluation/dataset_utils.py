from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_RANDOM_SEED = 747
REQUIRED_COLUMNS = {"text", "labels", "source"}


def normalize_text(text: Any) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    return " ".join(text.split())


def compute_text_hash(text: Any) -> str:
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_essay_id(text: Any) -> str:
    return f"essay_{compute_text_hash(text)}"


def compute_dataset_hash(df: pd.DataFrame) -> str:
    payload = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        raise ValueError("Dataset is empty.")

    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    row_count = int(len(df))
    labels = pd.to_numeric(df["labels"], errors="coerce")
    binary_valid = labels.notna() & labels.isin([0, 1])
    invalid_label_count = int((~binary_valid).sum())
    unique_labels = sorted(labels[binary_valid].unique().astype(int).tolist())

    text_series = df["text"].astype(str)
    missing_text_count = int((df["text"].isna() | (df["text"].astype(str).str.strip() == "")).sum())
    exact_duplicates = int(df["text"].duplicated(keep='first').sum())

    normalized = df["text"].map(normalize_text)
    normalized_duplicates = int(normalized.duplicated(keep='first').sum())

    conflict_counts = normalized.groupby(normalized).apply(lambda s: df.loc[s.index, 'labels'].nunique() if s.index.size else 0)
    conflicting_label_rows = int((conflict_counts > 1).sum())

    source_counts = df["source"].fillna("<missing>").value_counts().sort_index().to_dict()
    class_distribution = labels[labels.notna()].astype(int).value_counts().sort_index().to_dict()

    word_counts = text_series.fillna("").str.split().str.len()
    char_counts = text_series.fillna("").str.len()
    source_summary = [
        {
            'source': source_name,
            'count': int(count),
            'human_count': int(df.loc[df['source'] == source_name, 'labels'].eq(0).sum()),
            'ai_count': int(df.loc[df['source'] == source_name, 'labels'].eq(1).sum()),
        }
        for source_name, count in source_counts.items()
    ]

    summary = {
        'required_columns': sorted(REQUIRED_COLUMNS),
        'original_row_count': row_count,
        'missing_required_columns': missing_columns,
        'label_validity': {'binary_valid_rows': int(binary_valid.sum()), 'invalid_rows': invalid_label_count, 'valid_values': unique_labels},
        'missing_empty_text_rows': missing_text_count,
        'exact_duplicate_rows': exact_duplicates,
        'normalized_duplicate_rows': normalized_duplicates,
        'conflicting_label_rows': conflicting_label_rows,
        'class_distribution': {str(k): int(v) for k, v in class_distribution.items()},
        'source_distribution': source_counts,
        'source_summary': source_summary,
        'word_count_summary': {
            'min': int(word_counts.min()) if not word_counts.empty else 0,
            'median': float(word_counts.median()) if not word_counts.empty else 0.0,
            'mean': float(word_counts.mean()) if not word_counts.empty else 0.0,
            'max': int(word_counts.max()) if not word_counts.empty else 0,
        },
        'character_count_summary': {
            'min': int(char_counts.min()) if not char_counts.empty else 0,
            'median': float(char_counts.median()) if not char_counts.empty else 0.0,
            'mean': float(char_counts.mean()) if not char_counts.empty else 0.0,
            'max': int(char_counts.max()) if not char_counts.empty else 0,
        },
        'source_imbalance': {
            'max_source_count': max(source_counts.values()) if source_counts else 0,
            'min_source_count': min(source_counts.values()) if source_counts else 0,
            'largest_to_smallest_ratio': (max(source_counts.values()) / min(source_counts.values())) if source_counts and min(source_counts.values()) else 0.0,
        },
        'dataset_sha256': compute_dataset_hash(df),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    return summary


def deduplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if 'essay_id' not in work.columns:
        work['essay_id'] = work['text'].map(build_essay_id)
    work = work[work['text'].notna() & work['text'].astype(str).str.strip().ne("")].copy()
    work['normalized_text'] = work['text'].map(normalize_text)
    work['labels'] = pd.to_numeric(work['labels'], errors='coerce')
    work = work[work['labels'].notna() & work['labels'].isin([0, 1])].copy()

    exact_duplicates = work.duplicated(subset=['text'], keep='first')
    work = work.loc[~exact_duplicates].copy()

    normalized_conflicts = work.groupby('normalized_text')['labels'].nunique()
    conflict_hashes = set(normalized_conflicts[normalized_conflicts > 1].index)
    work = work.loc[~work['normalized_text'].isin(conflict_hashes)].copy()

    work = work.drop_duplicates(subset=['normalized_text'], keep='first').copy()
    work.reset_index(drop=True, inplace=True)
    if 'essay_id' not in work.columns:
        work['essay_id'] = work['text'].map(build_essay_id)
    # collision guard
    ids = work['essay_id'].tolist()
    if len(set(ids)) != len(ids):
        raise ValueError('Essay ID collisions detected after deduplication.')
    return work[['essay_id', 'text', 'labels', 'source'] + [c for c in work.columns if c not in ['essay_id', 'text', 'labels', 'source']]].copy()


def make_standard_splits(df: pd.DataFrame, random_seed: int = DEFAULT_RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError('Cannot split an empty dataframe.')
    if df['labels'].nunique() <= 1:
        stratify = None
    else:
        stratify = df['labels']
    train_df, temp_df = train_test_split(
        df,
        train_size=0.70,
        random_state=random_seed,
        shuffle=True,
        stratify=stratify,
    )
    val_df, test_df = train_test_split(
        temp_df,
        train_size=0.5,
        random_state=random_seed,
        shuffle=True,
        stratify=temp_df['labels'] if temp_df['labels'].nunique() > 1 else None,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_holdout_sources(df: pd.DataFrame) -> set[str]:
    holdout = set()
    for source_name in df['source'].dropna().astype(str).unique():
        if source_name.startswith('Mistral7B-CME-') or source_name == 'persuade_2.0_human_scores_demo_id_github':
            holdout.add(source_name)
    return holdout


def ensure_split_manifest(source_hash: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, unseen_df: pd.DataFrame, random_seed: int, holdout_sources: list[str]) -> dict[str, Any]:
    def fingerprint(frame: pd.DataFrame) -> str:
        joined = ''.join(frame['essay_id'].sort_values().tolist()) if not frame.empty else ''
        return hashlib.sha256(joined.encode('utf-8')).hexdigest()

    manifest = {
        'dataset_hash': source_hash,
        'random_seed': int(random_seed),
        'creation_time_utc': datetime.now(timezone.utc).isoformat(),
        'split_proportions': {'train': 0.70, 'validation': 0.15, 'test': 0.15, 'unseen_test': 0.0},
        'split_counts': {
            'train': int(len(train_df)),
            'validation': int(len(val_df)),
            'test': int(len(test_df)),
            'unseen_test': int(len(unseen_df)),
        },
        'class_distributions': {
            'train': train_df['labels'].value_counts().sort_index().to_dict(),
            'validation': val_df['labels'].value_counts().sort_index().to_dict(),
            'test': test_df['labels'].value_counts().sort_index().to_dict(),
            'unseen_test': unseen_df['labels'].value_counts().sort_index().to_dict(),
        },
        'split_fingerprints': {
            'train': fingerprint(train_df),
            'validation': fingerprint(val_df),
            'test': fingerprint(test_df),
            'unseen_test': fingerprint(unseen_df),
        },
        'held_out_sources': sorted(holdout_sources),
        'note': 'Processing is deterministic with the fixed seed and source-family holdout policy.',
    }
    return manifest


def write_split_assignments(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, unseen_df: pd.DataFrame, output_path) -> None:
    rows = []
    for split_name, frame in [('train', train_df), ('validation', val_df), ('test', test_df), ('unseen_test', unseen_df)]:
        for essay_id in frame['essay_id'].tolist():
            rows.append({'essay_id': essay_id, 'split': split_name})
    assignments = pd.DataFrame(rows)
    assignments.to_csv(output_path, index=False)
