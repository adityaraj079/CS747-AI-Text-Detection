#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / 'data'
FIG_DIR = DATA_DIR / 'figures'
MANIFESTS = DATA_DIR / 'manifests'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_split_data():
    splits = {}
    for name in ['train', 'validation', 'test']:
        path = DATA_DIR / 'processed' / f'{name}.csv'
        splits[name] = pd.read_csv(path)
    unseen_path = DATA_DIR / 'processed' / 'unseen_test.csv'
    if unseen_path.exists():
        splits['unseen_test'] = pd.read_csv(unseen_path)
    return splits


def build_figures():
    splits = load_split_data()
    manifest = json.loads((MANIFESTS / 'split_manifest.json').read_text())
    quality = json.loads((MANIFESTS / 'data_quality_report.json').read_text())
    source_counts = pd.concat([df[['source', 'labels']] for df in splits.values()], ignore_index=True).groupby('source').size().sort_values(ascending=False)
    accepted_rows = sum(manifest.get('split_counts', {}).values())
    original_rows = int(quality.get('original_row_count', 0))
    duplicate_rows = int(quality.get('normalized_duplicate_rows', 0)) + int(quality.get('conflicting_label_rows', 0)) + int(quality.get('missing_empty_text_rows', 0))
    source_count = len(quality.get('source_distribution', {}))
    human_total = int(quality.get('class_distribution', {}).get('0', 0))
    ai_total = int(quality.get('class_distribution', {}).get('1', 0))

    # Figure 1: split distribution
    counts = [len(splits.get('train', [])), len(splits.get('validation', [])), len(splits.get('test', [])), len(splits.get('unseen_test', [])) if 'unseen_test' in splits else 0]
    labels = ['Train', 'Validation', 'Standard Test', 'Unseen Test']
    total = sum(counts)
    percentages = [c / total * 100 for c in counts] if total else [0, 0, 0, 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color=['#2E86AB', '#A23B72', '#F18F01', '#4C956C'])
    ax.set_ylabel('Essay count')
    ax.set_title('Dataset split distribution')
    for rect, pct in zip(bars, percentages):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 2, f"{int(h)}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'dataset_split_distribution.png', dpi=300)
    plt.close(fig)

    # Figure 2: class distribution
    all_frames = pd.concat([df.assign(split=name) for name, df in splits.items()], ignore_index=True)
    class_summary = all_frames.groupby(['split', 'labels']).size().unstack(fill_value=0)
    class_summary = class_summary.reindex(['train', 'validation', 'test'] + (['unseen_test'] if 'unseen_test' in splits else []), fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    class_summary.plot(kind='bar', stacked=True, ax=ax, color=['#7EBDC2', '#E76F51'])
    ax.set_title('Human vs AI class balance across partitions')
    ax.set_xlabel('Split')
    ax.set_ylabel('Essay count')
    ax.legend(['Human (0)', 'AI (1)'])
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'class_distribution.png', dpi=300)
    plt.close(fig)

    # Figure 3: source composition
    source_df = all_frames.groupby('source').size().sort_values(ascending=False).head(15).reset_index()
    source_df.columns = ['source', 'count']
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(source_df['source'][::-1], source_df['count'][::-1], color='#5C7CFA')
    ax.set_title('Top sources by essay count')
    ax.set_xlabel('Essays')
    ax.set_ylabel('Source')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'source_distribution.png', dpi=300)
    plt.close(fig)

    # Figure 4: essay length distribution
    all_frames['word_count'] = all_frames['text'].fillna('').str.split().str.len()
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in [(0, '#7EBDC2'), (1, '#E76F51')]:
        subset = all_frames[all_frames['labels'] == label]['word_count']
        ax.hist(subset, bins=30, alpha=0.6, range=(0, 1500), label=f'Label {label}', color=color)
    ax.set_title('Essay word-count distribution by label')
    ax.set_xlabel('Word count')
    ax.set_ylabel('Count')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'essay_length_distribution.png', dpi=300)
    plt.close(fig)

    # Figure 5: pipeline flowchart
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.axis('off')
    boxes = [
        ('Original CSV', (0.05, 0.82)),
        ('Validation', (0.25, 0.82)),
        ('Deduplication', (0.47, 0.82)),
        ('Unseen source holdout', (0.68, 0.82)),
        ('Train / Validation / Test', (0.15, 0.52)),
        ('Evaluation datasets', (0.55, 0.52)),
    ]
    for label, (x, y) in boxes:
        ax.text(x, y, label, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.4', facecolor='#DDEBF7', edgecolor='black'), fontsize=10)
    ax.annotate('', xy=(0.2, 0.82), xytext=(0.15, 0.82), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.42, 0.82), xytext=(0.38, 0.82), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.63, 0.82), xytext=(0.56, 0.82), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.25, 0.52), xytext=(0.4, 0.82), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.65, 0.52), xytext=(0.72, 0.82), arrowprops=dict(arrowstyle='->'))
    ax.text(0.5, 0.2, f'Accepted rows: {accepted_rows}', ha='center', fontsize=12)
    fig.savefig(FIG_DIR / 'dataset_pipeline.png', dpi=300)
    plt.close(fig)

    # Figure 6: overview
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = [
        ('Original essays', int(original_rows)),
        ('Accepted', int(accepted_rows)),
        ('Removed duplicates', int(duplicate_rows)),
        ('Sources', int(source_count)),
        ('Human', int(human_total)),
        ('AI', int(ai_total)),
    ]
    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    ax.bar(labels, values, color=['#264653', '#2A9D8F', '#E76F51', '#F4A261', '#7EBDC2', '#A8DADC'])
    ax.set_title('Dataset overview')
    ax.set_ylabel('Count')
    for lx, vy in zip(labels, values):
        ax.text(lx, vy + 150, str(vy), ha='center', va='bottom')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'dataset_overview.png', dpi=300)
    plt.close(fig)

    print('Generated figures:', ', '.join(sorted(p.name for p in FIG_DIR.glob('*.png'))))


if __name__ == '__main__':
    build_figures()
