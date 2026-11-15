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
    quality = json.loads((MANIFESTS / 'data_quality_report.json').read_text())
    all_frames = pd.concat([df.assign(split=name) for name, df in splits.items()], ignore_index=True)
    accepted_rows = len(all_frames)
    original_rows = int(quality.get('original_row_count', 0))
    excluded_rows = original_rows - accepted_rows
    source_count = int(all_frames['source'].nunique())
    human_total = int(all_frames['labels'].eq(0).sum())
    ai_total = int(all_frames['labels'].eq(1).sum())
    standard_pool_rows = len(splits['train']) + len(splits['validation']) + len(splits['test'])

    # Figure 1: split distribution
    counts = [len(splits.get('train', [])), len(splits.get('validation', [])), len(splits.get('test', [])), len(splits.get('unseen_test', [])) if 'unseen_test' in splits else 0]
    labels = ['Train', 'Validation', 'Standard Test', 'Unseen Test']
    total = sum(counts)
    percentages = [c / total * 100 for c in counts] if total else [0, 0, 0, 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color=['#2E86AB', '#A23B72', '#F18F01', '#4C956C'])
    ax.set_ylabel('Essay count')
    ax.set_title('Dataset splits (percentages of all accepted records)')
    for rect, pct in zip(bars, percentages):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 2, f"{int(h):,}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'dataset_split_distribution.png', dpi=300)
    plt.close(fig)

    # Figure 2: class distribution
    class_summary = all_frames.groupby(['split', 'labels']).size().unstack(fill_value=0)
    class_summary = class_summary.reindex(['train', 'validation', 'test'] + (['unseen_test'] if 'unseen_test' in splits else []), fill_value=0)
    class_summary = class_summary.reindex(columns=[0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    x_values = list(range(len(class_summary)))
    totals = class_summary.sum(axis=1)
    bottoms = [0.0] * len(class_summary)
    colors = {0: '#287D83', 1: '#D46A4C'}
    for label in (0, 1):
        values = class_summary[label].tolist()
        percentages = [
            (value / int(totals.iloc[index]) * 100) if totals.iloc[index] else 0
            for index, value in enumerate(values)
        ]
        bars = ax.bar(x_values, percentages, bottom=bottoms, label=f'Label {label}', color=colors[label])
        for index, (bar, value, percentage) in enumerate(zip(bars, values, percentages)):
            if value:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottoms[index] + percentage / 2,
                    f'{value:,}\n({percentage:.1f}%)',
                    ha='center',
                    va='center',
                    color='white',
                    fontsize=8,
                )
        bottoms = [bottoms[index] + percentages[index] for index in range(len(values))]
    for index, total_count in enumerate(totals):
        ax.text(x_values[index], 103, f'n={total_count:,}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x_values, class_summary.index)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_title('Class proportions across partitions')
    ax.set_xlabel('Split')
    ax.set_ylabel('Share of essays (%)')
    ax.legend(['Human-written (0)', 'AI-generated (1)'])
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'class_distribution.png', dpi=300)
    plt.close(fig)

    # Figure 3: source composition
    source_totals = all_frames.groupby('source').size().sort_values(ascending=False)
    top_sources = set(source_totals.head(15).index)
    source_plot = all_frames[['source', 'labels']].copy()
    source_plot['source_group'] = source_plot['source'].where(source_plot['source'].isin(top_sources), 'Other')
    source_summary = source_plot.groupby(['source_group', 'labels']).size().unstack(fill_value=0)
    source_summary = source_summary.reindex(columns=[0, 1], fill_value=0)
    source_summary['total'] = source_summary.sum(axis=1)
    source_summary = source_summary.sort_values('total')
    fig, ax = plt.subplots(figsize=(10, 6))
    source_labels = source_summary.index.tolist()
    positions = list(range(len(source_labels)))
    left = [0] * len(source_labels)
    for label, color, legend in ((0, '#287D83', 'Human (0)'), (1, '#D46A4C', 'AI (1)')):
        values = source_summary[label].tolist()
        bars = ax.barh(positions, values, left=left, color=color, label=legend)
        for index, (bar, value) in enumerate(zip(bars, values)):
            if value >= max(1, int(source_summary['total'].max() * 0.035)):
                ax.text(
                    left[index] + value / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f'{value:,}',
                    ha='center',
                    va='center',
                    color='white',
                    fontsize=8,
                )
        left = [left[index] + values[index] for index in range(len(values))]
    ax.set_yticks(positions, source_labels)
    ax.set_title('Source composition by verified label')
    ax.set_xlabel('Essays')
    ax.set_ylabel('Source')
    ax.legend()
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
        (f'Original CSV\n{original_rows:,} rows', (0.14, 0.78)),
        (f'Validated + deduplicated\n{accepted_rows:,} accepted', (0.50, 0.78)),
        (f'Unseen source holdout\n{len(splits.get("unseen_test", [])):,} rows', (0.30, 0.40)),
        (
            f'Standard pool: {standard_pool_rows:,}\n'
            f'Train {len(splits["train"]):,} | Val {len(splits["validation"]):,}\n'
            f'Test {len(splits["test"]):,}',
            (0.72, 0.40),
        ),
    ]
    for label, (x, y) in boxes:
        ax.text(x, y, label, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.4', facecolor='#DDEBF7', edgecolor='black'), fontsize=10)
    ax.annotate('', xy=(0.38, 0.78), xytext=(0.25, 0.78), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.30, 0.49), xytext=(0.45, 0.69), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.72, 0.49), xytext=(0.55, 0.69), arrowprops=dict(arrowstyle='->'))
    ax.text(0.5, 0.12, 'Evaluation: standard test + unseen test', ha='center', fontsize=11)
    fig.savefig(FIG_DIR / 'dataset_pipeline.png', dpi=300)
    plt.close(fig)

    # Figure 6: overview
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = [
        ('Original essays', int(original_rows)),
        ('Accepted', int(accepted_rows)),
        ('Excluded / quarantined', int(excluded_rows)),
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
