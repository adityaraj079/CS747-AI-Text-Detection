#!/usr/bin/env python3
"""Score the custom ensemble (and each of its members) on the three external benchmarks.

Run from the repo root:
  python -m models.custom_model.evaluate_external --corpus CS747data.csv --out external.json \
      [--ghostbuster ghostbuster-data/essay] [--hc3 all.jsonl] [--wiki GPT-wiki-intro.csv.zip]

The samples come from the shared scripts' own selection code (same seed, pinned files, and overlap screening),
so they match the essays in results/metrics/standardized/{ghostbuster_essay,hc3_answers,wiki_intro_50}.json.
The shared scripts and result files are not modified. Results go to --out only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from evaluation.external_holdout import screen_pairs
from evaluation.metrics import compute_metrics
from models.custom_model.adapter import CustomEnsembleAdapter
from scripts import evaluate_ghostbuster as gb, evaluate_hc3 as hc, evaluate_wiki_intro as wk


def hc3_rows(path: Path, corpus: Path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != hc.SHA256:
        raise ValueError("Expected the pinned English HC3 all.jsonl file.")
    candidates = random.Random(hc.SEED).sample(hc.eligible_pairs(raw), hc.PAIR_COUNT + 100)
    rows, _, _ = screen_pairs([r for pair in candidates for r in pair], corpus)
    rows = rows[:2 * hc.PAIR_COUNT]
    if len(rows) != 2 * hc.PAIR_COUNT:
        raise ValueError("Insufficient eligible HC3 pairs after overlap screening.")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ghostbuster", type=Path)
    parser.add_argument("--hc3", type=Path)
    parser.add_argument("--wiki", type=Path)
    args = parser.parse_args()

    datasets = {}
    if args.ghostbuster:
        datasets["ghostbuster_essay"] = gb.collect(args.ghostbuster, args.corpus)[0]
    if args.hc3:
        datasets["hc3_answers"] = hc3_rows(args.hc3, args.corpus)
    if args.wiki:
        datasets["wiki_intro_50"] = wk.collect(args.wiki, args.corpus)[0]

    ensemble = CustomEnsembleAdapter()
    if not ensemble.model_ready:
        raise RuntimeError(ensemble.unavailable_reason)
    names = [n for n, w in zip(ensemble.combiner["members"], ensemble.combiner["weights"]) if w != 0]
    results = {}
    for dataset, rows in datasets.items():
        texts, labels = [r[4] for r in rows], [r[1] for r in rows]
        print(f"{dataset}: {len(texts)} texts", flush=True)
        member_p = []
        for name, member in zip(names, ensemble.members):
            p = np.concatenate([member.predict_proba(texts[i:i + 4096]) for i in range(0, len(texts), 4096)])
            member_p.append(p)
            print(f"  {name} done", flush=True)
        p = np.clip(np.column_stack(member_p), 1e-6, 1 - 1e-6)  # same formula as CustomEnsembleAdapter
        p_ens = 1 / (1 + np.exp(-(np.log(p / (1 - p)) @ ensemble.weights + ensemble.offset)))
        results[dataset] = {"record_count": len(texts),
                            "custom_model": compute_metrics(labels, p_ens),
                            **{f"member_{n}": compute_metrics(labels, member_p[i]) for i, n in enumerate(names)}}
        print(f"  custom_model AUC {results[dataset]['custom_model']['roc_auc']:.4f}", flush=True)
    args.out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print("RESULTS " + json.dumps({d: {k: round(v["roc_auc"], 4) for k, v in r.items() if isinstance(v, dict)}
                                   for d, r in results.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
