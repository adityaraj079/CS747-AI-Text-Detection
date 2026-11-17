"""Score frozen detectors on a seeded sample of the Ghostbuster essay dataset.

Pass --essays pointing at ghostbuster-data/essay from REVISION and --corpus
pointing at the full original CS747data.csv (including excluded/duplicate rows).
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app_components.model_registry import load_model_adapter
from evaluation.evaluator import evaluate_model
from evaluation.external_holdout import screen_pairs

REVISION = "86ebd72590556a81622986fab736ab9227a948af"
MODELS = ("aakash_7th_place", "aditya_3rd_place", "daniel_bert", "yash_5th_place")


def collect(essays, corpus):
    ids = sorted(random.Random(747).sample(range(1, 1001), 100))
    samples = []
    for number in ids:
        for label, source in enumerate(("human", "gpt")):
            path = f"{source}/{number}.txt"
            raw = (essays / path).read_bytes()
            text = raw.decode("utf-8-sig").strip()
            samples.append((number, label, path, hashlib.sha256(raw).hexdigest(), text))

    return screen_pairs(samples, corpus)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--essays", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    args = parser.parse_args()
    rows, excluded, count = collect(args.essays, args.corpus)
    if not rows:
        raise ValueError("No eligible pairs remain after overlap screening.")
    manifest = [{"path": r[2], "sha256": r[3], "label": r[1]} for r in rows]
    split = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    texts = [r[4] for r in rows]
    labels = [r[1] for r in rows]
    results = []
    for model_id in MODELS:
        print(f"Evaluating {model_id} on {len(rows)} essays", flush=True)
        metrics = evaluate_model(load_model_adapter(model_id), texts, labels,
                                 "ghostbuster_essay", model_id)
        results.append(dict(metrics, dataset_type="ghostbuster_essay", dataset_hash=split,
                            split_id=split, record_count=len(rows)))
    payload = dict(source="https://github.com/vivek3141/ghostbuster-data/tree/" + REVISION,
                   seed=747, candidate_pairs=100, excluded_pairs=excluded,
                   corpus_rows=count, samples=manifest, results=results)
    path = ROOT / "results/metrics/standardized/ghostbuster_essay.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    main()
