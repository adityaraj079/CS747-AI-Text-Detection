"""Evaluate 50-word Wikipedia and GPT continuations from 50,000 shared topics.

Use the pinned GPT-wiki-intro.csv.zip and the full project CS747data.csv.
Run with --bert-device mps on Apple Silicon or use the default CPU.
"""
import argparse
import csv
import hashlib
import io
import json
import multiprocessing
import random
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app_components.model_registry import load_model_adapter
from evaluation.external_holdout import screen_pairs
from evaluation.metrics import compute_metrics
from evaluation.model_adapters import require_probability_vector

REVISION = "017258d41991489752363bfa4cbcc12725786967"
ARCHIVE_SHA = "ad1fe3a40c5d52cf80f6545ed8f14ae6d2c0d3dda172e9744aeb101c45dc1a1e"
SEED = 747
PAIR_COUNT = 50_000
WORDS = 50
MODELS = ("aditya_3rd_place", "aakash_7th_place", "yash_5th_place", "daniel_bert")


def collect(archive, corpus):
    with archive.open("rb") as fh:
        digest = hashlib.sha256()
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != ARCHIVE_SHA:
        raise ValueError("Expected the pinned GPT-wiki-intro archive.")
    pairs = []
    seen = set()
    with zipfile.ZipFile(archive) as z, z.open("GPT-wiki-intro.csv") as data:
        for row in csv.DictReader(io.TextIOWrapper(data, encoding="utf-8-sig")):
            article = row["id"]
            if article in seen:
                continue
            pair = []
            for label, key in enumerate(("wiki_intro", "generated_text")):
                words = row[key].split()
                if len(words) < WORDS:
                    break
                text = " ".join(words[-WORDS:])
                sha = hashlib.sha256(text.encode()).hexdigest()
                pair.append((article, label, key, sha, text))
            if len(pair) == 2:
                pairs.append(pair)
                seen.add(article)
    candidates = random.Random(SEED).sample(pairs, PAIR_COUNT + 5000)
    del pairs
    rows, excluded, count = screen_pairs([r for pair in candidates for r in pair], corpus)
    rows = rows[:2 * PAIR_COUNT]
    if len(rows) != 2 * PAIR_COUNT:
        raise ValueError("Too few complete screened pairs.")
    return rows, len(excluded), count


def score(model_id, texts, labels, device):
    if model_id == "daniel_bert" and device != "cpu":
        from models.daniel_bert.adapter import DanielBertAdapter
        model = DanielBertAdapter(device=device, batch_size=32)
    else:
        model = load_model_adapter(model_id)
    predictions = []
    for start in range(0, len(texts), 512):
        batch = texts[start:start + 512]
        predictions.extend(require_probability_vector(model.predict_proba(batch), len(batch)))
        if start % 8192 == 0:
            print(f"{model_id}: {start + len(batch):,}/{len(texts):,}", flush=True)
    return compute_metrics(labels, predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--bert-device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    rows, excluded, count = collect(args.archive, args.corpus)
    split = hashlib.sha256()
    for article, label, _, sha, _ in rows:
        split.update(f"{article},{label},{sha}\n".encode())
    # Sort by length to reduce padding, preserving text/label alignment.
    rows.sort(key=lambda r: len(r[4]))
    texts = [r[4] for r in rows]
    labels = [r[1] for r in rows]
    results = []
    with multiprocessing.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
        for model_id in MODELS:
            metrics = pool.apply(score, (model_id, texts, labels, args.bert_device))
            results.append(dict(metrics, model_id=model_id, dataset_type="wiki_intro_50",
                                split_id=split.hexdigest(), dataset_hash=ARCHIVE_SHA,
                                record_count=len(texts)))
    payload = dict(source=f"https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro/tree/{REVISION}",
                   file_sha256=ARCHIVE_SHA, seed=SEED, words_per_passage=WORDS, pairs=PAIR_COUNT,
                   candidate_pairs=PAIR_COUNT + 5000, excluded_candidate_pairs=excluded,
                   corpus_rows=count, bert_device=args.bert_device, results=results)
    path = ROOT / "results/metrics/standardized/wiki_intro_50.json"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as fh:
        json.dump(payload, fh, indent=2, allow_nan=False)
        fh.write("\n")
    Path(fh.name).replace(path)


if __name__ == "__main__":
    main()
