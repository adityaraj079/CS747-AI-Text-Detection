"""Score the four frozen models on matched English HC3 answers.

Pass --hc3 all.jsonl from the pinned revision below and --corpus CS747data.csv.
Only answer texts are passed to the models. Retain hashes and metrics for the UI.
"""
import argparse
import hashlib
import json
import multiprocessing
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app_components.model_registry import load_model_adapter
from evaluation.evaluator import evaluate_model
from evaluation.external_holdout import TOKEN, WINDOW, screen_pairs

REVISION = "4d0ff18143b5a7e1b1e79beb540c04549d1e59d3"
SHA256 = "ee231f82283d754050a4dc58ecf9331afaf5daea7283c6da0b096ed0fb2774cf"
SEED = 747
PAIR_COUNT = 1000


def eligible_pairs(raw):
    pairs = []
    seen_questions = set()
    for index, line in enumerate(raw.splitlines()):
        record = json.loads(line)
        question = " ".join(TOKEN.findall(record["question"].casefold()))
        if not question or question in seen_questions:
            continue
        pair = []
        for label, key in enumerate(("human_answers", "chatgpt_answers")):
            for offset, answer in enumerate(record[key]):
                text = answer.strip()
                if len(TOKEN.findall(text)) >= WINDOW:
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    pair.append((index, label, f"{index}/{key}/{offset}", digest, text))
                    break
        if len(pair) == 2:
            seen_questions.add(question)
            pairs.append(pair)
    return pairs


def score(model_id, texts, labels):
    print(f"Evaluating {model_id} on {len(texts)} answers", flush=True)
    return evaluate_model(load_model_adapter(model_id), texts, labels, "hc3_answers", model_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hc3", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    args = parser.parse_args()
    raw = args.hc3.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("Expected the pinned English all.jsonl file.")
    candidates = random.Random(SEED).sample(eligible_pairs(raw), PAIR_COUNT + 100)
    rows, excluded, corpus_rows = screen_pairs([r for pair in candidates for r in pair], args.corpus)
    rows = rows[:2 * PAIR_COUNT]
    if len(rows) != 2 * PAIR_COUNT:
        raise ValueError("Insufficient eligible pairs after overlap screening.")
    manifest = [{"path": path, "label": label, "sha256": digest}
                for _, label, path, digest, _ in rows]
    split = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    texts = [r[4] for r in rows]
    labels = [r[1] for r in rows]
    results = []
    # Keep LightGBM and PyTorch's native CPU thread pools in separate processes.
    models = ("aakash_7th_place", "aditya_3rd_place", "yash_5th_place", "daniel_bert")
    with multiprocessing.get_context("spawn").Pool(1, maxtasksperchild=1) as workers:
        for model_id in models:
            metrics = workers.apply(score, (model_id, texts, labels))
            results.append(dict(metrics, dataset_type="hc3_answers", dataset_hash=SHA256,
                                split_id=split, record_count=len(rows)))
    payload = dict(source=f"https://huggingface.co/datasets/Hello-SimpleAI/HC3/tree/{REVISION}",
                   source_file="all.jsonl", file_sha256=SHA256, seed=SEED,
                   candidate_pairs=len(candidates), excluded_pairs=excluded, corpus_rows=corpus_rows,
                   samples=manifest, results=results)
    path = ROOT / "results/metrics/standardized/hc3_answers.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    main()
