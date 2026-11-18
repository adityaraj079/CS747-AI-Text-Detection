# Daniel BERT model

This directory is reserved for the BERT-family classifier assigned to Daniel Stiller for the CS747 AI text detection task.

## Current status

The adapter loads the existing public checkpoint from [Hugging Face](https://huggingface.co/grumpydan99/BERT), pinned to a specific immutable revision in `config.json`. The repository does not store the large weights. Hugging Face reports the model-card license as unknown; the model owner should verify reuse/redistribution rights.

## Expected package layout

```text
models/daniel_bert/
├── README.md
├── adapter.py
├── config.json
├── requirements.txt
├── original_notebook.ipynb
├── artifacts/
│   ├── README.md
└── tests/
    └── test_adapter.py
```

## Training provenance and comparability

- The original notebook uses a stratified 80/10/10 split with `random_state=42` and records 132,068 train, 16,509 validation, and 16,509 test records.
- That split differs from the repository's frozen partitions. Its essays may overlap the repository's standard or unseen evaluation sets, so evaluations here may be contaminated and are not directly comparable with frozen-split models.
- The notebook scores class index 1 as positive; the project label convention defines 1 as AI-generated. Dataset hash, split fingerprints, and exact overlap with frozen partitions are unknown.
- Do not infer the original model's license from the BERT architecture. The Hub model card says `unknown`.

## Checkpoint and inference

The checkpoint and tokenizer are downloaded from the pinned Hugging Face revision at inference time. The repo stores only package metadata and adapter code, avoiding GitHub's large-file limit. Public access does not require credentials; set `HF_TOKEN` in the environment only if the Hub repo becomes private.

Install the model-specific runtime dependencies with:

```bash
pip install -r models/daniel_bert/requirements.txt
```

## Adapter contract

The adapter implements `predict_proba(texts: list[str])` and returns a finite NumPy probability vector with shape `(n_samples,)`, values in `[0, 1]`, and orientation `P(labels == 1)` where `1` means AI-generated.

The adapter supports a local checkpoint override and the configured Hub model. It raises `ModelNotAvailableError` if loading fails instead of fabricating probabilities. The original notebook treats output class index 1 as the positive class.

## Evaluate and save results

The recreated original split is stored locally under `data/processed/daniel_bert_original_80_10_10/`. Run the Daniel-only evaluator from the repository root:

```bash
python models/daniel_bert/evaluate_original_split.py
```

It reports processed essay count, elapsed time, and an estimated time remaining every ten seconds. The ETA is based on average inference throughput and becomes more reliable as the run progresses. The evaluation uses the shared metric functions and saves one historical result after inference and the CPU benchmark complete:
`results/metrics/historical/daniel_bert_original_80_10_10_test.json`.

The evaluation does not use or overwrite the frozen project test partitions. Because the model was trained using the separate 80/10/10 split, its result is historical and should not be treated as a leakage-free comparison with models trained on the repository's frozen split.
