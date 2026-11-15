# CS747 model training and integration prompt

Copy this prompt into a coding agent when training or packaging a team model. Use it for:

- `models/kaggle_first_place/`
- `models/yash_5th_place/`
- `models/aakash_7th_place/`
- `models/aditya_3rd_place/`
- `models/daniel_bert/`
- `models/custom_model/`

```text
You are training and packaging a team model for the CS747 AI Text Detection dashboard. Inspect the assigned model directory, notebook, requirements, and existing artifacts before editing.

Dataset and split contract:
- Fit the tokenizer, vectorizer, feature pipeline, and model only on data/processed/train.csv (86,227 records).
- Use data/processed/validation.csv (18,477 records) only for model selection, early stopping, calibration, or ensemble selection.
- Do not create a new random split or use an 80/10/10 split. The project already froze its partitions: the unseen-source holdout was reserved first, then the remaining standard pool was split 70/15/15.
- Do not fit, tune, calibrate, or select features on data/processed/test.csv or data/processed/unseen_test.csv. Use these only for final evaluation through the shared evaluator.
- Keep processed CSVs local. They are ignored by Git and must never be staged or pushed.
- Read data/manifests/split_manifest.json and record the dataset hash and train/validation split fingerprints with the model metadata.

Training and provenance:
- Training a missing model artifact is expected for this task. Preserve the original notebook; if it uses a different split or data path, adapt only the training data-loading/split code or create a model-specific reproducible training script.
- Do not claim an adaptation is the original competition model. Record which settings, pretrained weights, and ensemble rules are verified versus adapted or unknown.
- Do not invent leaderboard scores, provenance, licenses, predictions, or performance results.
- Verify licenses and redistribution rights before committing tokenizer/model weights. If uncertain, stop and ask the model owner.

Scientific and safety constraints:
- Preserve the original notebook and intended inference pipeline.
- Do not modify the frozen evaluation datasets, manifests, or another member's code.
- Use only trusted, repository-owned model artifacts. Never load user-uploaded or arbitrary serialized files.
- Use relative paths; support CPU inference where the model permits it.
- Keep model-specific dependencies in this model directory's requirements.txt.
- Do not determine a final test threshold or fit calibration on the final test set. Use the shared validation-selected threshold and calibration policy.

Train or adapt the model using its documented pipeline, then package it using:

member_model/
├── README.md
├── adapter.py
├── config.json
├── requirements.txt
├── original_notebook.ipynb
├── artifacts/
│   └── README.md
└── tests/
    └── test_adapter.py

Tasks:
1. Inspect the notebook and existing files; identify preprocessing, tokenizers, vectorizers, feature extraction, classifiers, neural weights, calibration, and ensemble logic.
2. Train only on the frozen training partition and use validation only for selection. Verify the fitted artifact matches the documented inference path.
3. Save only final inference assets in models/<model_id>/artifacts/: tokenizer, vectorizer/feature pipeline, final classifier weights, and any required ensemble configuration. Do not save raw data, temporary checkpoints, sparse training matrices, caches, or per-essay prediction dumps.
4. Record training configuration, random seed, dataset hash, train/validation split fingerprints, dependency versions, ensemble weights, and verified/unverified provenance in configuration.json.
5. Extract the complete inference-only pipeline into adapter.py. Implement a class derived from evaluation.model_adapters.ModelAdapter.
6. Implement predict_proba(texts: list[str]) returning a finite float NumPy array with shape (n_samples,), values in [0, 1], and orientation P(labels == 1), where 1 means AI-generated.
7. Raise evaluation.model_adapters.ModelNotAvailableError when implementation or fitted artifacts are unavailable. Never return random, constant, or fabricated probabilities to imitate a working model.
8. Add tests for empty input, valid output shape/range/orientation, and clear missing-artifact errors. Do not require large model files for basic package tests.
9. Record verified metadata in config.json: model ID, member, competition track, source reference, architecture, dependencies, training-data requirements, artifact paths, CPU compatibility, and integration status. Use null or explicit “unknown/not verified” values when facts are unavailable.
10. Document provenance, licenses/redistribution limits, training data, known limitations, artifact locations, and how the adapter is used with the shared evaluator.
11. Register the trusted adapter explicitly in app_components/model_registry.py. Do not add unrestricted model discovery, public artifact upload, or arbitrary pickle loading.
12. Run focused tests and evaluate once on standard_test and unseen_test using scripts/evaluate_registered_model.py. Do not use final evaluation results to tune the model.
13. Before committing, inspect git status and stage only the assigned model package, required dependencies/registry metadata, and genuine evaluation summaries. Verify no processed CSVs or temporary files are staged.
14. Review the staged diff, commit the model package, and push the branch or open a pull request. Never push the dataset or unrelated work.
```

Final model artifacts belong directly under `models/<model_id>/artifacts/`. The repository `.gitignore` permits common artifact formats there while continuing to ignore them elsewhere. Check each file size before pushing: use Git LFS only if it is configured and available; do not push individual files over GitHub's 100 MB limit or upload restricted weights.

## Dashboard registration

The dashboard allowlists model IDs in `app_components/model_registry.py`. After implementing and testing an adapter, add an explicit loader for that model ID and update its metadata in `app_components/model_catalog.py` or its verified `config.json`. A model is not considered evaluated until a result record is saved under `results/metrics/`.
