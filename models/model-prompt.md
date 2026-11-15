# Model Packaging Prompt for Team Members

Copy and paste the following prompt into your coding agent.

```text
You are working inside the CS747 AI Text Detection project. Your job is to inspect an existing completed or near-complete model notebook and package it for compatibility with our shared evaluation framework.

Important constraints:
- Do NOT retrain the model unless I explicitly ask for it.
- Preserve the original inference pipeline and all fitted artifacts used at prediction time.
- Do not modify or delete the original notebook unless it is required for a clear packaging step.
- Keep the model compatible with CPU-only inference.
- Avoid absolute paths or environment-specific assumptions.
- Do not load arbitrary pickle/joblib files from untrusted sources.
- The package must expose a `ModelAdapter` interface consistent with the shared evaluator.

Task:
1. Inspect the existing notebook or model files in the current working directory.
2. Identify all preprocessing steps, tokenizers, vectorizers, feature transformations, and encoders.
3. Identify the trained classifier or neural weights used at inference time.
4. Confirm the original inference pipeline and preserve it exactly as much as possible.
5. Extract the reusable inference logic into a file named `adapter.py`.
6. Implement a class or loader that can initialize the model from the repository artifacts saved alongside the notebook.
7. Add a `predict_proba(self, texts: list[str]) -> np.ndarray` method that returns a 1D NumPy array of probabilities for `P(AI-generated)` for each essay.
8. The returned array must:
   - have length equal to the number of input texts
   - have shape `(n_samples,)`
   - contain float probabilities in `[0, 1]`
   - have no NaN or infinite values
   - use label orientation consistent with `1 = AI-generated`
9. Create `config.json` with metadata such as model id, team member name, architecture, training notes, version, and dependencies.
10. Add `requirements.txt` listing any model-specific dependencies and versions.
11. Create a `tests/test_adapter.py` smoke test with a few sample strings to ensure the adapter runs on CPU and returns valid probability arrays.
12. Add `artifacts/README.md` explaining what fitted assets are stored and how they are loaded.
13. Document provenance and references to the original Kaggle solution or notebook.
14. Document training data and known limitations.
15. Do not modify evaluation datasets or general project data files.
16. Do not hardcode absolute file paths; use relative paths inside the package.
17. Ensure inference is safe on CPU only, with no GPU assumptions.
18. Provide a README that explains how the packaged model integrates into the shared evaluator.
19. Keep the original notebook intact, and do not rerun training unless I ask for it.

Required package structure:

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

Required model interface:

```python
class ModelAdapter:
    def predict_proba(self, texts: list[str]):
        """
        Return a 1D NumPy array of probabilities representing P(AI-generated) for each essay.
        """
        raise NotImplementedError
```

After packaging, explain how to hand the model to the shared evaluator and how it should be registered with the project evaluation contract.

Do not determine the final threshold in the model itself. The shared evaluation framework will select or validate the threshold.

Any probability calibration must be learned using only training/validation data, never on the final test dataset.
```
