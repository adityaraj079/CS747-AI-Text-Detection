Final inference artifacts for the custom ensemble. They were trained on Hopper with the scripts in `..`.

- `combiner.json`: the ensemble rule, member list, weights, and validation-selected threshold. It also records
  source-held-out validation AUCs and the one-time test and unseen_test report, written by `build_ensemble.py`.
- `seventh/`: the 7th-place pipeline retrained on the frozen train split by `train_seventh.py`. It contains
  `tokenizer.json`, `model.joblib` (plain arrays only), and `configuration.json`. It loads with
  `AakashSeventhPlaceAdapter`.
- `bert/`: BERT retrained on the frozen train split by `train_bert.py`. This folder is **not in Git** because
  `model.safetensors` is 418 MB. The adapter downloads the folder from the Hugging Face repo and revision pinned
  in `../config.json`, and its training configuration is copied into that file.

Aditya's and Yash's members are loaded from their own packages (`models/aditya_3rd_place`,
`models/yash_5th_place`) and are not copied here.

No raw data, per-essay scores, checkpoints, or training matrices are stored here. These are trusted,
repository-generated files. Don't replace them with files from an untrusted source.
