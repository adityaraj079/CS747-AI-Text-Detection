# Results

This directory stores actual outputs from the shared evaluation framework. The dashboard reads only small result files from `metrics/`; it never loads essay CSVs or model weights on startup.

## Result categories

- `metrics/standardized/`: controlled results using identical frozen essay IDs and compatible benchmark settings.
- `metrics/historical/`: imported notebook results for progress tracking. These may use different splits, hardware, and thresholds and must not be presented as a fair head-to-head.
- `predictions/`: optional prediction exports. Do not commit essay text or sensitive student data.
- `benchmarks/`: detailed CPU/RAM benchmark records.
- `figures/`: saved evaluation figures.

## Metric record fields

Each standardized metric record should include:

- `model_id`
- `dataset_hash`
- `split_id` (the fingerprint for the evaluated partition)
- `dataset_type` (`standard_test` or verified `unseen_test`)
- `roc_auc`, `accuracy`, `precision`, `recall`, `f1`
- `confusion_matrix` in `[[TN, FP], [FN, TP]]` order
- `latency_ms_per_essay`, `throughput`, `model_size_mb`, `peak_ram_mb`
- `cpu_model`, `thread_count`, `batch_size`
- `record_count`, `threshold`, `timestamp`

Missing values should be omitted or null; never write placeholder scores. `results/metrics/historical/` is the import location for old notebook summaries. The dashboard labels these as historical and checks dataset/hardware compatibility before showing deltas or conclusions.
