# Results directory

This directory is reserved for model metrics, predictions, and benchmark outputs. Each result should include a dataset hash, model id, split id, metric values, and runtime metadata.

Recommended fields:

- `model_id`
- `dataset_hash`
- `split_id`
- `dataset_type`
- `roc_auc`
- `accuracy`
- `precision`
- `recall`
- `f1`
- `latency_ms_per_essay`
- `throughput`
- `model_size_mb`
- `cpu_model`
- `thread_count`
- `timestamp`
