# Evaluation package

This package defines the shared contract for dataset validation, metrics, model registration, and CPU benchmarking.

Key modules:

- `dataset_utils.py`: dataset validation, text hashing, deduplication, and partitioning utilities
- `evaluator.py`: general model evaluation helper
- `cpu_benchmark.py`: CPU-only inference benchmark
- `metrics.py`: metric computations and serialization
- `model_adapters.py`: adapter interface and registration contract
