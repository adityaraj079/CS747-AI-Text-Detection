from __future__ import annotations

import numpy as np

from evaluation.cpu_benchmark import benchmark_cpu_inference


class ConstantAdapter:
    def predict_proba(self, texts):
        return np.full(len(texts), 0.5)


def test_cpu_benchmark_reports_peak_rss_without_changing_prediction_metrics():
    result = benchmark_cpu_inference(
        ConstantAdapter(),
        ["essay one", "essay two"],
        batch_size=2,
        warmup_steps=1,
        repeats=5,
    )

    assert result["peak_ram_mb"] > 0
    assert result["milliseconds_per_essay"] >= 0
    assert result["throughput_essays_per_second"] > 0
