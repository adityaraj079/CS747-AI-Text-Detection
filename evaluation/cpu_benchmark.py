from __future__ import annotations

import os
import platform
import statistics
import threading
import time
from typing import Any, Callable, Dict, Iterable

import numpy as np
import psutil

from .model_adapters import ModelAdapter, require_probability_vector


def _measure_peak_rss_mb(operation: Callable[[], Any]) -> float:
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    stop_sampling = threading.Event()
    sampling_errors: list[psutil.Error] = []

    def sample_rss() -> None:
        nonlocal peak_rss
        try:
            while not stop_sampling.wait(0.005):
                peak_rss = max(peak_rss, process.memory_info().rss)
        except psutil.Error as exc:
            sampling_errors.append(exc)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    try:
        operation()
    finally:
        stop_sampling.set()
        sampler.join()
    if sampling_errors:
        raise RuntimeError("Unable to sample peak process memory during inference.") from sampling_errors[0]
    peak_rss = max(peak_rss, process.memory_info().rss)
    return float(peak_rss / (1024 * 1024))


def benchmark_cpu_inference(model: ModelAdapter, texts: Iterable[str], batch_size: int = 32, warmup_steps: int = 3, repeats: int = 5, include_cold_start: bool = False):
    texts = list(texts)
    if not texts:
        raise ValueError('Input texts cannot be empty.')
    if batch_size <= 0:
        raise ValueError('batch_size must be positive.')
    if repeats < 5:
        raise ValueError('At least five repetitions are required for a robust median benchmark.')

    for _ in range(warmup_steps):
        probs = model.predict_proba(texts[:batch_size])
        require_probability_vector(probs, len(texts[:batch_size]))

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        probs = model.predict_proba(texts)
        elapsed = time.perf_counter() - start
        require_probability_vector(probs, len(texts))
        timings.append(elapsed)

    memory_batch = texts[:batch_size]
    peak_ram_mb = _measure_peak_rss_mb(
        lambda: require_probability_vector(model.predict_proba(memory_batch), len(memory_batch))
    )
    median_elapsed = float(statistics.median(timings))
    total_essays = len(texts)
    result = {
        'batch_size': int(batch_size),
        'warmup_steps': int(warmup_steps),
        'repeats': int(repeats),
        'median_elapsed_seconds': median_elapsed,
        'milliseconds_per_essay': float((median_elapsed / total_essays) * 1000.0),
        'throughput_essays_per_second': float(total_essays / median_elapsed),
        'peak_ram_mb': peak_ram_mb,
        'cpu_model': platform.processor() or platform.machine(),
        'system_info': platform.platform(),
        'python_version': platform.python_version(),
        'thread_count': os.cpu_count() or 1,
    }
    if include_cold_start:
        cold_start_clock = time.perf_counter()
        model.predict_proba(texts)
        result['cold_start_seconds'] = time.perf_counter() - cold_start_clock
    return result
