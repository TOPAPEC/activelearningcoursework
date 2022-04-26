import numpy as np
from typing import Callable, Tuple, List
from matplotlib import pyplot as plt


def get_confidence_intervals(y: np.ndarray, y_pred: np.ndarray, metrics: List[Callable], n_samples: int = 1000,
                             confidence_level: float = 95) -> List[Tuple[float, Tuple[
    float, float]]]:
    assert confidence_level > 0.5, "Confidence level should be higher than 0.5."
    metrics_bootstrapped = np.zeros((len(metrics), n_samples))
    for i, metric in enumerate(metrics):
        for sampling_retry in range(n_samples):
            sampled_ind = np.random.choice(y.shape[0], int(y.shape[0]), replace=True)
            metrics_bootstrapped[i][sampling_retry] = metric(y_pred[sampled_ind], y[sampled_ind])
    return [(float(np.mean(metric_arr)),
             (np.percentile(metric_arr, 100 - confidence_level), np.percentile(metric_arr, confidence_level))) for
            metric_arr in metrics_bootstrapped]


def get_bootstrapped_metrics(y: np.ndarray, y_pred: np.ndarray, metrics: List[Callable], n_samples: int = 1000) -> \
        np.ndarray:
    metrics_bootstrapped = np.zeros((len(metrics), n_samples))
    for i, metric in enumerate(metrics):
        for sampling_retry in range(n_samples):
            sampled_ind = np.random.choice(y.shape[0], int(y.shape[0]), replace=True)
            metrics_bootstrapped[i][sampling_retry] = metric(y_pred[sampled_ind], y[sampled_ind])
    return metrics_bootstrapped


def confidence_intervals_to_plot(interval: List[Tuple[float, Tuple[float, float]]], label="label") -> None:
    length = len(interval)
    metrics = np.zeros(length)
    ci_bottoms = np.zeros(length)
    ci_tops = np.zeros(length)
    for it, metric_values in enumerate(interval):
        metrics[it], (ci_bottoms[it], ci_tops[it]) = metric_values
    plt.plot(np.arange(0, length), metrics)
    plt.fill_between(np.arange(length), ci_bottoms, ci_tops, color="blue", alpha=0.1, label=label)
