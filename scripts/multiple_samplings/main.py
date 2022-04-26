import functools
import logging
import pickle

import numpy as np
from matplotlib import pyplot as plt
from modAL import ActiveLearner
from modAL.density import information_density
from modAL.uncertainty import classifier_entropy, classifier_uncertainty, classifier_margin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from lib.models.Linear import LogReg
from lib.models.Wraps import TorchClassifierWrap
from lib.visualize import get_confidence_intervals, confidence_intervals_to_plot

logging.basicConfig(level=logging.INFO)



def multiple_sampler(X_pool, samplers, n_instances=1, **kwargs):
    ranks = np.zeros((len(samplers), X_pool.shape[0]), dtype=np.int64)
    information_density(X_pool, metric="cosine")
    for i, spr in enumerate(samplers):
        query_idx = np.flip(np.argsort(spr(X=X_pool)))
        ranks[i] = query_idx
    print(ranks.shape)
    ranks = np.log(ranks)
    merged_rank = np.sum(ranks, axis=1)
    print(merged_rank.shape)
    return np.argsort(merged_rank)[:n_instances]


def main():
    X = np.load("../../resources/cachedData/redditSelfPostVectorized/vectorized.npy", allow_pickle=True)
    y = np.load("../../resources/cachedData/redditSelfPostVectorized/vectorized_labels.npy", allow_pickle=True)
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.999, random_state=42)
    X_pool, X_val, y_pool, y_val = train_test_split(X_test, y_test, test_size=0.3, random_state=42)
    classes = np.unique(y)
    print(X_train.shape)
    metrics = []
    model = TorchClassifierWrap(LogReg(X_train.shape[1], len(classes)), 100, X_train.shape[1], 1000, random_state=42)
    model.fit(X_train, y_train)
    samplers = [
        functools.partial(classifier_margin, classifier=model),
        functools.partial(classifier_uncertainty, classifier=model),
        functools.partial(classifier_entropy, classifier=model),
        functools.partial(information_density, metric="cosine"),
    ]
    y_pred = model.predict(X_val)
    metrics.append(get_confidence_intervals(y_pred, y_val, [accuracy_score])[0])
    for it in tqdm(range(10)):
        query_idx, _ = multiple_sampler(X_pool, samplers, n_instances=1000)
        X_train = np.append(X_train, X_pool[query_idx], axis=0)
        y_train = np.append(y_train, y_pool[query_idx], axis=0)
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)
        model = TorchClassifierWrap(LogReg(X_train.shape[1], len(classes)), 100, X_train.shape[1], 1000,
                                    random_state=42)
        model.fit(X_train, y_train)
        samplers = [
            functools.partial(classifier_margin, classifier=model),
            functools.partial(classifier_uncertainty, classifier=model),
            functools.partial(classifier_entropy, classifier=model),
            functools.partial(information_density, metric="cosine"),
        ]
        y_pred = model.predict(X_val)
        metrics.append(get_confidence_intervals(y_pred, y_val, [accuracy_score])[0])
    with open("../../resources/cachedData/miscellaneous/bootstraptesting_saved_metrics/multiplesamp1_10it_1000batch.pkl",
              "wb") as file:
        pickle.dump(metrics, file)

    confidence_intervals_to_plot(metrics, label="val")
    plt.show()

if __name__ == "__main__":
    main()