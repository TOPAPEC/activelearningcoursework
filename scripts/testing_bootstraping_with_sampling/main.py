import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from modAL import ActiveLearner
from tqdm import tqdm
from modAL.uncertainty import uncertainty_sampling, margin_sampling
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split

from lib.models.Linear import LogReg
from lib.models.Wraps import TorchClassifierWrap
from lib.visualize import get_confidence_intervals, confidence_intervals_to_plot
plt.rcParams["figure.figsize"] = (20, 20)
mpl.style.use('ggplot')


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
    train_metrics = []
    model = TorchClassifierWrap(LogReg(X_train.shape[1], len(classes)), 100, X_train.shape[1], 1000, random_state=42)
    learner = ActiveLearner(
        estimator=model,
        query_strategy=margin_sampling
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_pred_train = model.predict(X_train)
    metrics.append(get_confidence_intervals(y_pred, y_val, [accuracy_score])[0])
    train_metrics.append(get_confidence_intervals(y_pred_train, y_train, [accuracy_score])[0])
    for it in tqdm(range(10)):
        query_idx, _ = learner.query(X_pool, n_instances=1000)
        X_train = np.append(X_train, X_pool[query_idx], axis=0)
        y_train = np.append(y_train, y_pool[query_idx], axis=0)
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)
        model = TorchClassifierWrap(LogReg(X_train.shape[1], len(classes)), 100, X_train.shape[1], 1000, random_state=42)
        model.fit(X_train, y_train)
        learner.estimator = model
        y_pred = model.predict(X_val)
        y_pred_train = model.predict(X_train)
        metrics.append(get_confidence_intervals(y_pred, y_val, [accuracy_score])[0])
        train_metrics.append(get_confidence_intervals(y_pred_train, y_train, [accuracy_score])[0])
    with open("../../resources/cachedData/miscellaneous/bootstraptesting_saved_metrics/uncertancy_10it_1000batch.pkl", "wb") as file:
        pickle.dump(metrics, file)
    with open("../../resources/cachedData/miscellaneous/bootstraptesting_saved_metrics/uncertancy_10it_1000batch_train.pkl", "wb") as file:
        pickle.dump(train_metrics, file)
    confidence_intervals_to_plot(metrics, label="val")
    confidence_intervals_to_plot(train_metrics, label="learn")
    plt.show()

if __name__ == "__main__":
    main()