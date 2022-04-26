import pickle
import random
import logging
import numpy as np
import pickle
from lib.models.Wraps import TorchClassifierWrap
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from lib.models.Linear import LogReg
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
X = np.load("../../resources/cachedData/redditSelfPostVectorized/vectorized.npy", allow_pickle=True)
y = np.load("../../resources/cachedData/redditSelfPostVectorized/vectorized_labels.npy", allow_pickle=True)
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=42)
X_pool, X_val, y_pool, y_val = train_test_split(X_test, y_test, test_size=0.3, random_state=42)
classes = np.unique(y)


logging.info("Full model start")
for i in range(1):
    model_full = TorchClassifierWrap(LogReg(X_train.shape[1], len(classes)), 1000, X_train.shape[1], 100, random_state=42, verbose=True)
    model_full.fit(X_train, y_train)
    with open("../../resources/cachedData/trainedModels/redditSelfPost/logreg1000it100batch.pkl", "wb") as file:
        pickle.dump(model_full.get_state(), file)
    y_pred = model_full.predict(X_val)
    print((accuracy_score(y_pred, y_val), precision_score(y_pred, y_val, average="macro")))

print("---")
X_truncated = PCA(n_components=30).fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_truncated, y, test_size=0.95, random_state=42)
X_pool, X_val, y_pool, y_val = train_test_split(X_test, y_test, test_size=0.3, random_state=42)

logging.info("Truncated model start")
for i in range(1):
    model_truncated = TorchClassifierWrap(LogReg(30, len(classes)), 1000, X_train.shape[1], 100, random_state=42, verbose=True)
    model_truncated.fit(X_train, y_train)
    y_pred = model_truncated.predict(X_val)
    print((accuracy_score(y_pred, y_val), precision_score(y_pred, y_val, average="macro")))
