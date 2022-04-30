import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def main():
    X = np.load("../../resources/cachedData/redditSelfPostVectorized/vectorized.npy", allow_pickle=True)
    y = np.load("../../resources/cachedData/redditSelfPostVectorized/vectorized_labels.npy", allow_pickle=True)
    sampled = np.random.choice(X.shape[0], 100000, replace=False)
    X = X[sampled]
    y = y[sampled]
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.999, random_state=42)
    X_pool, X_val, y_pool, y_val = train_test_split(X_test, y_test, test_size=0.3, random_state=42)
    classes = np.unique(y)
    print(X_train.shape, X_val.shape, X_pool.shape, sep=" ")
    model = TorchClassifierWrap(LogReg(X_train.shape[1], len(classes)), 100, X_train.shape[1], 1000, random_state=42)
