import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from lib.visualize import get_confidence_intervals, confidence_intervals_to_plot
plt.rcParams["figure.figsize"] = (20, 20)
mpl.style.use('ggplot')

def main():
    metrics = [[] for _ in range(2)]
    for it in range(1, 10):
        print((100 - it**2, 100))
        for i, metric_values in enumerate(get_confidence_intervals(np.append(np.zeros(100 - it**2), np.ones(it**2)), np.ones(100), [accuracy_score, precision_score])):
            metrics[i].append(metric_values)

    confidence_intervals_to_plot(metrics[0])
    confidence_intervals_to_plot(metrics[1])
    plt.show()


if __name__ == "__main__":
    main()