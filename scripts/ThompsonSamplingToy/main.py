import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Estimator:
    def __init__(self):
        self.n = 0  # the number of times this socket has been tried
        self.x = []  # list of all samples

        self.al = 1  # gamma shape parameter
        self.b = 10  # gamma rate parameter

        self.mean = 1  # the prior (estimated) mean
        self.var = self.b / (self.al + 1)  # the prior (estimated) variance

    def sample(self):
        precision = np.random.gamma(self.al, 1 / self.b)
        if precision == 0 or self.n == 0: precision = 0.001

        estimated_variance = 1 / precision
        return np.random.normal(self.mean, np.sqrt(estimated_variance))

    def update(self, x):
        n = 1
        v = self.n

        self.al = self.al + n / 2
        self.b = self.b + ((n * v / (v + n)) * (((x - self.mean) ** 2) / 2))

        self.var = self.b / (self.al + 1)

        self.x.append(x)
        self.n += 1
        self.mean = np.array(self.x).mean()


class Bandit:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def sample(self):
        return np.random.normal(self.mean, np.sqrt(self.var))


def run():
    bandits = [Bandit(100, 12), Bandit(200, 20), Bandit(100, 31), Bandit(65, 1)]
    estimators = [Estimator() for _ in bandits]
    count = [0 for _ in bandits]
    for it in range(100):
        sampled = [estimator.sample() for estimator in estimators]
        highest = max(range(len(sampled)), key=lambda x: sampled[x])
        estimators[highest].update(bandits[highest].sample())
        count[highest] += 1
    return count


def main():
    summed_count = run()
    for att in tqdm(range(100)):
        new_count = run()
        summed_count = [summed + new for summed, new in zip(summed_count, new_count)]
    plt.bar([str(i) for i in range(len(summed_count))], summed_count)
    plt.show()


if __name__ == "__main__":
    main()
