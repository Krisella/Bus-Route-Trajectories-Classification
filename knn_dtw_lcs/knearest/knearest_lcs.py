import ast

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from fastdtw import fastdtw
from collections import Counter
from haversine import haversine
from sklearn.neighbors.dist_metrics import DistanceMetric


def lcs(a, b):
    lengths = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if (haversine(x, y) * 1000) <= 200:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])

    result = []
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x - 1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y - 1]:
            y -= 1
        else:
            # assert (haversine(a[x - 1], b[y - 1]) * 1000) <= 200
            result.append(a[x-1])
            x -= 1
            y -= 1
    return result


class KnnClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y, **kwargs):
        if 'k' in kwargs:
            self.k = kwargs['k']
        self.isCrossVal = kwargs['isCrossVal']
        self.X = X
        self.y = y
        return self

    def predict(self, X):
        # loop over all observations
        predictions = []
        for index, row in X.iterrows():
            predictions.append(self.knn_predict(self.X, self.y, row['Trajectory'], self.k))
        return predictions

    def knn_predict(self, X_train, y_train, x_test, k):
        # create list for distances and targets
        distances = []
        targets = []

        i = 0;
        for index, row in X_train.iterrows():
            # distance, path = fastdtw(x_test, row['Trajectory'], dist=haversine)
            results = lcs(row['Trajectory'], x_test)
            distance = len(results)
            # add it to list of distances
            distances.append([distance, i, index, results])
            i = i + 1
        # sort the list
        distances = sorted(distances, reverse=True)

        # make a list of the k neighbors' targets
        for i in range(k):
            if self.isCrossVal:
                targets.append(y_train[distances[i][1]])
            else:
                targets.append([y_train[distances[i][1]], distances[i][0], distances[i][2], distances[i][3]])
        # return most common target
        if self.isCrossVal:
            return Counter(targets).most_common(1)[0][0]
        else:
            return targets
