import ast

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from fastdtw import fastdtw
from collections import Counter
from haversine import haversine
from sklearn.neighbors.dist_metrics import DistanceMetric

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
        for row in X:
            predictions.append(self.knn_predict(self.X, self.y, row, self.k))
        return predictions

    def knn_predict(self, X_train, y_train, x_test, k):
        # create list for distances and targets
        distances = []
        targets = []

        i = 0;
        for row in X_train:
            distance, path = fastdtw(x_test, row, dist=haversine)
            # add it to list of distances
            distances.append([distance, i])
            i = i + 1
        # sort the list
        distances = sorted(distances)

        # make a list of the k neighbors' targets
        for i in range(k):
                targets.append(y_train[distances[i][1]])

        # return most common target
        return Counter(targets).most_common(1)[0][0]

