# coding=utf-8
import ast
import csv
import os

import pandas as pd
from ast import literal_eval
import gmplot as gm
from fastdtw import fastdtw
from sklearn.metrics import accuracy_score
from sklearn.neighbors.dist_metrics import DistanceMetric
import time
from sklearn.cross_validation import KFold, cross_val_score
import sys

from knearest import KnnClassifier
from sklearn import preprocessing
from sklearn import cross_validation

start_time = time.time()
trainSet = pd.read_csv(
    '../../datasets/train_set.csv',
    converters={"Trajectory": literal_eval}, index_col='tripId'
)

testSet = pd.read_csv('../../datasets/test_set_a1.csv', sep="/t")

trainSet_proc = []
label_proc = []
i = 1;
for index, row in trainSet.iterrows():
    flag = True
    temp_traj = []
    prev_pair = []
    line = row['Trajectory']
    for x in line:
        if flag or (x[1] != prev_pair[0] and x[2] != prev_pair[1]):
            temp_pair = [x[1], x[2]]
            temp_traj.append(temp_pair)
            prev_pair = temp_pair
            flag = False
    trainSet_proc.append(temp_traj)
    label_proc.append(row['journeyPatternId'])
    i = i + 1
    if i > len(trainSet['Trajectory']) / 50:
        break

testSet_proc = []
for index, row in testSet.iterrows():
    flag = True
    prev_pair = []
    temp_traj = []
    line = ast.literal_eval(row['Trajectory'])
    for x in line:
        if flag or (x[1] != prev_pair[0] and x[2] != prev_pair[1]):
            temp_pair = [x[1], x[2]]
            temp_traj.append(temp_pair)
            prev_pair = temp_pair
            flag = False
    testSet_proc.append(temp_traj)

le = preprocessing.LabelEncoder()
le.fit(label_proc)
y = le.transform(label_proc)
knn = KnnClassifier()
k_fold = cross_validation.KFold(len(trainSet_proc), n_folds=10, shuffle=True, random_state=42)
results = cross_val_score(knn, trainSet_proc,
                          y, fit_params={'isCrossVal': True},
                          cv=k_fold,
                          scoring='accuracy')
print results

# knn = KnnClassifier()
# knn.fit(trainSet_proc, y, isCrossVal=False)
# results = knn.predict(testSet_proc)
#
# with open("./testSet_JourneyPatternIDs.csv", "wb") as csv_file:
#     writer = csv.writer(csv_file)
#     writer.writerow(("Test_Trip_ID", "Predicted_JourneyPatternID"))
#     for i in range(len(results)):
#         writer.writerow((i, le.inverse_transform(results[i])))

print("--- %s seconds ---" % (time.time() - start_time))
