import ast
import os

import pandas as pd
from ast import literal_eval
import gmplot as gm
from fastdtw import fastdtw
from sklearn.neighbors.dist_metrics import DistanceMetric
import time
import sys

sys.path.insert(0, '../knearest')
import knearest_dtw
from sklearn import preprocessing

start_time = time.time()
trainSet = pd.read_csv(
    '../../datasets/train_set.csv',
    converters={"Trajectory": literal_eval}, index_col='tripId'
)

testSet = pd.read_csv('../../datasets/test_set_a1.csv', sep="/t")

for index, row in trainSet.iterrows():
    temp_traj = []
    line = row['Trajectory']
    for x in line:
        temp_pair = []
        temp_pair.append(x[1])
        temp_pair.append(x[2])
        temp_traj.append(temp_pair)
    row['Trajectory'] = temp_traj

for index, row in testSet.iterrows():
    temp_traj = []
    line = ast.literal_eval(row['Trajectory'])
    for x in line:
        temp_pair = []
        temp_pair.append(x[1])
        temp_pair.append(x[2])
        temp_traj.append(temp_pair)
    row['Trajectory'] = temp_traj

le = preprocessing.LabelEncoder()
le.fit(trainSet['journeyPatternId'])
y = le.transform(trainSet['journeyPatternId'])
knn = knearest_dtw.KnnClassifier()
knn.fit(trainSet, y, isCrossVal=False)
predictions = knn.predict(testSet)

for index, row in testSet.iterrows():
    prediction = predictions[index]
    directory = './test_' + str(index)
    if not os.path.exists(directory):
        os.makedirs(directory)
    line = row['Trajectory']
    pathlon = []
    pathlat = []
    for x in line:
        pathlon.append(x[0])
        pathlat.append(x[1])
    gmap = gm.GoogleMapPlotter(pathlat[(len(pathlat) // 2) - 1], pathlon[(len(pathlon) // 2) - 1], 13)
    gmap.plot(pathlat, pathlon, 'cornflowerblue', edge_width=5)
    gmap.draw(directory + '/map' + '.html')

    i = 1
    for y in prediction:
        traj_row = trainSet['Trajectory'][y[2]]
        pathlon = []
        pathlat = []
        for x in traj_row:
            pathlon.append(x[0])
            pathlat.append(x[1])
        gmap = gm.GoogleMapPlotter(pathlat[(len(pathlat) // 2) - 1], pathlon[(len(pathlon) // 2) - 1], 13)
        gmap.plot(pathlat, pathlon, 'cornflowerblue', edge_width=5)
        gmap.draw(directory + '/map_neighbor_' + str(i) + '_' + le.inverse_transform(y[0]) + '_' + str(y[1]) + '.html')
        i = i + 1

print("--- %s seconds ---" % (time.time() - start_time))
