import ast
import os
import time
from ast import literal_eval

import pandas as pd
from haversine import haversine
from sklearn import preprocessing

import sys
sys.path.insert(0, '../knearest')
import gmplot as gm
import knearest_lcs


start_time = time.time()
trainSet = pd.read_csv(
    '../../datasets/train_set.csv',
    converters={"Trajectory": literal_eval}, index_col='tripId'
)

testSet = pd.read_csv('../../datasets/test_set_a2.csv', sep="/t")

trainSet_lst = []
for index, row in trainSet.iterrows():
    flag = True
    temp_traj = []
    line = row['Trajectory']
    prev_pair = []
    for x in line:
        if flag or (x[1] != prev_pair[0] and x[2] != prev_pair[1]):
            temp_pair = [x[1], x[2]]
            temp_traj.append(temp_pair)
            prev_pair = temp_pair
            flag = False
    # trainSet_lst.append(temp_traj)
    row['Trajectory'] = temp_traj

testSet_lst = []
for index, row in testSet.iterrows():
    flag = True
    temp_traj = []
    line = ast.literal_eval(row['Trajectory'])
    prev_pair = []
    for x in line:
        if flag or (x[1] != prev_pair[0] and x[2] != prev_pair[1]):
            temp_pair = [x[1], x[2]]
            temp_traj.append(temp_pair)
            prev_pair = temp_pair
            flag = False
    # testSet_lst.append(temp_traj)
    row['Trajectory'] = temp_traj

le = preprocessing.LabelEncoder()
le.fit(trainSet['journeyPatternId'])
y = le.transform(trainSet['journeyPatternId'])
knn = knearest_lcs.KnnClassifier()
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

    i=1

    for y in prediction:
        start_red = False
        finished = False
        traj_row = trainSet['Trajectory'][y[2]]
        pathlon = []
        pathlat = []
        redpathlon = []
        redpathlat = []

        for k in y[3]:
            redpathlon.append(k[0])
            redpathlat.append(k[1])
        for x in traj_row:
            pathlon.append(x[0])
            pathlat.append(x[1])

        #     if round(x[0], 6) == round(y[3][0][0], 6):
        #         if round(x[1], 6) == round(y[3][0][1], 6):
        #             start_red = True
        #     if start_red and not finished:
        #         redpathlon.append(x[0])
        #         redpathlat.append(x[1])
        #     if round(x[0], 6) == round(y[3][len(y[3])-1][0], 6) and round(x[1], 6) == round(y[3][len(y[3])-1][1], 6):
        #         start_red = False
        #         finished = True
        # print redpathlon
        gmap = gm.GoogleMapPlotter(pathlat[(len(pathlat) // 2) - 1], pathlon[(len(pathlon) // 2) - 1], 13)
        gmap.plot(pathlat, pathlon, 'cornflowerblue', edge_width=5)
        gmap.plot(redpathlat, redpathlon, 'red', edge_width=5)
        gmap.draw(directory + '/map_neighbor_' + str(i) + '_' + le.inverse_transform(y[0]) + '_' + str(y[1]) + '.html')
        i=i+1

print("--- %s seconds ---" % (time.time() - start_time))
