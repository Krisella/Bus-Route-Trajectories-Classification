import pandas as pd
from ast import literal_eval
import gmplot as gm

trainSet = pd.read_csv(
'../datasets/train_set.csv', 
converters={"Trajectory": literal_eval},
index_col='tripId'
)
dict = {}
for index, row in trainSet.iterrows():
    if row['journeyPatternId'] not in dict:
        dict[row['journeyPatternId']] = row['Trajectory']
        print(row['Trajectory'])
        line = row['Trajectory']
        pathlon = [];
        pathlat = [];
        for x in line:
            pathlon.append(x[1])
            pathlat.append(x[2])
        gmap = gm.GoogleMapPlotter(pathlat[(len(pathlat)//2)-1], pathlon[(len(pathlon)//2)-1], 13)
        gmap.plot(pathlat, pathlon, 'cornflowerblue', edge_width=5)
        gmap.draw('map_' + row['journeyPatternId'] + '.html')
    if len(dict.keys()) >= 5:
        break


