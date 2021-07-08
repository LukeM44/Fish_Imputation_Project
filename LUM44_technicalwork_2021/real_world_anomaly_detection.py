#run anomaly detection code to data set

#BY Luke Mullis

#Done
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statistics
import math
import random
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import full_library2

from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor

#get data
perfect = pd.read_csv('real_world_7_normalised_2.csv')
deeplab = pd.read_csv('real_world_7_normalised_anomaly_detected_2.csv')

perfect.drop(perfect.columns[perfect.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
deeplab.drop(deeplab.columns[deeplab.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)


#print(perfect)
#print(deeplab)

attribute = 'midx4'
diffrence_needed = 0.05

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(deeplab.index,deeplab[attribute], color='blue', label = 'Normal')

plt.show();
#anomaly detection function	
full_library2.collective_anomaly_detection(deeplab,attribute, diffrence_needed)

#save data
#deeplab.to_csv('real_world_7_normalised_anomaly_detected_2.csv', index=True)








