#runs imputation code on real world data set 1

#By Luke Mullis

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

####
attribute = 'midx7'

#get training data
perf_data_1 = pd.read_csv('perfect_fish_7_0.csv')
perf_data_2 = pd.read_csv('perfect_fish_7_1.csv')
perf_data_3 = pd.read_csv('perfect_fish_7_2.csv')
perf_data_4 = pd.read_csv('perfect_fish_7_3.csv')
perf_data_5 = pd.read_csv('perfect_fish_7_4.csv')
perf_data_6 = pd.read_csv('perfect_fish_7_5.csv')
perf_data_7 = pd.read_csv('real_world_7_normalised_2.csv')


perf_data = pd.concat([perf_data_1,perf_data_2,perf_data_3, perf_data_4,perf_data_5, perf_data_6,perf_data_7], ignore_index=True)
#print(perf_data)
del perf_data['t']

#get test data
perfect = pd.read_csv('real_world_7_normalised_1.csv')
deeplab = pd.read_csv('real_world_7_normalised_anomaly_detected_1.csv')
original = pd.read_csv('real_world_7_deeplab_normalised_1.csv')

perfect.drop(perfect.columns[perfect.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
deeplab.drop(deeplab.columns[deeplab.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

#reorder test set attributes
deeplab = full_library2.reorder (deeplab)

###
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(original[attribute], color='black', label = 'Normal')
plt.show();



fig, ax = plt.subplots(figsize=(10,6))
ax.plot(deeplab[attribute], color='black', label = 'Normal')
plt.show();
###

#make test and train set fish relative
train = full_library2.fish_relative (perf_data, 'midx1', 'midy1')

test = full_library2.fish_relative (deeplab, 'midx1', 'midy1')
train.drop(train.columns[train.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
#print(train)
#print(test)

#impute new data
imp = IterativeImputer(ExtraTreesRegressor() )	
imp.fit(train)

result = pd.DataFrame(data=np.array(imp.transform(test)), columns=test.columns, index=test.index)

perf_data_1 = pd.read_csv('real_world_7_normalised_anomaly_detected_1.csv')
#luke = pd.read_csv('real_world_7_deeplab_normalised_2.csv')
#luke.drop(luke.columns[luke.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

#del perf_data_1['t']
perf_data_1 =perf_data_1.reset_index(drop=True)

#make camera relative
cam_rel = full_library2.camera_relative(result,perf_data_1, 'midx1', 'midy1')

#visulise
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(cam_rel[attribute], color='red', label = 'Normal')
ax.plot(perf_data_1[attribute], color='black', label = 'Normal')
plt.show();

#test imputation sucess
full_library2.contexual_anomaly_imputation_test (cam_rel,perfect)



###
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(cam_rel[attribute], color='red', label = 'Normal')
ax.plot(original[attribute], color='black', label = 'Normal')
plt.show();
###
