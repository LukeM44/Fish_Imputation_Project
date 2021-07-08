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

max_anom = 0.15
min_anom = -0.15
chance_of_anom = 0.08
min_size = 0.1


#perf_data_1 = pd.read_csv('perfect_data_1.csv')
#perf_data_2 = pd.read_csv('perfect_data_2.csv')
#perf_data_1.drop(perf_data_1.columns[perf_data_1.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
#perf_data_1=perf_data_1.dropna()
#perf_data_2.drop(perf_data_2.columns[perf_data_2.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
#print(perf_data_2)


#fig, ax = plt.subplots(figsize=(10,6))
#ax.plot(perf_data_1['midx3'], color='black', label = 'Normal')
#ax.plot(perf_data_2['midx3'], color='red', label = 'Normal')

#plt.show();

#perf_data = pd.concat([perf_data_1, perf_data_2], ignore_index=True)

perf_data = pd.read_csv('perfect_data_2.csv')
anom_data = pd.read_csv('anom_data_2.csv')
anom = pd.read_csv('anom_2.csv')



perf_data.drop(perf_data.columns[perf_data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
anom_data.drop(anom_data.columns[anom_data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
anom.drop(anom.columns[anom.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

perf_data=perf_data.dropna()
anom_data=anom_data.dropna()
perf_data =perf_data.reset_index(drop=True)
anom_data =anom_data.reset_index(drop=True)
anom =anom.reset_index(drop=True)

full_library2.make_all_anom_nan(anom_data,anom)



print(anom_data)



train = full_library2.fish_relative (perf_data, 'midx1', 'midy1')


test = full_library2.fish_relative (anom_data, 'midx1', 'midy1')

#fig, ax = plt.subplots(figsize=(10,6))
#ax.plot(train['midy7'], color='black', label = 'Normal')
#ax.plot(test['midy7'], color='red', label = 'Normal')

#plt.show();

imp = IterativeImputer(  KNeighborsRegressor(n_neighbors=2),n_nearest_features = 1)	
imp.fit(train)



columnss = list(test)
result = pd.DataFrame(data=np.array(imp.transform(test)), columns=test.columns, index=test.index)
#print(result)
cam_rel = full_library2.camera_relative(result,anom_data, 'midx1', 'midy1')

#anom_data = pd.read_csv('anom_data_1.csv')

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(cam_rel['midy6'], color='black', label = 'Normal')
ax.plot(anom_data['midy6'], color='red', label = 'Normal')

plt.show();

data = pd.read_csv('perfect_data_2.csv') 
data.drop(data.columns[data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
data=data.dropna()
data =data.reset_index(drop=True)
full_library2.contexual_anomaly_imputation_test (data,cam_rel)
