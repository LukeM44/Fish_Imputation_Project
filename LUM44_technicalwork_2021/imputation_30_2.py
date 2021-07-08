#tests imputation code on datasets with 30 points

#by Luke Mullis

#done

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

#get training data
perf_data_1 = pd.read_csv('perfect_fish_30_0.csv')
#perf_data_2 = pd.read_csv('perfect_fish_30_1.csv')
perf_data_3 = pd.read_csv('perfect_fish_30_2.csv')
perf_data_4 = pd.read_csv('perfect_fish_30_3.csv')
perf_data_5 = pd.read_csv('perfect_fish_30_4.csv')
perf_data_6 = pd.read_csv('perfect_fish_30_5.csv')


perf_data = pd.concat([ perf_data_1,perf_data_3, perf_data_4,perf_data_5, perf_data_6], ignore_index=True)
del perf_data['t']
#print(perf_data)

#get test data
anom_data = pd.read_csv('anom_data_30_2.csv')
anom = pd.read_csv('anom_30_2.csv')
anom_data.drop(anom_data.columns[anom_data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
anom.drop(anom.columns[anom.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)



#set all anoamlies to 'nan'
full_library2.make_all_anom_nan(anom_data,anom)

#make train and sets set realtive to the fish's snout
train = full_library2.fish_relative (perf_data, 'midx1', 'midy1')

test = full_library2.fish_relative (anom_data, 'midx1', 'midy1')
#print(test)

#set up imputer
imp = IterativeImputer( ExtraTreesRegressor())	
imp.fit(train)

#impute new data
columnss = list(test)
result = pd.DataFrame(data=np.array(imp.transform(test)), columns=test.columns, index=test.index)
#print(result)
perf_data_1 = pd.read_csv('perfect_fish_30_1.csv')
del perf_data_1['t']
cam_rel = full_library2.camera_relative(result,perf_data_1, 'midx1', 'midy1')

#visulise
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(cam_rel['midy15'], color='red', label = 'Normal')
ax.plot(anom_data['midy15'], color='black', label = 'Normal')
plt.show();
#test imputation success
full_library2.contexual_anomaly_imputation_test (perf_data_1,cam_rel)
