#imputing x and y data seperatly

#by Luke Mullis

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

#get trinaing data
perf_data_1 = pd.read_csv('perfect_fish_30_0.csv')
#perf_data_2 = pd.read_csv('perfect_fish_30_1.csv')
perf_data_3 = pd.read_csv('perfect_fish_30_2.csv')
perf_data_4 = pd.read_csv('perfect_fish_30_3.csv')
perf_data_5 = pd.read_csv('perfect_fish_30_4.csv')
perf_data_6 = pd.read_csv('perfect_fish_30_5.csv')


perf_data = pd.concat([ perf_data_1,perf_data_3, perf_data_4,perf_data_5, perf_data_6], ignore_index=True)
del perf_data['t']

#get test data
anom_data = pd.read_csv('anom_data_30_2.csv')
anom = pd.read_csv('anom_30_2.csv')
anom_data.drop(anom_data.columns[anom_data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
anom.drop(anom.columns[anom.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)


#make all anomalies 'nan'
full_library2.make_all_anom_nan(anom_data,anom)



#make train and test set realtive to fish's snout
train = full_library2.fish_relative (perf_data, 'midx1', 'midy1')
test = full_library2.fish_relative (anom_data, 'midx1', 'midy1')

#split the data into x and y
train_data_x = [train["midx1"], train["midx2"],train["midx3"], train["midx4"],train["midx5"], train["midx6"],train["midx7"], train["midx8"],train["midx9"], train["midx10"],train["midx11"], train["midx12"],train["midx13"], train["midx14"],train["midx15"], train["midx16"],train["midx17"], train["midx18"],train["midx19"], train["midx20"],train["midx21"], train["midx22"],train["midx23"], train["midx24"],train["midx25"], train["midx26"],train["midx27"], train["midx28"],train["midx29"], train["midx30"]]

train_headers_x = ["midx1", "midx2","midx3", "midx4","midx5", "midx6","midx7", "midx8","midx9", "midx10","midx11", "midx12","midx13", "midx14","midx15", "midx16","midx17", "midx18","midx19", "midx20","midx21", "midx22","midx23", "midx24","midx25", "midx26","midx27", "midx28","midx29", "midx30"]

train_just_x = pd.concat(train_data_x, axis=1, keys=train_headers_x)



train_data_y = [train["midy1"], train["midy2"],train["midy3"], train["midy4"],train["midy5"], train["midy6"],train["midy7"], train["midy8"],train["midy9"], train["midy10"],train["midy11"], train["midy12"],train["midy13"], train["midy14"],train["midy15"], train["midy16"],train["midy17"], train["midy18"],train["midy19"], train["midy20"],train["midy21"], train["midy22"],train["midy23"], train["midy24"],train["midy25"], train["midy26"],train["midy27"], train["midy28"],train["midy29"], train["midy30"]]

train_headers_y = ["midy1", "midy2","midy3", "midy4","midy5", "midy6","midy7", "midy8","midy9", "midy10","midy11", "midy12","midy13", "midy14","midy15", "midy16","midy17", "midy18","midy19", "midy20","midy21", "midy22","midy23", "midy24","midy25", "midy26","midy27", "midy28","midy29", "midy30"]

train_just_y = pd.concat(train_data_y, axis=1, keys=train_headers_y)



test_data_x = [test["midx1"], test["midx2"],test["midx3"], test["midx4"],test["midx5"], test["midx6"],test["midx7"], test["midx8"],test["midx9"], test["midx10"],test["midx11"], test["midx12"],test["midx13"], test["midx14"],test["midx15"], test["midx16"],test["midx17"], test["midx18"],test["midx19"], test["midx20"],test["midx21"], test["midx22"],test["midx23"], test["midx24"],test["midx25"], test["midx26"],test["midx27"], test["midx28"],test["midx29"], test["midx30"]]

test_headers_x = ["midx1", "midx2","midx3", "midx4","midx5", "midx6","midx7", "midx8","midx9", "midx10","midx11", "midx12","midx13", "midx14","midx15", "midx16","midx17", "midx18","midx19", "midx20","midx21", "midx22","midx23", "midx24","midx25", "midx26","midx27", "midx28","midx29", "midx30"]

test_just_x = pd.concat(test_data_x, axis=1, keys=test_headers_x)



test_data_y = [test["midy1"], test["midy2"],test["midy3"], test["midy4"],test["midy5"], test["midy6"],test["midy7"], test["midy8"],test["midy9"], test["midy10"],test["midy11"], test["midy12"],test["midy13"], test["midy14"],test["midy15"], test["midy16"],test["midy17"], test["midy18"],test["midy19"], test["midy20"],test["midy21"], test["midy22"],test["midy23"], test["midy24"],test["midy25"], test["midy26"],test["midy27"], test["midy28"],test["midy29"], test["midy30"]]

test_headers_y = ["midy1", "midy2","midy3", "midy4","midy5", "midy6","midy7", "midy8","midy9", "midy10","midy11", "midy12","midy13", "midy14","midy15", "midy16","midy17", "midy18","midy19", "midy20","midy21", "midy22","midy23", "midy24","midy25", "midy26","midy27", "midy28","midy29", "midy30"]

test_just_y = pd.concat(test_data_y, axis=1, keys=test_headers_y)

#impute new data
#################
impx = IterativeImputer()	
impx.fit(train_just_x)
###############
impy = IterativeImputer()	
impy.fit(train_just_y)
####################
columnss = list(test_just_x)
df4 = pd.DataFrame(data=np.array(impx.transform(test_just_x)), columns=test_just_x.columns, index=test_just_x.index)

data1 = pd.read_csv('perfect_fish_30_1.csv',index_col =0)
data1.reset_index(drop=True, inplace=True)

#convert back to camera relaive
cam_rel_x = full_library2.camera_relative (df4,data1, 'midx1', 'midy1')


columnss = list(test_just_y)
df5 = pd.DataFrame(data=np.array(impy.transform(test_just_y)), columns=test_just_y.columns, index=test_just_y.index)

data1 = pd.read_csv('perfect_fish_30_1.csv',index_col =0)
data1.reset_index(drop=True, inplace=True)
cam_rel_y = full_library2.camera_relative (df5,data1, 'midx1', 'midy1')

#put x and y back together
data = [cam_rel_x["midx1"], cam_rel_x["midx2"],cam_rel_x["midx3"], cam_rel_x["midx4"],cam_rel_x["midx5"], cam_rel_x["midx6"],cam_rel_x["midx7"], cam_rel_x["midx8"],cam_rel_x["midx9"], cam_rel_x["midx10"],cam_rel_x["midx11"], cam_rel_x["midx12"],cam_rel_x["midx13"], cam_rel_x["midx14"],cam_rel_x["midx15"], cam_rel_x["midx16"],cam_rel_x["midx17"], cam_rel_x["midx18"],cam_rel_x["midx19"], cam_rel_x["midx20"],cam_rel_x["midx21"], cam_rel_x["midx22"],cam_rel_x["midx23"], cam_rel_x["midx24"],cam_rel_x["midx25"], cam_rel_x["midx26"],cam_rel_x["midx27"], cam_rel_x["midx28"],cam_rel_x["midx29"], cam_rel_x["midx30"],cam_rel_y["midy1"], cam_rel_y["midy2"],cam_rel_y["midy3"], cam_rel_y["midy4"],cam_rel_y["midy5"], cam_rel_y["midy6"],cam_rel_y["midy7"], cam_rel_y["midy8"],cam_rel_y["midy9"], cam_rel_y["midy10"],cam_rel_y["midy11"], cam_rel_y["midy12"],cam_rel_y["midy13"], cam_rel_y["midy14"],cam_rel_y["midy15"], cam_rel_y["midy16"],cam_rel_y["midy17"], cam_rel_y["midy18"],cam_rel_y["midy19"], cam_rel_y["midy20"],cam_rel_y["midy21"], cam_rel_y["midy22"],cam_rel_y["midy23"], cam_rel_y["midy24"],cam_rel_y["midy25"], cam_rel_y["midy26"],cam_rel_y["midy27"], cam_rel_y["midy28"],cam_rel_y["midy29"], cam_rel_y["midy30"]]

headers = ["midx1", "midx2","midx3", "midx4","midx5", "midx6","midx7", "midx8","midx9", "midx10","midx11", "midx12","midx13", "midx14","midx15", "midx16","midx17", "midx18","midx19", "midx20","midx21", "midx22","midx23", "midx24","midx25", "midx26","midx27", "midx28","midx29", "midx30","midy1", "midy2","midy3", "midy4","midy5", "midy6","midy7", "midy8","midy9", "midy10","midy11", "midy12","midy13", "midy14","midy15", "midy16","midy17", "midy18","midy19", "midy20","midy21", "midy22","midy23", "midy24","midy25", "midy26","midy27", "midy28","midy29", "midy30"]

done = pd.concat(data, axis=1, keys=headers)
print(done)

data_1 = pd.read_csv('perfect_fish_30_1.csv')

#test imputation success
full_library2.contexual_anomaly_imputation_test (data1,done)

#visulise
anom_data = pd.read_csv('anom_data_30_2.csv')
anom_data.drop(anom_data.columns[anom_data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)


fig, ax = plt.subplots(figsize=(10,6))

ax.plot(anom_data['midy28'], color='red', label = 'Normal')
ax.plot(done['midy28'], color='black', label = 'Normal')
plt.show();
