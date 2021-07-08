#tests the contexualanomaly detecion code

#By luke mullis

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



	

#get data
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

#make data fish relative
df1 = full_library2.fish_relative (anom_data, 'midx1', 'midy1')

#set parameters
y2=50
y3=100
y4=200
y5=300
y6=400
y7=500
x=100

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(df1['midy6'], color='black', label = 'Normal')
#ax.plot(anom_data['midy6'], color='red', label = 'Normal')

plt.show();
#run anomaly detection code
full_library2.contexual_anomaly_detection (df1,y2,y3,y4,y5,y6,y7,x)
#make camera relative
df1 = full_library2.camera_relative (df1,perf_data, 'midx1', 'midy1')
#test success of detection code
full_library2.contexual_anomaly_test (df1,anom)
