#adds anomalies to a perfect data set and saves the new data as well as the anomaly data

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

#set parameters for anoamlies
max_anom = 0.05
min_anom = -0.05
chance_of_anom = 0.18
min_size = 0.05

#get perfect data set
perf_data_2 = pd.read_csv('perfect_fish_30_1.csv')
del perf_data_2['t']

#print(perf_data_2)

#perf_data_2.drop(perf_data_2.columns[perf_data_2.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
#perf_data_2=perf_data_2.dropna()

#print(perf_data_2)

#add anomalies
luke = full_library2.contexual_anomaly_adder (perf_data_2,max_anom, min_anom, chance_of_anom, min_size)

#save data
#luke[0].to_csv('anom_data_30_2.csv', index=True)
#luke[1].to_csv('anom_30_2.csv', index=True)

#visulaise
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(luke[0]['midy2'], color='black', label = 'Normal')
#ax.plot(anom_data['midy7'], color='red', label = 'Normal')

plt.show();
