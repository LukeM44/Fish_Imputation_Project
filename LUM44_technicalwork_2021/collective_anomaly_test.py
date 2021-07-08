#Collective anomaly code test
#By Luke Mullis

#tests the collective anomaly code

#complete

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statistics
import math
import random

import full_library

data = pd.read_csv('perfect_data_2.csv') 

attribute = 'midx3'


max_anom = 100
min_anom = -100
#0 to 1
chance_of_anom = 0.08
min_size = 100



#add anoamlies
luke = full_library.collective_anomaly_adder (attribute,data,max_anom, min_anom, chance_of_anom,min_size)

#detect anomalies
df = full_library.collective_anomaly_detection(luke[0],attribute, 50)

#test success of anomaly detection code
df2 = full_library.collective_anomally_test(df, attribute,luke[1])

#imputenew data
df3 =full_library.collective_impute_new_data(df,attribute)
data = pd.read_csv('perfect_data_2.csv') 

#test success of imputation
full_library.collective_imputation_success_test(data,luke[0],attribute)

#visulise
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(data[attribute], color='red', label = 'Normal')
ax.plot(luke[0][attribute], color='black', label = 'Normal')
plt.show();
