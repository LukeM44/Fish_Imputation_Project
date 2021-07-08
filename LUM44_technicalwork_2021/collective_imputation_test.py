#tests collective imputation code

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
#get data
anom_data = pd.read_csv('anom_data_30_2.csv')
anom = pd.read_csv('anom_30_2.csv')
anom_data.drop(anom_data.columns[anom_data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
anom.drop(anom.columns[anom.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

#make all anomlaies 'nan'
full_library2.make_all_anom_nan(anom_data,anom)




columns = list(anom_data)
for attribute in (columns):
#	if pd.isnull(anom_data[attribute][1]):
#		anom_data.at[1,attribute] = anom_data[attribute][2]
#	
#	
#	if pd.isnull(anom_data[attribute][0]):
#		anom_data.at[0,attribute] = anom_data[attribute][1]
#	 	
	if pd.isnull(anom_data[attribute][74]):
		anom_data.at[74,attribute] = anom_data[attribute][73]




#run imputation code
full_library2.collective_imputation (anom_data)

perf_data_1 = pd.read_csv('perfect_fish_30_1.csv')
del perf_data_1['t']
perf_data_1 =perf_data_1.reset_index(drop=True)

#test success of imputation code
full_library2.contexual_anomaly_imputation_test (perf_data_1,anom_data)



fig, ax = plt.subplots(figsize=(10,6))
ax.plot(perf_data_1['midy5'], color='red', label = 'Normal')
ax.plot(anom_data['midy5'], color='black', label = 'Normal')
plt.show();













