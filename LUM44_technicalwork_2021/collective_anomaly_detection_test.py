#collective anomaly detection anomaly code test
#By Luke Mullis

#tests the success of the collective anoamly detection code
#for time series data.

#complete

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


anom_data = pd.read_csv('anom_data_30_2.csv')
anom = pd.read_csv('anom_30_2.csv')
anom_data.drop(anom_data.columns[anom_data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
anom.drop(anom.columns[anom.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)


attribute = 'midx3'
diffrence_needed = 0.05

#detect anomalies
full_library2.collective_anomaly_detection(anom_data,attribute, diffrence_needed)

#tests the success of the anomaly detection code
full_library2.contexual_anomaly_test (anom_data,anom)


































