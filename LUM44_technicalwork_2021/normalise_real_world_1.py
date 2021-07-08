#normalises data sets

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
perfect = pd.read_csv('real_world_7_2.csv')
deeplab = pd.read_csv('real_world_2_deeplabcut.csv')


perfect.drop(perfect.columns[perfect.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
deeplab.drop(deeplab.columns[deeplab.columns.str.contains('t',case = False)],axis = 1, inplace = True)

#perfect = perfect.drop(['t'],axis = 1)
deeplab = deeplab.drop(['midx8','midy8','midx9','midy9','midx10','midy10','midx11','midy11','midx12','midy12','midx13','midy13','midx14','midy14','midx15','midy15','midx16','midy16','midx17','midy17'], axis = 1)



#make 'nan' cosistant
full_library2.make_nan_cosistant (perfect, deeplab)


perfect =perfect.reset_index(drop=True)
deeplab =deeplab.reset_index(drop=True)


perfect = perfect.drop(83)

print(perfect)
print(deeplab)

#normalise data
full_library2.normalise_data (deeplab,perfect)

print(perfect)
print(deeplab)

perfect = perfect.dropna()
deeplab = deeplab.dropna()



#print(perfect)
#print(deeplab)

#save data
#perfect.to_csv('real_world_7_normalised_2.csv', index=True)
#deeplab.to_csv('real_world_7_deeplab_normalised_2.csv', index=True)










