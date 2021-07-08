#convers 30 point data sets to 7 point data sets

#By Luke Mullis

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
from sklearn.neighbors import KNeighborsRegressor

#perf_data_1 = pd.read_csv('perfect_fish_30_0.csv')
#perf_data_1 = pd.read_csv('perfect_fish_30_1.csv')
#perf_data_1 = pd.read_csv('perfect_fish_30_2.csv')
#perf_data_1 = pd.read_csv('perfect_fish_30_3.csv')
#perf_data_1 = pd.read_csv('perfect_fish_30_4.csv')
#perf_data_1 = pd.read_csv('perfect_fish_30_5.csv')

#perf_data_1 = pd.read_csv('anom_data_30_1.csv')
#perf_data_1 = pd.read_csv('anom_30_1.csv')
#perf_data_1 = pd.read_csv('anom_data_30_2.csv')
#perf_data_1 = pd.read_csv('anom_30_2.csv')
#perf_data_1 = pd.read_csv('real_world_1_deeplabcut.csv')
#perf_data_1 = pd.read_csv('perfect_data_1.csv')
#perf_data_1 = pd.read_csv('midline_digfish_ perfect_2.csv')
perf_data_1 = pd.read_csv('real_world_2_deeplabcut.csv')


new_data = [perf_data_1["midx1"], perf_data_1["midx5"],perf_data_1["midx9"], perf_data_1["midx16"],perf_data_1['midx21'],perf_data_1['midx26'],perf_data_1['midx30'],perf_data_1["midy1"], perf_data_1["midy5"],perf_data_1["midy9"], perf_data_1["midy16"],perf_data_1['midy21'],perf_data_1['midy26'],perf_data_1['midy30']]

new_collumns = ["midx1", "midx2","midx3", "midx4","midx5", "midx6","midx7","midy1", "midy2","midy3", "midy4","midy5", "midy6","midy7"]


done = pd.concat(new_data, axis=1, keys=new_collumns)

#done = done.set_index('t')
print(done)



#done.to_csv('real_world_deeplab_7_2.csv', index=True)



















