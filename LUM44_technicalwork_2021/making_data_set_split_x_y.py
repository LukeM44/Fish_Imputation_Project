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

max_anom = 0.15
min_anom = -0.15
chance_of_anom = 0.08
min_size = 0.1



#data = pd.read_csv('midline_digfish_ perfect_2.csv') 
#data=data.dropna()
#data =data.reset_index(drop=True)
#new_data = [data['midx1'], data['midx5'], data['midx9'],data['midx16'],data['midx21'],data['midx26'],data['midx30'],data['midy1'], data['midy5'], data['midy9'],data['midy16'],data['midy21'],data['midy26'],data['midy30']]
#new_headers = ['midx1','midx2','midx3','midx4','midx5','midx6','midx7','midy1','midy2','midy3','midy4','midy5','midy6','midy7']

#new_data = pd.concat(new_data, axis=1, keys=new_headers)
#perf_df1 = pd.DataFrame(data=new_data, columns=new_headers, index=data.index)

#perf_df1.to_csv('perfect_data_2.csv', index=True)

#luke = full_library2.contexual_anomaly_adder (perf_df1,max_anom, min_anom, chance_of_anom, min_size)

#luke[0].to_csv('anom_data_2.csv', index=True)
#luke[1].to_csv('anom_2.csv', index=True)
#perf_data = pd.read_csv('perfect_data_1.csv')
#perf_data=perf_data.dropna()
#luke = full_library2.contexual_anomaly_adder (perf_data,max_anom, min_anom, chance_of_anom, min_size)
#luke[0].to_csv('anom_data_2.csv', index=True)
#luke[1].to_csv('anom_2.csv', index=True)

perf_data_1 = pd.read_csv('perfect_data_1.csv')
perf_data_2 = pd.read_csv('perfect_data_2.csv')

perf_data = pd.concat([perf_data_1, perf_data_2], ignore_index=True)

anom_data = pd.read_csv('anom_data_1.csv')
anom = pd.read_csv('anom_1.csv')


perf_data.drop(perf_data.columns[perf_data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
anom_data.drop(anom_data.columns[anom_data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
anom.drop(anom.columns[anom.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

perf_data=perf_data.dropna()
anom_data=anom_data.dropna()
perf_data =perf_data.reset_index(drop=True)


full_library2.make_all_anom_nan(anom_data,anom)

#fig, ax = plt.subplots(figsize=(10,6))
#ax.plot(anom_data['midy2'], color='red', label = 'Normal')
#plt.show();
#print(anom_data)

train = full_library2.fish_relative (perf_data, 'midx1', 'midy1')
test = full_library2.fish_relative (anom_data, 'midx1', 'midy1')


train_data_x = [train["midx1"], train["midx2"],train["midx3"], train["midx4"],train["midx5"], train["midx6"],train["midx7"]]

train_headers_x = ["midx1", "midx2","midx3", "midx4","midx5", "midx6","midx7"]

train_just_x = pd.concat(train_data_x, axis=1, keys=train_headers_x)


train_data_y = [train["midy1"], train["midy2"],train["midy3"], train["midy4"],train["midy5"], train["midy6"],train["midy7"]]

train_headers_y = ["midy1", "midy2","midy3", "midy4","midy5", "midy6","midy7"]

train_just_y = pd.concat(train_data_y, axis=1, keys=train_headers_y)


test_data_x = [test["midx1"], test["midx2"],test["midx3"], test["midx4"],test["midx5"], test["midx6"],test["midx7"]]

test_headers_x = ["midx1", "midx2","midx3", "midx4","midx5", "midx6","midx7"]

test_just_x = pd.concat(test_data_x, axis=1, keys=test_headers_x)



test_data_y = [test["midy1"], test["midy2"],test["midy3"], test["midy4"],test["midy5"], test["midy6"],test["midy7"]]

test_headers_y = ["midy1", "midy2","midy3", "midy4","midy5", "midy6","midy7"]

test_just_y = pd.concat(test_data_y, axis=1, keys=test_headers_y)





impx = IterativeImputer(max_iter=10, random_state=0, sample_posterior = False, n_nearest_features=7)	
impx.fit(train_just_x)

impy = IterativeImputer(max_iter=10, random_state=0, sample_posterior = False, n_nearest_features=7)	
impy.fit(train_just_y)






perf_data = pd.read_csv('perfect_data_1.csv')
perf_data=perf_data.dropna()
perf_data =perf_data.reset_index(drop=True)

columnss = list(test_just_x)
df4 = pd.DataFrame(data=np.array(impx.transform(test_just_x)), columns=test_just_x.columns, index=test_just_x.index)
#print(df4)
data1 = pd.read_csv('perfect_data_1.csv',index_col =0)
cam_rel_x = full_library2.camera_relative (df4,perf_data, 'midx1', 'midy1')
#print(cam_rel_x)

columnss = list(test_just_y)
df5 = pd.DataFrame(data=np.array(impy.transform(test_just_y)), columns=test_just_y.columns, index=test_just_y.index)
#print(df4)
data1 = pd.read_csv('perfect_data_1.csv',index_col =0)
cam_rel_y = full_library2.camera_relative (df5,perf_data, 'midx1', 'midy1')
#print(cam_rel_y)

data = [cam_rel_x["midx1"], cam_rel_x["midx2"],cam_rel_x["midx3"], cam_rel_x["midx4"],cam_rel_x["midx5"], cam_rel_x["midx6"],cam_rel_x["midx7"],cam_rel_y["midy1"], cam_rel_y["midy2"],cam_rel_y["midy3"], cam_rel_y["midy4"],cam_rel_y["midy5"], cam_rel_y["midy6"],cam_rel_y["midy7"]]
headers = ["midx1", "midx2","midx3", "midx4","midx5", "midx6","midx7","midy1", "midy2","midy3", "midy4","midy5", "midy6","midy7"]
done = pd.concat(data, axis=1, keys=headers)


#anom_data = pd.read_csv('anom_data_1.csv')

fig, ax = plt.subplots(figsize=(10,6))

#ax.plot(anom_data['midy2'], color='green', label = 'Normal')
ax.plot(done['midy7'], color='black', label = 'Normal')
ax.plot(anom_data['midy7'], color='red', label = 'Normal')
#ax.plot(anom['midy1'], color='green', label = 'Normal')
plt.show();
