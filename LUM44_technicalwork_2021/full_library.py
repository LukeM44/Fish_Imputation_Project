#Collective anomaly code
#By Luke Mullis

#This is a series of funcitons that can be used to test the success of anomaly detection and imputation code
#for time series data.

#complete

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statistics
import math
import random

#This function adds anomalies to a dataset

#parameters:
#attribute: String , which attribute to add anomlaies to
#data: Dataframe , The data having anomalies added to
#max_anom: int , upper limit of the random number generator
#min_anom: int , lower limit of the random number generator
#chance_of_anom: float , chance of an anomaly occurung for each data point
#min_size: int , minimum anomaly size

#returns: a dataframe with one attribute with added values and a dataframe tracking anomaly location and size.


def collective_anomaly_adder (attribute,data,max_anom, min_anom, chance_of_anom,min_size):
	fig, ax = plt.subplots(figsize=(10,6))
	ax.plot(data[attribute], color='red', label = 'Normal')

	plt.show();

	df2 = pd.DataFrame(0, index=range(len(data.index)), columns=range(1))
	temp = []
	temp = data[attribute]
	anom_track = []
	for i in data.index:
		chance = random.random()
		if chance_of_anom> chance:
			new_value = round(random.uniform(min_anom,max_anom),4)
			if new_value > 0:
				data.at[i, attribute] = data[attribute][i] + (new_value+min_size)
				df2.at[i] = (new_value + min_size)
			elif new_value< 0:
				data.at[i, attribute] = data[attribute][i] + (new_value-min_size)
				df2.at[i] = (new_value - min_size)	
			
			
	fig, ax = plt.subplots(figsize=(10,6))
	ax.plot(data[attribute], color='red', label = 'Normal')

	plt.show();	
	df2.columns=['anomay']	
	return [data,df2]	

# This function tests the sucess of a collective anomaly detection algorithm
#df: dataframe , containing the anomaly detected data
#attribute: which attribute is being tested
#anom_list: dataframe , tracking the anomaly location (from anom adder) 

#return : nothing

def collective_anomally_test(df, attribute,anom_list):
	#print(anom_list[0])
	#print(anom_list)
	tp =0
	fp =0
	missed = 0
	for i in df.index:
		
		if (df['anomaly2'][i] == -1) & (anom_list['anomay'][i] !=0):
			tp +=1
			print('success: ', i,' : ',anom_list['anomay'][i])
		elif (df['anomaly2'][i] == -1) & (anom_list['anomay'][i] ==0):
			fp +=1
			print('fail: ', i,' : ',anom_list['anomay'][i])
		elif (df['anomaly2'][i] == 1) & (anom_list['anomay'][i] !=0):
			missed +=1
			print('missed: ', i,' : ',anom_list['anomay'][i])	
	print('tp : ',tp)
	print('fp : ',fp)
	print('missed : ',missed)
	#print('percent : ', (tp/(tp+fp+missed)))


#function that detects collective anomalies

#data: dataframe, data being worked on
#attribute: string , attribute being worked on
#diffrence_needed: int, the threshold for what is considered an anomaly.

#returns: dataframe , with anomalies identified.

def collective_anomaly_detection(data,attribute, diffrence_needed):
	data_columns = [attribute]

	data['anomaly2'] = 1
		
	temp = data.index
	for ind in temp:
		last_norm = 1
		if ind ==0:
			ind +=1
		while data['anomaly2'][ind-last_norm] == -1:
			last_norm += 1
			
			
		luke = ind-last_norm
		if abs(data[attribute][ind] - data[attribute][luke]) >diffrence_needed:
			data.at[ind, 'anomaly2'] =-1
		
    

	fig, ax = plt.subplots(figsize=(10,6))

	b = data.loc[data['anomaly2'] == -1, [attribute]]
	#print(b)
	ax.plot(data.index,data[attribute], color='blue', label = 'Normal')
	ax.scatter(b.index,b[attribute], color='red', label = 'Anomaly')
	plt.legend()

	ax.legend()
	ax.set_xlabel('time')
	ax.set_ylabel(attribute)
	ax.set_title('original data')
	plt.show();
	return data

#code imputes new data for data points identified as anomalies

#data: dataframe , being worked on
#attribute:	string , attribute being worked on

#returns: dataframe , with imputed data
def collective_impute_new_data(data,attribute):	
	for ind in data.index: 
		if data['anomaly2'][ind] == -1:
			sub_anom = 0
			temp_location = ind
			while data['anomaly2'][round(temp_location,2)] == -1:
				temp_location +=1
				sub_anom +=1
			diffrence = (data[attribute][ind + sub_anom]) - (data[attribute][ind-1])
			amount = 1/sub_anom
			for i in range(sub_anom):
				data.at[ind+i, attribute] = (data[attribute][ind-1]+ (amount * diffrence* (i+1)))            
				data.at[ind+i, 'anomaly2'] = 1
				
			

	fig, ax = plt.subplots(figsize=(10,6))
	ax.set_title('imputed data')
	ax.plot(data[attribute], color='blue', label = 'Normal')
	plt.legend()
	ax.set_xlabel('time')
	ax.set_ylabel(attribute)
	plt.show();	
	return data

#compares two dataframes to measure the sucess of an imputation algorithm

#df1: dataframe , data with imputed values
#df2 dataframe , perfect dataset
#attribute: string , which attribute is being compared

#returns: nothing
def collective_imputation_success_test(df1,df2,attribute):
	total_diffrence = 0
	for i in df1.index: 
		
		total_diffrence += abs(df2[attribute][i] - df1[attribute][i])
			
	print(total_diffrence)	
