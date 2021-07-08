#Contexual anomaly code
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
#data: Dataframe , The data having anomalies added to
#max_anom: int , upper limit of the random number generator
#min_anom: int , lower limit of the random number generator
#chance_of_anom: float , chance of an anomaly occurung for each data point
#min_size: int , minimum anomaly size

#returns: a dataframe with one attribute with added values and a dataframe tracking anomaly location and size.

def contexual_anomaly_adder (df1,max_anom, min_anom, chance_of_anom,min_size):
	#fig, ax = plt.subplots(figsize=(10,6))
	#ax.plot(df1['midx6'], color='red', label = 'Normal')

	#plt.show();
	
	df2 = pd.DataFrame(data=None, columns=df1.columns, index=df1.index)
	columns = list(df1)
	#columns.pop(1)
	#columns.pop(31)
	#columns.pop(0)
	columns.pop(0)
	columns.pop(29)
	#print(columns)
	column = 0
	
	for index in df1.index:
		for attribute in (columns):
			chance = random.random()
			if (attribute != 'midx1') or (attribute != 'midy1'):
				if chance_of_anom> chance:				
					new_value = round(random.uniform(min_anom,max_anom),2)
					if new_value > 0:
						df1.at[index, attribute] = df1[attribute][index] + (new_value + min_size)
						df2.at[index, attribute] = (new_value + min_size)
					elif new_value< 0:	
						df1.at[index, attribute] = df1[attribute][index] + (new_value - min_size)
						df2.at[index, attribute] = (new_value - min_size)
	#fig, ax = plt.subplots(figsize=(10,6))
	#ax.plot(df1['midx6'], color='red', label = 'Normal')

	#plt.show();	
	return [df1,df2]	


#used to find if an anomaly is a x of y value

#parameters:
#name: string , attribute name

#returns: string , 'x' or 'y' depending on the attribute type
def xory(name):
	if 'x' in name:
		return 'x'
	elif 'y' in name:
		return 'y'
	else:
		return 'error'

#turns the data relative to the fish snout

#df1: dataframe , data being worked on
#xpoint: string , x point all other x points will become relative too. should be midx1
#ypoint: string , y point all other y points will become relative too. should be midy1

#returns: dataframe , relative to the fish snout.

def fish_relative (df1, xpoint, ypoint):
	df2 = pd.DataFrame(data=None, columns=df1.columns, index=df1.index)
	columns = list(df1)
	for index in df1.index:
		for attribute in (columns):
			if xory(attribute)=='x':
				df2.at[index,attribute] = df1[attribute][index] - df1[xpoint][index]
			elif xory(attribute)=='y':
				df2.at[index,attribute] = df1[attribute][index] - df1[ypoint][index]	
	#print(df2)
	return(df2)

#turns the data relative to the fish relative to the camera

#df1: dataframe , data being worked on
#original: dataframe, original dataframe, before any manipulation
#xpoint: string , x point all other x points are currently relative too. should be midx1
#ypoint: string , y point all other y points are currently relative too. should be midy1

#returns: dataframe , relative to the fish snout.


def camera_relative (df1,original, xpoint, ypoint):
	df2 = pd.DataFrame(data=None, columns=df1.columns, index=df1.index)
	columns = list(df1)
	for index in df1.index:
		for attribute in (columns):
			if xory(attribute)=='x':
				df2.at[index,attribute] = df1[attribute][index] + original[xpoint][index]
			elif xory(attribute)=='y':
				df2.at[index,attribute] = df1[attribute][index] + original[ypoint][index]	
	#print(df2)
	return(df2)

# This function tests the sucess of a contexual anomaly detection algorithm
#df1: dataframe , containing the anomaly detected data
#df2: dataframe , tracking the anomaly location (from anom adder) 

#return : nothing

					
def contexual_anomaly_test (df1,df2):
	#print(df2['midx1'][0.008])
	#print(df2)
	columns = list(df1)
	columns.pop(0)
	columns.pop(6)
	tp = 0
	fp = 0
	missed = 0
	collumn_tp = 0
	collumn_fp = 0
	collumn_missed = 0
	for attribute in (columns):
		for index in df1.index:
			#print(df1[attribute][index])
			if not(pd.isnull(df2[attribute][index])) and pd.isnull(df1[attribute][index]):
				collumn_tp +=1
				tp+=1
				print('success: attribute:',attribute,' row:  ,',index,' anom_val: ',df2[attribute][index])
			elif pd.isnull(df2[attribute][index]) and pd.isnull(df1[attribute][index]):
				collumn_fp +=1
				fp+=1
				print('fail: attribute:',attribute,' row:  ,',index,' anom_val: ',df2[attribute][index])
			elif not(pd.isnull(df2[attribute][index])) and not(pd.isnull(df1[attribute][index])):
				collumn_missed +=1
				missed+=1
				print('missed: attribute:',attribute,' row:  ,',index,' anom_val: ',df2[attribute][index])	
		print(attribute)		
		print('collumn_tp : ',collumn_tp)
		print('collumn_fp : ',collumn_fp)
		print('collumn_missed : ',collumn_missed)
		print('percent : ', (collumn_tp/(collumn_tp+collumn_fp+collumn_missed)))
		collumn_tp = 0
		collumn_fp = 0
		collumn_missed= 0
	print('overall:')		
	print('overall tp : ',tp)
	print('overall fp ',attribute,' : ',fp)
	print('overall missed : ',missed)
	print('overall percent : ', (tp/(tp+fp+missed)))
	

#compares two dataframes to measure the sucess of an imputation algorithm

#df1: dataframe , data with imputed values
#df2 dataframe , perfect dataset
#attribute: string , which attribute is being compared

#returns: nothing

def contexual_anomaly_imputation_test (df1,df2):
	collumn_diffrence = 0
	total_diffrence = 0
	columns = list(df1)
	for attribute in (columns):
		for index in df1.index:
			
			
			collumn_diffrence += abs(df1[attribute][index] - df2[attribute][index])	
			#print(collumn_diffrence)
		total_diffrence +=	collumn_diffrence
		print(attribute,' total diffrence: ',round(collumn_diffrence, 4))	
		collumn_diffrence = 0	
	print('total diffrence: ')
	print(round(total_diffrence, 4))	



#function that detects collective anomalies

#data: dataframe, data being worked on
#attribute: string , attribute being worked on
#diffrence_needed: int, the threshold for what is considered an anomaly.

#returns: dataframe , with anomalies identified.

def identify_anomalies (df1,df2,threshold):
	df3 = pd.DataFrame(data=None, columns=df1.columns, index=df1.index)
	columns = list(df1)
	for index in df1.index:
		for attribute in (columns):
			if abs(df1[attribute][index]-df2[attribute][index]) > threshold:
				df3.at[index,attribute] = (df2[attribute][index]-df1[attribute][index])
	return (df3)		

#makes all anomalies nan

#parmaters:
#df1: dataframe , being worked on
#anom:	dataframe , tracking anomaly locations made by 'contexual_anomaly_adder'

#returns: dataframe , with all anomaly values set to nan
def make_all_anom_nan (df1,anom):
	columns = list(df1)
	columns.pop(0)
	columns.pop(6)
	print(columns)
	for attribute in (columns):
		for index in df1.index:
			if not pd.isnull(anom[attribute][index]):
				df1.at[index,attribute] = np.nan
	return(df1)			


#code to detect contexual anoamies

#parmaters:
#df1: data frame being performed on
#y2: int , distance that this attribute can stray from 0 and not be an anomaly
#y3: int , distance that this attribute can stray from 0 and not be an anomaly
#y4: int , distance that this attribute can stray from 0 and not be an anomaly
#y5: int , distance that this attribute can stray from 0 and not be an anomaly
#y6: int , distance that this attribute can stray from 0 and not be an anomaly
#y7: int , distance that this attribute can stray from 0 and not be an anomaly
#x: int , distance that all x values can stray from 0 and not be an anomaly

#returns:
def contexual_anomaly_detection (df1,y2,y3,y4,y5,y6,y7,x):
	columns = list(df1)
	for attribute in (columns):
		if xory(attribute)=='x':
			total = 0
			count = 0
			average = 0
			min_val = 0
			max_val = 0
			for index in df1.index:
				total = total + df1[attribute][index]
				count +=1
			average = total / count
			min_val = average - x
			max_val = average +x
			for index in df1.index:
				if 	df1[attribute][index]<min_val:
					
					df1.at[index,attribute] = np.nan
				elif df1[attribute][index]>max_val:	
					
					df1.at[index,attribute] = np.nan
					
	if xory(attribute)=='y':
		for index in df1.index:
			if 	(df1['midy2'][index]<-y2) or (df1['midy2'][index]> (df1['midy1'][index] + y2)):
				df1.at[index,'midy2'] = np.nan
			if 	(df1['midy3'][index]< -y3) or (df1['midy3'][index]> (df1['midy1'][index] + y3)):
				df1.at[index,'midy3'] = np.nan
			if 	(df1['midy4'][index]< -y4) or (df1['midy4'][index]> (df1['midy1'][index] + y4)):
				df1.at[index,'midy4'] = np.nan
			if 	(df1['midy5'][index]< -y5) or (df1['midy5'][index]> (df1['midy1'][index] + y5)):
				df1.at[index,'midy5'] = np.nan
			if 	(df1['midy6'][index]< - y6) or (df1['midy6'][index]> (df1['midy1'][index] + y6)):
				df1.at[index,'midy6'] = np.nan
			if 	(df1['midy7'][index]< -y7) or (df1['midy7'][index]> (df1['midy1'][index] + y7)):
				df1.at[index,'midy7'] = np.nan
	print(df1)			
	return (df1)
	
# imputes new datausing the collective imputation method
#parmaters:
#data: dataframe, algorithm is run on
#returns: dataframe with imputed data	
def collective_imputation (data):
	columns = list(data)
	for attribute in (columns):
		if pd.isnull(data[attribute][1]):
			data.at[1,attribute] = data[attribute][2]
		if pd.isnull(data[attribute][0]):
			data.at[0,attribute] = data[attribute][1]
			
	for attribute in (columns):
		for ind in data.index: 
			if pd.isnull(data[attribute][ind]):
				sub_anom = 0
				temp_location = ind
				while pd.isnull(data[attribute][round(temp_location,2)]):
					temp_location +=1
					sub_anom +=1
				diffrence = (data[attribute][ind + sub_anom]) - (data[attribute][ind-1])
				amount = 1/sub_anom
				for i in range(sub_anom):
					data.at[ind+i, attribute] = (data[attribute][ind-1]+ (amount * diffrence* (i+1)))            
						
	return data
	
#collective anomaly detection function
#parmaters:
#data: dataframe, algorithm is run on
#returns: dataframe with anomalies replaced with 'nan'	
def collective_anomaly_detection(data,attribute, diffrence_needed):					
	data_columns = [attribute]
		
	temp = data.index
	for ind in temp:
		last_norm = 1
		if ind ==0:
			ind +=1
		while pd.isnull(data[attribute][ind-last_norm]):
			last_norm += 1
			#print('luke')
			
			
		luke = ind-last_norm
		#print(abs(data[attribute][ind] - data[attribute][luke]))
		if abs(data[attribute][ind] - data[attribute][luke]) >diffrence_needed:
			data.at[ind, attribute] =np.nan
			
	fig, ax = plt.subplots(figsize=(10,6))
	ax.plot(data.index,data[attribute], color='blue', label = 'Normal')
	plt.legend()

	ax.legend()
	ax.set_xlabel('time')
	ax.set_ylabel(attribute)
	ax.set_title('original data')
	plt.show();
	return data

#normalises the data to the length of the fish
#parmaters:	
#df1: dataframe , deeplabcut data
#df2: dataframe , perfect human tracked data
#returns: two normalised data sets
def normalise_data (df1,df2):
	biggest_def = 0 
	for ind in df1.index:
		if math.sqrt((df1['midx7'][ind] - df1['midx1'][ind])**2 + (df1['midy7'][ind] - df1['midy1'][ind])**2) > biggest_def:
			biggest_def = (df1['midx7'][ind] - df1['midx1'][ind])
	#print(biggest_def)
	
	columns = list(df1)
	for attribute in (columns):
		for ind in df1.index: 
			df1.at[ind, attribute] = (df1[attribute][ind]/biggest_def)
			df2.at[ind, attribute] = (df2[attribute][ind]/biggest_def)


#names 'nan' values consistant between two identically sized dataframes
#parmaters:	
#df1: dataframe , deeplabcut data
#df2: dataframe , perfect human tracked data
#returns: two dataframes with all 'nan' values in df1 now 'nan' values in df2	
def make_nan_cosistant (df1, df2):
	columns = list(df1)
	for ind in df1.index: 
		for attribute in (columns):
			if 	pd.isnull(df1[attribute][ind]):
				df2.at[ind, attribute] = np.nan
#reorders the attribute order of a dataframe
#parmaters:	
#df1: dataframe being worked on
#returns: dataframe with attributes in new order		
def reorder (df1):
	new_data = [df1["midx1"], df1["midx2"],df1["midx3"],df1["midx4"], df1["midx5"],df1["midx6"],df1["midx7"],df1["midy1"], df1["midy2"],df1["midy3"],df1["midy4"], df1["midy5"],df1["midy6"],df1["midy7"]]
	new_collumns = ["midx1", "midx2","midx3", "midx4","midx5", "midx6","midx7","midy1", "midy2","midy3", "midy4","midy5", "midy6","midy7"]
	done = pd.concat(new_data, axis=1, keys=new_collumns)
	
	return done
	
	
	
	
	
	
	
	
	
	
	
	
	
