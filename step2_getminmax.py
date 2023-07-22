import numpy as np
import pandas as pd
import os
import math
import json


#open files and get min_max values.
directory= './data/raw_data/'

files = os.listdir(directory)

try:
	data = pd.read_csv(directory+files[0])
except:
	data = pd.read_csv(directory+files[0],encoding='utf-16')
	

#get max and min to save to json
columns = list(data.columns)
auto_normval = {}
for column in columns:
	auto_normval[column] = [data[column].max(),data[column].min()]

#get min:max values
for file in files:
	#open the file
	print('reading ' + file)
	try:
		data = pd.read_csv(directory+file)
	except:
		data = pd.read_csv(directory+file,encoding='utf-16')
	
	data = data.dropna()
	if list(data.columns)!= columns:
		print('failed - columns not the same')

	for column in columns:
		if data[column].max() > auto_normval[column][0]:
			auto_normval[column][0] = data[column].max()
		if data[column].min() < auto_normval[column][1]:
			auto_normval[column][1] = data[column].min()

#convert all values to float so that they can be saved to Json
for tag in auto_normval:
	if 'Time' in tag:
		pass
	else:
		auto_normval[tag][0] = math.ceil(float(auto_normval[tag][0]))

		if math.floor(float(auto_normval[tag][1])) >= 0:
			auto_normval[tag][1] = math.floor(float(auto_normval[tag][1]))
		else:
			auto_normval[tag][1] = 0

#number values 0-n
number_norm_val = {}
num = 0
for tag in auto_normval:
	number_norm_val[num] = {tag:auto_normval[tag]}
	num +=1	

with open('min_max.json', 'w') as outfile:
	json.dump(number_norm_val, outfile,indent=4)
