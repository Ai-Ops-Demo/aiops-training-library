import numpy as np
import pandas as pd
import os
import math
import json
import datetime


#open and normalize files.
directory= './data/raw_data/'
savedirectory = './data/norm_data/'
setpoints = []

if not os.path.exists(savedirectory):
	os.mkdir(savedirectory)
files = os.listdir(directory)

#get column names in the first file so we can compare them to other files for discrepancy
try:
	data = pd.read_csv(directory+files[0])
except:
	data = pd.read_csv(directory+files[0],encoding='utf-16')
columns = list(data.columns)

#import the min and max values file
with open('min_max.json', 'r') as norm_file:
	tag_dict = json.load(norm_file)

for file in files:
	#open the file
	print('normalizing ' + file)
	try:
		data = pd.read_csv(directory+file)
	except:
		data = pd.read_csv(directory+file,encoding='utf-16')

	data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])
	
	#normalize it
	for index in tag_dict:
		tag = list(tag_dict[index].keys())[0]
		if 'TimeStamp' in tag:
			pass
		else:
			data[tag] = (data[tag] - tag_dict[index][tag][1]) / \
				(tag_dict[index][tag][0]-tag_dict[index][tag][1])

	for pv in setpoints:
		#add setpoint
		data[pv.replace('PV','SV')] = pd.DataFrame(np.zeros(data.shape[0])\
			+data[pv].mean())

	#save data
	data.to_csv(savedirectory+file,sep=',',index=False)


#Add Setpoints to JSON
last_tag = list(tag_dict.keys())[-1]
for pv in setpoints:
	for tag in list(tag_dict.keys()):
		if pv in tag_dict[tag].keys():
			last_tag = str(int(last_tag)+1)
			tag_dict[last_tag]  = {pv.replace('PV','SV'):\
				[tag_dict[tag][pv][0],
				tag_dict[tag][pv][1]]}

	
with open('norm_vals.json', 'w') as outfile:
	json.dump(tag_dict, outfile,indent=4)



