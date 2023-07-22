import numpy as np
import tensorflow as tf
import pandas as pd 
import random
import json
import os
from tensorflow.python.keras.models import load_model

# silence tf warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


class Simulator(object):
	def __init__(self,data_dir,agentIndex,MVindex,SVindex,PVindex,
				agent_lookback=5,dt1=None,episode_length=240,SVnoise=0.1,stability=.5,
				stability_tolerance=0.01,response=.5,response_tolerance=.005,
				general=.5):

		with open('timestamps.json', 'r') as savefile:
			self.data_dict = json.load(savefile)

		self.agentIndex = agentIndex
		self.dt1 = dt1
		self.data_dir = data_dir
		self.MVindex = MVindex
		self.SVindex = SVindex
		self.PVindex=PVindex
		self.SVnoise = SVnoise
		self.episode_length = episode_length
		self.agent_lookback = agent_lookback
		self.response_tolerance = response_tolerance
		self.stability = stability
		self.response = response
		self.general = general
		self.stability_tolerance = stability_tolerance
		self.get_min_max()
		
	def loadEnv(self):
		
		#load environment model
		dt1 = random.choice(self.dt1)

		# Load the TFLite model and allocate tensors.
		self.dt1_model = tf.lite.Interpreter(model_path= dt1 +'/DT.tflite')
		self.dt1_model.allocate_tensors()

		# Get input and output tensors.
		self.dt1_input_details = self.dt1_model.get_input_details()
		self.dt1_output_details = self.dt1_model.get_output_details()


		with open(dt1+'/config.json', 'r') as savefile:
			dt1_config = json.load(savefile)

		#load input index and lookback
		self.dt1_lookback = dt1_config['dt_lookback']
		self.dt1_independantVars = dt1_config['independantVars']
		self.dt1_dependantVar = dt1_config['dependantVar']
		self.dt1_velocity = dt1_config['velocity']
		self.dt1_targetmin = dt1_config['targetmin']
		self.dt1_targetmax = dt1_config['targetmax']
		self.dt1_scanrate = dt1_config['scanrate']

	
	def get_min_max(self):
		#find the max and min SV
		self.SV_max = 0
		self.SV_min = 1
		self.MV_max = 0
		self.MV_min = 1
		
		for record in self.data_dict:
			data = pd.read_csv(self.data_dir + self.data_dict[record]['file'])
			data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])
				
			#get time ranges
			data = data[data['TimeStamp']>self.data_dict[record]['xmin']]
			data = data[data['TimeStamp']<self.data_dict[record]['xmax']]

			data['TimeStamp'] = 0
			data = np.asarray(data).astype('float32')
			
			#get SV_max and SV_min
			maxSV = data[:,self.SVindex].max()
			minSV = data[:,self.SVindex].min()
			self.SV_max = max(self.SV_max,maxSV)
			self.SV_min = min(self.SV_min,minSV)

			#get MV_max and MV_min
			maxMV = data[:,self.MVindex].max()
			minMV = data[:,self.MVindex].min()
			self.MV_max = max(self.MV_max,maxMV)
			self.MV_min = min(self.MV_min,minMV)
	
	def get_data(self):
		record = random.choice(list(self.data_dict.keys()))
		data = pd.read_csv(self.data_dir + self.data_dict[record]['file']).iloc[::self.dt1_scanrate,:]
		data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])
				
		#get time ranges
		data = data[data['TimeStamp']>self.data_dict[record]['xmin']]
		data = data[data['TimeStamp']<self.data_dict[record]['xmax']]

		data['TimeStamp'] = 0
		data = np.asarray(data).astype('float32')
		return data
	
	def reset(self):
		self.loadEnv()
		#load data 
		data_needed = self.episode_length + self.dt1_lookback
		data = self.get_data()
		while data_needed > data.shape[0]:
			data = self.get_data()

		#get random data to generate an episode
		startline = random.choice(range(0,data.shape[0]-data_needed))
		endline = startline+data_needed
		self.episodedata = data[startline:endline]

		#add some noise to the SV so it starts in an odd place.
		if np.random.rand() < .5:
			noise = self.SVnoise*np.random.rand()
		else:
			noise = -self.SVnoise*np.random.rand()

		for row in range(self.dt1_lookback,data_needed):
			#append and clip data to make sure it doesnt exceed reality
			self.episodedata[row,self.SVindex] = np.clip(self.episodedata[row,self.SVindex]+noise,self.SV_min,self.SV_max)


		#get the first rows as the start state to return
		start_state = self.episodedata[self.dt1_lookback-self.agent_lookback:self.dt1_lookback,self.agentIndex]

		#make an empty array to start the episode
		self.episode_array = np.zeros((self.episodedata.shape),dtype='float32')

		#fill it with enough data to make first line
		self.episode_array[0:self.dt1_lookback] = self.episodedata[0:self.dt1_lookback]

		#initalize a counter to keep track of the episode
		self.transition_count = self.dt1_lookback

		#inatilize a done flag to end the episode
		self.done = False

		#create a variable to show where in the agent_state space the controller positioin is
		self.MVpos = self.agentIndex.index(self.MVindex)

		return start_state,self.done

	def step(self,action):

		#copy episode data to the episode array
		self.episode_array[self.transition_count] = self.episodedata[self.transition_count]

		#predict the PV
		dt1_inputs = self.episode_array[self.transition_count-self.dt1_lookback:self.transition_count,self.dt1_independantVars]\
			.reshape(1,self.dt1_lookback,len(self.dt1_independantVars))

		self.dt1_model.set_tensor(self.dt1_input_details[0]['index'], dt1_inputs)
		self.dt1_model.invoke()
		pv_ = self.dt1_model.get_tensor(self.dt1_output_details[0]['index'])[0][0]

		if self.dt1_velocity:
			pv = self.episode_array[self.transition_count-1,self.dt1_dependantVar]
			pv_ = pv + pv_
			
		#overwrite the PV
		self.episode_array[self.transition_count,self.dt1_dependantVar] = np.clip(pv_,0,1)

		#overwrite the MV with the agents action in the episode data into the future
		self.episode_array[self.transition_count,self.MVindex] = action

		#get the new state to return
		state_ = self.episode_array[self.transition_count-self.agent_lookback+1:self.transition_count+1,self.agentIndex]
		
		#calculate reward
		error = abs(pv_ - self.episode_array[self.transition_count,self.SVindex])
		reward = self.general-error*self.general
		oldMV = self.episode_array[self.transition_count-1,self.MVindex]

		if error < self.response_tolerance:
			reward += self.response
		
		if error < .5*self.response_tolerance:
			reward += self.response

		if error < .25*self.response_tolerance:
			reward += self.response
			
		if abs(action-oldMV) < self.stability_tolerance and error < self.response_tolerance:
			reward += self.stability
		
		#check if done
		if self.transition_count > self.episode_length + self.dt1_lookback-2:
			self.done = True

		#adVance counter
		self.transition_count +=1
		
		return state_,reward,self.done
		
