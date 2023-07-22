import numpy as np
import tensorflow as tf
import pandas as pd 
import random
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt
import json

#cuDnn bug fix
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

class Policy_Validate(object):
	def __init__(self,data_dir,agentIndex,MVindex,SVindex,PVindex,dt1=None,episode_length=240):

		with open('val.json', 'r') as savefile:
			self.data_dict = json.load(savefile)

		self.agentIndex = agentIndex
		self.dt1 = dt1
		self.data_dir = data_dir
		self.MVindex = MVindex
		self.SVindex = SVindex
		self.PVindex=PVindex
		self.episode_length = episode_length
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


	def reset(self):
		self.loadEnv()
		#load data 
		record = random.choice(list(self.data_dict.keys()))
		data = pd.read_csv(self.data_dir + self.data_dict[record]['file']).iloc[::self.dt1_scanrate,:]
		data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])
				
		#get time ranges
		data = data[data['TimeStamp']>self.data_dict[record]['xmin']]
		data = data[data['TimeStamp']<self.data_dict[record]['xmax']]

		data['TimeStamp'] = 0
		data = np.asarray(data).astype('float32')

		data_needed = self.episode_length + self.dt1_lookback

		#get random data to generate an episode
		startline = random.choice(range(0,data.shape[0]-data_needed))
		endline = startline+data_needed
		self.episodedata = data[startline:endline]

		#get the first rows as the start state to return
		start_state = self.episodedata[:self.dt1_lookback,self.agentIndex]

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
		state_ = self.episode_array[self.transition_count-self.dt1_lookback+1:self.transition_count+1,self.agentIndex]
		
		#check if done
		if self.transition_count > self.episode_length + self.dt1_lookback-2:
			self.done = True

		#adVance counter
		self.transition_count +=1
		
		return state_,self.done
		

	def plot_eps(self,model_dir):
		self.stats()
		#plot validation response
		plt.plot(self.episodedata[:,self.SVindex], label = 'SetPoint')
		plt.plot(self.episode_array[:,self.PVindex], label = 'AiPV')
		plt.plot(self.episodedata[:,self.PVindex], label = 'PV')
		plt.plot(self.episode_array[:,self.MVindex], label = 'AiMV')
		plt.plot(self.episodedata[:,self.MVindex], label = 'MV')
		plt.legend(loc='lower right', shadow=True, fontsize='small')
		plt.xlabel('time (s)')
		plt.title(label = ' error sum '+str(np.round(self.policy_error,2)), loc='center')
		plt.savefig(model_dir + str(int(np.random.rand()*100000))+'.png')
		plt.clf()

	def stats(self):
		#get error sum
		self.policy_error = 0
		self.PID_error = 0
		for i in range(self.dt1_lookback,self.episode_length-1):
			self.policy_error += abs(self.episode_array[i,self.SVindex]-self.episode_array[i,self.PVindex])
			self.PID_error += abs(self.episodedata[i,self.SVindex]-self.episodedata[i,self.PVindex])

		
		print('policy error ',self.policy_error, 'PID error ', self.PID_error)


