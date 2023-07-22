import numpy as np
import pandas as pd 
import json
import matplotlib.pyplot as plt
# from tensorflow.python.keras.models import load_model
import tensorflow as tf

#cuDnn bug fix
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

class DT_Validate(object):
	def __init__(self,dt_dir,data_dict,data_dir,record,max_len):
		
		self.dt_dir = dt_dir
		with open(dt_dir+'config.json') as json_file:
			config = json.load(json_file)
		self.independantVars = config['independantVars']
		self.dependantVar = config['dependantVar']
		self.dt_lookback = config['dt_lookback']
		self.velocity = config['velocity']
		self.targetmin = config['targetmin']
		self.targetmax = config['targetmax']
		self.scanrate = config['scanrate']
		self.data_dict = data_dict
		self.record=record
		self.max_len = max_len
		self.data_dir = data_dir
		self.LoadDT()
		

	def LoadDT(self):	
		# Load the TFLite model and allocate tensors.
		self.interpreter = tf.lite.Interpreter(model_path=self.dt_dir +'DT.tflite')
		self.interpreter.allocate_tensors()

		# Get input and output tensors.
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()



	def reset(self):
		#load data 
		data = pd.read_csv(self.data_dir + self.data_dict[self.record]['file']).iloc[::self.scanrate,:]
		#get rid of the datetime columns because it causes errors later
		data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])
				
		#get time ranges
		data = data[data['TimeStamp']>self.data_dict[self.record]['xmin']]
		data = data[data['TimeStamp']<self.data_dict[self.record]['xmax']]
		data['TimeStamp'] = 0
		
		self.episode_data = np.asarray(data)

		nsamples = self.episode_data.shape[0]
		if nsamples > self.max_len:
			randstart=int(np.random.rand()*(nsamples - self.max_len))
			self.episode_data = self.episode_data[randstart:randstart+self.max_len]
			
		self.len_data = self.episode_data.shape[0]
		
		#make an empty array to start the episode
		self.episode_array = np.zeros((self.episode_data.shape[0],self.episode_data.shape[1]),dtype='float32')

		#fill it with enough data to make first line
		self.episode_array[0:self.dt_lookback] = self.episode_data[0:self.dt_lookback]

		#initalize a counter to keep track of the episode
		self.transition_count = self.dt_lookback

		#inatilize a done flag to end the episode
		self.done = False

		return self.done

	def step(self):
		#copy episode data to the episode array
		self.episode_array[self.transition_count] = self.episode_data[self.transition_count]

		#predict the PV
		dt_inputs = self.episode_array[self.transition_count-self.dt_lookback:self.transition_count,self.independantVars]\
			.reshape(1,self.dt_lookback,len(self.independantVars))

		self.interpreter.set_tensor(self.input_details[0]['index'], dt_inputs)
		self.interpreter.invoke()
		pv_ = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]

		
		if self.velocity:
			pv = self.episode_array[self.transition_count-1,self.dependantVar]
			pv_ = pv + pv_
		
			
		#overwrite the PV
		self.episode_array[self.transition_count,self.dependantVar] = np.clip(pv_,0,1)

		#advance counter		
		self.transition_count +=1

		if self.transition_count > self.len_data-1:
			self.done=True

		return self.done

	def plot_eps(self):
		self.stats()
		#plot validation response
		plt.plot(self.episode_array[:,self.dependantVar], label = 'AiPV')
		plt.plot(self.episode_data[:,self.dependantVar], label = 'PV')
		plt.legend(loc='lower right', shadow=True, fontsize='large')
		plt.xlabel('time (s)')
		plt.title(label = 'Response Mean Absolute Error %' + str(np.round(self.mean_error*100,2)), loc='center')
		plt.savefig(self.dt_dir + str(int(np.random.rand()*10000)) + '.png')
		plt.clf()

	def stats(self):
		#get error sum
		self.sum_error= 0
		self.max_error = 0
		for i in range(self.dt_lookback,self.len_data-2):
			error = abs(self.episode_array[i,self.dependantVar]-self.episode_data[i,self.dependantVar])
			self.sum_error += error
			if error >self.max_error:
				self.max_error = error

		self.mean_error = self.sum_error/(self.len_data-self.dt_lookback-1)
	

		print('Environment Mean Absolute Error %',np.round(self.mean_error*100,2))
		print('Environment Max Error %',np.round(self.max_error*100,2))
