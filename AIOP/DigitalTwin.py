import numpy as np
import tensorflow as tf
import pandas as pd 
import json
import os
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense,GRU,Activation
from tensorflow.python.keras.optimizer_v2.adam import Adam
from AIOP.DT_Validate import DT_Validate

#cuDnn bug fix
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

class DigitalTwin(object):
	def __init__(self,dt_modelname,data_dir,num_val_sets,
				independantVars=None,dependantVar=None,
				dt_lookback=20,velocity=False,scanrate=1):

		with open('timestamps.json', 'r') as savefile:
			self.data_dict = json.load(savefile)
		self.data_dir = data_dir
		self.num_val_sets = num_val_sets
		self.dt_modelname = dt_modelname
		self.dependantVar = dependantVar
		self.independantVars = independantVars
		self.dt_lookback = dt_lookback
		self.velocity = velocity
		self.scanrate=scanrate

		self.TrainPreprocess()
		self.TrainTestSplit()


	def TrainPreprocess (self):
		self.valdata = np.random.choice([i for i in self.data_dict],self.num_val_sets)
		
		#using python lists because np.append takes forever...
		self.targets = []
		self.variables = []
		total = 0
		for record in self.data_dict:
			if record in self.valdata:
				pass
			else:
				data = pd.read_csv(self.data_dir + self.data_dict[record]['file']).iloc[::self.scanrate,:]
				data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])
				
				#get time ranges
				data = data[data['TimeStamp']>self.data_dict[record]['xmin']]
				data = data[data['TimeStamp']<self.data_dict[record]['xmax']]

				data['TimeStamp'] = 0
				data = np.asarray(data).astype('float32')

				# data = np.asarray(data)
				new_records = data.shape[0]-self.dt_lookback-1
				total += new_records
				print(record + ' new samples '+ str(new_records) + ' total  ' + str(total))

				var_data = data[:,self.independantVars]
				target_data = data[:,[self.dependantVar]].reshape(data.shape[0])

				for t in range(new_records - self.dt_lookback-1):
					self.variables.append(var_data[t:t+self.dt_lookback,:])
					
					if self.velocity:
						self.targets.append(target_data[t+self.dt_lookback]-target_data[t-1+self.dt_lookback])
					else:
						self.targets.append(target_data[t+self.dt_lookback])
					
		
		#convert to np array
		self.targets = np.asarray(self.targets)
		self.variables = np.asarray(self.variables)

		self.targetmin = self.targets.min()
		self.targetmax = self.targets.max()

	def TrainTestSplit (self):
		#calculate seperate 80/20 train test val splits
		nx = self.variables.shape[0]
		trainset = int(round(.8*nx,0))

		#Shuffle the Data and create train test dataframes
		perm = np.random.permutation(nx)
		self.x_train = self.variables[perm[0:trainset]]
		self.y_train = self.targets[perm[0:trainset]]
		self.x_test  = self.variables[perm[trainset:nx]]
		self.y_test = self.targets[perm[trainset:nx]]

	def trainDt(self,gru1_dims=64,gru2_dims=64,lr=.01,ep=500,batch_size = 1000):

		# clear previous model just in case
		tf.keras.backend.clear_session()

		self.model = Sequential([
			GRU(gru1_dims, input_shape=(self.x_train.shape[1],
				self.x_train.shape[2]),return_sequences=True), 
			Activation('linear'),
			GRU(gru2_dims),
			Activation('linear'),
			Dense(1),
			Activation('linear')])

		self.model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
		print(self.model.summary())
		self.model.fit(self.x_train, self.y_train, 
          				batch_size=batch_size, epochs=ep, verbose=1, 
						validation_data = (self.x_test,self.y_test))
		
		self.saveDt()
		print('Model Saved')
		# self.validate()
		

	def saveDt(self):
		#create a experiment folder to save model to
		rand_num = str(int(round(np.random.rand()*10000,0)))
		self.dt_dir = self.dt_modelname + rand_num +'/'
		print(self.dt_dir)
		os.makedirs(self.dt_dir)

		# Convert the model.
		converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
		converter.target_spec.supported_ops = [
  			tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  			tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
		]
		
		tflite_model = converter.convert()

		# Save the model.
		with open(self.dt_dir + 'DT.tflite', 'wb') as f:
			f.write(tflite_model)

		#create a config file 
		config = {
					'dependantVar':self.dependantVar,
					'independantVars':self.independantVars,
					'dt_lookback': self.dt_lookback,
					'velocity':self.velocity,
					'targetmin':float(self.targetmin),
					'targetmax':float(self.targetmax),
					'scanrate': self.scanrate
					 }
		with open(self.dt_dir +'config.json', 'w') as outfile:
			json.dump(config, outfile, indent=4)

	def validate(self,max_len):	
		for record in self.valdata:
			#initalize validation simulator
			self.val = DT_Validate(dt_dir=self.dt_dir,data_dict=self.data_dict,
						data_dir=self.data_dir,record=record,max_len=max_len)

			#initalize validation loop
			done = self.val.reset()
			
			#execute the episode
			while not done:
				done = self.val.step()

			self.val.plot_eps()


