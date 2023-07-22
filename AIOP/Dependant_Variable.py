import numpy as np
import tensorflow as tf
import pandas as pd 
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense,Activation,Flatten
from tensorflow.python.keras.optimizer_v2.adam import Adam

#cuDnn bug fix
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

class Dependant_Variable(object):
	def __init__(self,modelname,files,dVindex,dVInputs,lookback=5,Default=True,UseExisting=False):
		self.modelname = modelname
		self.files = files
		self.dVindex = dVindex
		self.lookback = lookback
		self.dVInputs = dVInputs
		if UseExisting:
			self.loadmodel()
			Default = False
		else:
			self.TrainPreprocess()
			self.TrainTestSplit()
		#Train and save model if Default is True
		if Default:
			self.TraindVar()
			self.SaveEnv()

	
	def loadmodel(self):
		self.model = load_model(self.modelname +'.h5')

	def TrainPreprocess (self):
		#calculate how big the sum of datasets are
		nsamples = 0
		for file in self.files:
			data = pd.read_csv(file)
			nsamples += data.shape[0] - self.lookback - 2
			
		#initalize arrays
		self.variables = np.zeros((nsamples,self.lookback,len(self.dVInputs)), dtype = 'float32')
		self.targets = np.zeros((nsamples,1),dtype = 'float32')
		
		#take data from csv into arrays
		#initalize counter
		self.count = 0
		for file in self.files:
			data = np.asarray(pd.read_csv(file))
			variable_data = data[:,self.dVInputs]
			for t in range(data.shape[0]-self.lookback-2):
				self.targets[self.count] = data[t+self.lookback+1,self.dVindex]
				self.variables[self.count] = variable_data[t:t+self.lookback,:]
				self.count +=1
		
	def TrainTestSplit (self):
		#calculate seperate 70/20/10 train test val splits
		nx = self.variables.shape[0]
		trainset = int(round(.7*nx,0))
		testset = int(round(.2*nx,0))
		valset = trainset + testset

		#Shuffle the Data and create train test dataframes
		perm = np.random.permutation(nx)
		self.x_train = self.variables[perm[0:trainset]]
		self.y_train = self.targets[perm[0:trainset]]
		self.x_test  = self.variables[perm[trainset:valset]]
		self.y_test = self.targets[perm[trainset:valset]]
		self.x_val  = self.variables[perm[valset:nx]]
		self.y_val = self.targets[perm[valset:nx]]


	def TraindVar(self,fc1_dims=32,fc2_dims=32,lr=.001,ep=500):

		# clear previous model just in case
		tf.keras.backend.clear_session()

		self.model = Sequential([
			Dense(fc1_dims, input_shape=(self.x_train.shape[1],self.x_train.shape[2])), #,return_sequences=True
			Activation('tanh'),
			Flatten(),
			Dense(fc2_dims),
			Activation('relu'),
			Dense(1),
			Activation('linear')])

		self.model.compile(optimizer=Adam(lr=lr), loss='mse')
		print(self.model.summary())
		self.model.fit(self.x_train, self.y_train, 
          				batch_size=500, epochs=ep, verbose=1, 
						validation_data = (self.x_test,self.y_test))
		
		score = self.model.evaluate(self.x_val, self.y_val, verbose=0)
		self.SaveEnv()
		print('Model Saved.  Valadation mean squared error ',score)
		

	def SaveEnv(self):
		#create unique string to save model to working directory
		self.model.save(self.modelname +'.h5')
