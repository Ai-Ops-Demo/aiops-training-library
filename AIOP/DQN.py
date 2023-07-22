from tensorflow.python.keras.layers import Dense, Activation,Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam
import numpy as np
import pandas as pd
import tensorflow as tf
from AIOP.PID import PID


class ReplayBuffer(object):
	def __init__(self,agentIndex,agent_lookback=1,capacity=1000000,batch_size=64):
		#mini batch size
		self.batch_size = batch_size
		#column names for the state variables
		self.agentIndex = agentIndex
		self.state_size = len(agentIndex)
		self.agent_lookback = agent_lookback
	
		#buffer for state, action, reward, state_
		self.capacity = capacity
		self.buffer_s = np.zeros((self.capacity,self.agent_lookback,self.state_size), dtype=np.float32)
		self.buffer_a = np.zeros((self.capacity), dtype=int)
		self.buffer_r = np.zeros(self.capacity)
		self.buffer_s_ = np.zeros((self.capacity,self.agent_lookback,self.state_size), dtype=np.float32)
		self.mem_counter = 0
	
	def storeTransition(self,state,action,reward,state_):
		# store transition
		index = self.mem_counter % self.capacity
		self.buffer_s[index] = state
		self.buffer_a[index] = action
		self.buffer_r[index] = reward
		self.buffer_s_[index] = state_
		self.mem_counter += 1
	
	def sampleMiniBatch(self):
		# sample random batch from transition
		sample = np.random.choice(min(self.mem_counter, self.capacity), min(self.mem_counter, self.batch_size),replace = False)
		sample = np.sort(sample)
		
		# store minibatch of transitions
		sample_s = self.buffer_s[sample]
		sample_a = self.buffer_a[sample]
		sample_r = self.buffer_r[sample]
		sample_s_ = self.buffer_s_[sample]
		
		return sample_s,sample_a,sample_r,sample_s_

	def get_min_max(self):
		#convert episode to pd.DataFrame
		if self.mem_counter > self.capacity:
			numrecords = self.capacity
		else:
			numrecords = self.mem_counter

		hist = np.zeros((numrecords,self.state_size),dtype=np.float32)
		for i in range(numrecords):
			hist[i] = self.buffer_s[i,0]
		episodes = pd.DataFrame(hist)
		episodes.columns = self.agentIndex
		return episodes.min(),episodes.max()

	
	def saveEpisodes(self,filename):
		#convert episode to pd.DataFrame
		if self.mem_counter > self.capacity:
			numrecords = self.capacity
		else:
			numrecords = self.mem_counter

		hist = np.zeros((numrecords,self.state_size),dtype=np.float32)
		for i in range(numrecords):
			hist[i] = self.buffer_s_[i,self.agent_lookback-1]
		episodes = pd.DataFrame(hist)
		episodes.columns = self.agentIndex
		episodes['action'] = self.buffer_a[:numrecords]
		episodes['reward'] = self.buffer_r[:numrecords]
		episodes.to_csv(filename,sep=',', header = True)


class Agent(object):
	def __init__(self,agentIndex,MVindex,agent_lookback=1,default_agent=True,
				gamma=.95,epsilon_decay=.99995,min_epsilon=.01,
				max_step=.05,training_scanrate = 1):
		#initalize dictionary of actions the agent can take
		self.agentIndex = agentIndex
		self.state_size = len(self.agentIndex)
		self.MVpos = agentIndex.index(MVindex)
		self.gamma = gamma
		self.epsilon = 1
		self.epsilon_decay = epsilon_decay
		self.min_epsilon = min_epsilon
		self.agent_lookback = agent_lookback
		self.fit_counter = 0 #counts how many times we have fit Q
		self.training_scanrate=training_scanrate
		
		#initalize vairable to know if there is a PID to learn from
		self.PIDcontroller = False

		#initalize and scale action dictionary
		self.action_dict = { 0: 0,
							1: 0.0156,
							2: -0.0156,
							3: 0.0625,
							4: -0.0625,
							5: 0.25,
							6: -0.25,
							7: 1,
							8: -1
										}

		for i in range(len(self.action_dict)):
			self.action_dict[i] *= max_step

		self.n_actions = len(self.action_dict)
		self.action_space = [i for i in range(self.n_actions)]
		self.action_val = np.asarray(list(self.action_dict.values()))
		
		if default_agent:
			self.buildQ()
			self.buildPolicy()
		
	def buildQ(self,fc1_dims=128,fc2_dims=128,lr=.001):

		# clear previous model just in case
		tf.keras.backend.clear_session()

		self.Q = Sequential([
			Dense(fc1_dims, input_shape=(self.agent_lookback,self.state_size,)),
			Activation('relu'),
			Flatten(),
			Dense(fc2_dims),
			Activation('relu'),
			Dense(self.n_actions),
			Activation('linear')])

		self.Q.compile(optimizer=Adam(learning_rate=lr), loss='mse')

		#initalize target Q
		self.Q_Target = self.Q
		
	#function for when we can directly compute the policy from the Q values
	def buildPolicy(self,fc1_dims=64,fc2_dims=64,lr=.001):

		# clear previous model just in case
		tf.keras.backend.clear_session()

		self.actor = Sequential([
			Dense(fc1_dims, input_shape=(self.agent_lookback,self.state_size,)),
			Activation('relu'),
			Flatten(),
			Dense(fc2_dims),
			Activation('relu'),
			Dense(1),
			Activation('sigmoid')])

		self.actor.compile(optimizer=Adam(learning_rate=lr), loss='mse')

	def selectAction(self,state):
		# with prob epsilon select random action from action_space or action from pretrainer
		rand = np.random.rand()
		if rand < self.epsilon:
			action = self.explore(state)
		else:
			action = np.argmax(self.Q.predict(np.asarray([state]),verbose=0))

		#decriment epsilon
		if self.epsilon > self.min_epsilon:
			self.epsilon *= self.epsilon_decay
		else:
			self.epsilon = self.min_epsilon
			
		return action

	def qlearn(self,sample_s,sample_a,sample_r,sample_s_):
		# learning
		self.Q_now = self.Q.predict(sample_s,verbose=0)
		self.Q_next = self.Q_Target.predict(sample_s_,verbose=0)

		# bellman update
		Y = self.Q_now.copy()

		for q in range(sample_s.shape[0]):
			Y[q, sample_a[q]] = sample_r[q] + self.gamma * max(self.Q_next[q])
		
		#Y[:,sample_a] = sample_r + self.gamma * self.Q_next.max(axis=1)

		# fit Q with samples
		self.Q.fit(sample_s, Y, epochs=1, verbose = 0)
		
		# fixed target update
		if self.fit_counter % 64 == 0:
			self.Q_Target = self.Q
		self.fit_counter+=1

	def policyLearn(self,sample_s):
		#fit the actor
		num_samples = sample_s.shape[0]
		best_action = np.zeros(num_samples)
		y_action = np.argmax(self.Q_next,axis=1)
		for k in range(num_samples):
			best_action[k] = sample_s[k,self.agent_lookback-1,self.MVpos] +\
				(self.action_dict[y_action[k]]/self.training_scanrate)
			
		self.actor.fit(sample_s, best_action, epochs=1, verbose = 0)

	def savePolicy(self,filename):
		
		# Convert the model.
		converter = tf.lite.TFLiteConverter.from_keras_model(self.actor)
		converter.target_spec.supported_ops = [
  			tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  			tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
		]
		
		tflite_model = converter.convert()
		
		# Save the model.
		with open(filename + '_policy.tflite', 'wb') as f:
			f.write(tflite_model)

	def PID(self,P,I,D,PVindex,SVindex,scanrate=1):
		self.PID = PID(P,I,D,PVindex,SVindex,scanrate)
		self.PIDcontroller = True

	def explore (self,state):
		if self.PIDcontroller:
			diff = self.PID.step(state)
			action = np.argmin(abs(self.action_val-diff))
		
		else:
			action = np.random.choice(self.action_space)

		return action