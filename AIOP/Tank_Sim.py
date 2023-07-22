import numpy as np
from AIOP.OUNoise import OUActionNoise

class Tank(object):
	def __init__(self,episode_length,agent_lookback,scan_rate=1,noise_stdv = 0.01):
		self.diameter = 72 #inches
		self.height = 96 #inches
		self.area = 3.14159*(self.diameter/2)**2
		self.gallons_to_in3 = 231
		self.volume = self.area*self.height/self.gallons_to_in3
		self.state_variables = ['LIC101_PV','LIC101_SV','LIC101_MV','FIC11_PV','FIC12_PV','FIC13_PV']
		self.PV = 'LIC101_PV'
		self.MV = 'LIC101_MV'
		self.SV = 'LIC101_SV'
		self.PV_index = self.state_variables.index(self.PV)
		self.MV_index = self.state_variables.index(self.MV)
		self.SV_index = self.state_variables.index(self.SV)
		self.episode_length = episode_length
		self.agent_lookback=agent_lookback
		self.scan_rate = scan_rate
		self.MV_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(noise_stdv) * np.ones(1))
		self.FIC1_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(noise_stdv) * np.ones(1))
		self.FIC2_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(noise_stdv) * np.ones(1))
		self.FIC3_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(noise_stdv) * np.ones(1))



	def reset (self):
		#make an episode array
		self.episode_array = np.zeros((self.episode_length+self.agent_lookback,len(self.state_variables)),dtype='float32')
		initial_data = np.random.rand(len(self.state_variables))

		#calculate a the initial cv position so that it starts out stable
		flow_in = (initial_data[3]*175 + initial_data[4]*215+initial_data[5]*140)
		action_us = np.log(flow_in/35.594)/0.02679
		action = action_us / 100
		initial_data[self.MV_index] = action

		for i in range(self.agent_lookback):
			self.episode_array[i] = initial_data
		
		state = self.episode_array[0:self.agent_lookback]

		#initalize the episode counter and done flag
		self.episode_counter = self.agent_lookback-1  #index from 0
		self.done = False

		return state,self.done

	def step(self,action):
		action_us = np.clip(action+self.MV_noise()/2,0,1) * 100
		
		flow_out = 35.594*(2.718**(0.02679*action_us))*(self.scan_rate / 60)
		flow_in = (
					(self.episode_array[self.episode_counter,3]+self.FIC1_noise())*175 + \
					(self.episode_array[self.episode_counter,4]+self.FIC2_noise())*215 + \
					(self.episode_array[self.episode_counter,5]+self.FIC3_noise())*140) * \
					(self.scan_rate/60)

		#calculate new level in inches
		volume_diff = (flow_in - flow_out) * self.gallons_to_in3 #in3
		level_diff = volume_diff / self.area #in
		level_ = (self.episode_array[self.episode_counter,self.PV_index] + level_diff/self.height) 
		
		#increment episode counter
		self.episode_counter+=1

		#TODO detirmine if we want to change in-flow rates else leave the below there
		self.episode_array[self.episode_counter] = self.episode_array[self.episode_counter-1]

		#fill in new state space
		self.episode_array[self.episode_counter,self.PV_index] = level_
		self.episode_array[self.episode_counter,self.MV_index] = action

		#get state_ to return
		state_ = self.episode_array[self.episode_counter-self.agent_lookback:self.episode_counter]

		if self.episode_counter > self.episode_length + self.agent_lookback-2:
			self.done = True
		
		
		return state_,self.done