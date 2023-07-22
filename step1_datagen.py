import os
from AIOP.PID import PID
from AIOP.Tank_Sim import Tank
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime


def datagen(name,epsl=2000,scanrate=5,plot=False,drift=.01):
	tank = Tank(episode_length=epsl,agent_lookback=1,scan_rate=scanrate,noise_stdv = drift)
	state,done = tank.reset()

	#['Level','Setpoint','CV_pos','flow_in1','flow_in2','flow_in3']
	pid = PID(P=2,I=20,D=.2,PVindex=0,SVindex=1,scanrate=scanrate)

	while not done:
		dMV = pid.step(state)
		action = state[0,2] - dMV
		if action > 1:
			action = 1
		
		if action < 0:
			action = 0
		state_,done = tank.step(action)
		state = state_

		#change setpoint and inlet flows every so often
		if tank.episode_counter % 1000 == 0:
			tank.episode_array[tank.episode_counter,1] = .2 + np.random.rand()*.6
		
		if tank.episode_counter % 721 == 0:
			tank.episode_array[tank.episode_counter,3] = np.random.rand()
		
		if tank.episode_counter % 1432 == 0:
			tank.episode_array[tank.episode_counter,4] = np.random.rand()
		
		if tank.episode_counter % 955 == 0:
			tank.episode_array[tank.episode_counter,5] = np.random.rand()

		if tank.episode_counter % 2000 == 0:
			print(tank.episode_counter)

	#save data
	eps = pd.DataFrame(tank.episode_array, columns=tank.state_variables)

	#create timestamps so we can use app.py
	timestamp = [datetime.datetime(2020,3,1)]
	for t in range(1,eps.shape[0]):
		timestamp.append(timestamp[t-1]+ datetime.timedelta(seconds = scanrate))
	eps['TimeStamp'] = pd.to_datetime(timestamp)

	#unnormalize
	#['LIC101_PV','LCI101_SV','LIC101_MV','FIC11_PV','FIC12_PV','FIC13_PV']
	eps['LIC101_PV'] = eps['LIC101_PV']*96
	eps['LIC101_SV'] = eps['LIC101_SV']*96
	eps['LIC101_MV'] = eps['LIC101_MV']*100
	eps['FIC11_PV'] = eps['FIC11_PV']*175
	eps['FIC12_PV'] = eps['FIC12_PV']*215
	eps['FIC13_PV'] = eps['FIC13_PV']*140

	eps.to_csv(name, sep=',',index=False,header=True)
	
	if plot:
		#plot validation response
		plt.plot(eps[tank.SV], label = 'SetPoint')
		plt.plot(eps[tank.PV], label = 'PV')
		plt.plot(eps[tank.MV], label = 'MV')
		plt.legend(loc='lower right', shadow=True, fontsize='x-large')
		plt.xlabel('time (s)')
		plt.show()
		

if __name__ == "__main__":
	if not os.path.exists('data/'):
		os.mkdir('data/')
	if not os.path.exists('data/raw_data/'):
		os.mkdir('data/raw_data/')
	for i in range(10):
		datagen(name='data/raw_data/tank'+str(i)+'.csv', epsl=20000,scanrate=5,plot=False,drift = .02)




