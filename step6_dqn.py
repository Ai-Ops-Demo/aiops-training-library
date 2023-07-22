
from AIOP.SimMaker import Simulator
from AIOP.DQN import ReplayBuffer,Agent
import numpy as np
import os
import json
import tensorflow as tf

# silence tf warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#keep tf from reserving 100% of GPU for this instance
import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

#--------------------Variables for Simulator-------------------------------
data_dir='data/norm_data/'
dt1 = [
	'10sDT/LIC101_AiPV9046',
	'10sDT/LIC101_AiPV7847'
	]
#what variables do you want the agent to see?
MVindex = 2
PVindex = 0
SVindex = 1

agentIndex = [0,1,2,3,4,5] 
agent_lookback=5

#about 3-5x as long as the system needs to respond to SP change
episode_length = 250

#add some noise to the SV to help with simulating responses.
SVnoise = 0.1

#reward paramaters
general = 1  				#proportional to error signal
stability = 1 				#for stability near setpoint
stability_tolerance =.003		#within this tolerance
response = 1 				#for reaching the setpoint quickly
response_tolerance =.05			#within this tolerance

#--------------------Variables for Agent-----------------------
controller_name = 'LIC01_AiMV'
gamma = 0.95
epsilon_decay = 0.99995
max_step = 0.1       	#how much can the agent move each timestep
training_scanrate = 2   #scanrate that the dt was trained on
execution_scanrate = 1  #rate that the model is to be executed


#make a directory to save stuff in
model_dir = controller_name + str(int(round(np.random.rand()*10000,0))) + '/'
os.mkdir(model_dir)
print(model_dir)

######################################################################
#------------Initalize Simulator from trained Environment-------------
######################################################################
sim = Simulator(dt1=dt1,data_dir=data_dir,agentIndex=agentIndex,
				MVindex=MVindex,SVindex=SVindex,PVindex=PVindex,
				agent_lookback=agent_lookback,episode_length=episode_length,
				SVnoise=SVnoise,stability=stability,stability_tolerance=stability_tolerance,
				response=response,response_tolerance=response_tolerance,general=general)

#######################################################################
#--------------Initalize DQN Agent/ReplayBuffer-----------------------
#######################################################################
buff = ReplayBuffer(agentIndex,agent_lookback=sim.agent_lookback,
						capacity=1000000,batch_size=64)
							
#Initialize Agent
agent = Agent(agentIndex=agentIndex,MVindex=MVindex,agent_lookback=agent_lookback,
			default_agent=False,gamma=gamma,epsilon_decay=epsilon_decay,
			min_epsilon=.01,max_step=max_step,training_scanrate=training_scanrate)

#you have the option to explore with a PID controller, or comment out for random exploration
#agent.PID(100,10,0,PVindex,SVindex,scanrate=5)

agent.buildQ(fc1_dims=128,fc2_dims=128,lr=.002)
agent.buildPolicy(fc1_dims=64,fc2_dims=64,lr=.002)

################################################################################
#---------------------config file--------------------------------------------
################################################################################

#import tag_dictionary
with open('norm_vals.json', 'r') as norm_file:
	tag_dict = json.load(norm_file)


#create a config file 
config = {}
config['variables'] = {
						'dependantVar':MVindex,
						'independantVars':agentIndex,
						'svIndex':SVindex,
						'pvIndex':PVindex,
						}
config['agent_lookback'] = agent_lookback
config['training_scanrate'] = training_scanrate
config['execution_scanrate'] = execution_scanrate
config['tag_normalize_dict'] = tag_dict
config['epsilon_decay'] = epsilon_decay
config['gamma'] = gamma
config['svNoise'] = SVnoise
config['episode_length'] = episode_length
config['max_step'] = max_step
config['rewards'] = {'general':general,
					'stability':stability,
					'stability_tolerance':stability_tolerance,
					'response': response,
					'response_tolerance':response_tolerance
						}
with open(model_dir +'config.json', 'w') as outfile:
	json.dump(config, outfile,indent=4)



##############################___DQN___#######################################
# ---------------------For Eposode 1, M do------------------------------------
##############################################################################
#calculate num of episodes to decay epsilon +60.
num_episodes = int(round(np.log(agent.min_epsilon)/(sim.episode_length*np.log(agent.epsilon_decay)),0))

scores = []

for episode in range(num_episodes):

	#reset score
	score = 0

	#reset simulator
	state,done = sim.reset()

	#initalize the first control position to be the same as the start of the episode data
	control=state[sim.agent_lookback-1,sim.MVpos]

	#execute the episode
	while not done:

		#select action
		action = agent.selectAction(state)
		
		#change controller with max action
		control += agent.action_dict[action]
		
		#keep controller from going past what the env has seen in training
		control=np.clip(control,sim.MV_min,sim.MV_max)

		# advance Environment with max action and get state_
		state_,reward,done = sim.step(control)
		
		#Store Transition
		buff.storeTransition(state,action,reward,state_)
		
		#Sample Mini Batch of Transitions
		sample_s,sample_a,sample_r,sample_s_= buff.sampleMiniBatch()

		#fit Q
		agent.qlearn(sample_s,sample_a,sample_r,sample_s_)

		#fit policy
		agent.policyLearn(sample_s)

		# advance state
		state = state_

		# get the score for the episode
		score += reward

	#save a history of episode scores
	scores = np.append(scores,score)
	
	if episode > 25:
		moving_average = np.mean(scores[episode-25:episode])
	else: 
		moving_average = 0
	
	if episode % 10 == 0:
		buff.saveEpisodes(model_dir + 'replaybuffer.csv') 
		agent.savePolicy(model_dir + controller_name)
		
		#Update config with max and min
		with open(model_dir +'config.json', 'r') as config_file:
			config = json.load(config_file)

		buff_min,buff_max= buff.get_min_max()

		min_training_range = {}
		for i in buff_min.index:
			min_training_range[i] = str(buff_min[i])
		config['min_training_range'] = min_training_range

		max_training_range = {}
		for i in buff_max.index:
			max_training_range[i] = str(buff_max[i])
		config['max_training_range'] = max_training_range

		with open(model_dir +'config.json', 'w') as outfile:
			json.dump(config, outfile,indent=4)

	print('episode_',episode ,' of ',num_episodes,' score_', round(score,0),
	 ' average_', round(moving_average,0), ' epsilon_', round(agent.epsilon,3))
	
	 

