import tensorflow as tf
import json
from AIOP.Policy_Validate import Policy_Validate


# gpu = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu[0], True)

#get a list of the training files
data_dir= './data/norm_data/'
with open('val.json', 'r') as savefile:
	data_dict = json.load(savefile)

#list digital twin experiment files
dt1 = [
	'dt/LIC101_AiPV7588', 
	'dt/LIC101_AiPV7911', 
	'dt/LIC101_AiPV9215', 
		]

#list controller models
model_dir = 'LIC01_AiMV6625/' 
modelname = 'LIC01_AiMV_policy.tflite'


# Load the TFLite model and allocate tensors.
model = tf.lite.Interpreter(model_dir+modelname)
model.allocate_tensors()

# Get input and output tensors.
input_details = model.get_input_details()
output_details = model.get_output_details()

#Pull info from the controller config file
with open(model_dir +'config.json', 'r') as config_file:
	config = json.load(config_file)

PVindex = config['variables']['pvIndex']
SVindex = config['variables']['svIndex']
MVindex = config['variables']['dependantVar']
agentIndex = config['variables']['independantVars']
agent_lookback = config['agent_lookback']
training_scanrate = config['training_scanrate']


episode_length = 500

##################################################################
#---------------------Validate Policy----------------------------
##################################################################

for ep in range(4):
	#Import Validation Tool
	val = Policy_Validate(data_dir=data_dir,agentIndex=agentIndex,MVindex=MVindex,
				SVindex=SVindex,PVindex=PVindex,dt1=dt1,episode_length=episode_length,
				agent_lookback=agent_lookback)

	#initalize validation loop
	state,done = val.reset()
	#execute the episode
	while not done:

		#change controller with max action
		state = state[::training_scanrate].astype('float32')
		#select the last n samples
		state = state[-agent_lookback:].reshape(1,agent_lookback,len(agentIndex))
		#run model
		model.set_tensor(input_details[0]['index'], state)
		model.invoke()
		control = model.get_tensor(output_details[0]['index'])[0][0]

		# advance Environment with policy action
		state_,done = val.step(control)

		# advance state
		state = state_

	val.plot_eps(model_dir)
