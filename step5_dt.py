import tensorflow as tf
import os
from AIOP.DigitalTwin import DigitalTwin

# silence tf warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#keep tf from reserving 100% of GPU for this instance
import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

####################################################################
#----------User Defined Inputs--------------------------------------
####################################################################
data_dir='data/norm_data/'
dt_modelname = 'LIC101_AiPV'
# LIC101_PV,LIC101_SV,LIC101_MV,FIC11_PV,FIC12_PV,FIC13_PV,TimeStamp
dependantVar = 0
independantVars = [2,3,4,5]
dt_lookback = 10
scanrate = 1
##################################################################
#----------------Build and Train Environment Model------------
###################################################################
dt = DigitalTwin(dt_modelname = dt_modelname,data_dir=data_dir,
				num_val_sets=2,independantVars=independantVars,
				dependantVar=dependantVar,dt_lookback=dt_lookback,
                velocity=True,scanrate=scanrate)
			
dt.trainDt(gru1_dims=8,gru2_dims=16,lr=.001,ep=100,batch_size = 5000)
dt.validate(max_len = 500)