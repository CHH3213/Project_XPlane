def getParameters():

	  # Global dictionary for acrquiring the parameters for the training
	globalDictionary = {
	 # State Variables. This together with other parameters (to be defined later) will give us the 
	 # state of the aircraft. Note that this variables will be parsed to our function and the function
	 # returns a set of values. check xplane dataref file for definition of stateVariable

		"stateVariable" : ["sim/flightmodel/position/true_airspeed"],
		# State variables values are stored in list sateVariableValue
		"velocityVariable": ["sim/flightmodel/position/local_vx", "sim/flightmodel/position/local_vy",
						  "sim/flightmodel/position/local_vz"],
		"stateVariableValue": [],
		"on_ground":["sim/flightmodel2/gear/on_ground"],  # Is this wheel on the ground
		"crash":["sim/flightmodel/engine/ENGN_running"],  # Engine on and using fuel (only reliable in 740 and later)
		# Aircraft Position state Variables
		"stateAircraftPosition" : [],
		"episodeReward": 0.,
		"flag": False,
		"state":[0.,0.,0.,0.,0.,0.,0.],
		"state8":{"roll_rate":0,"pitch_rate":0,"yaw_rate":0,"Roll":0,"Pitch":0, "Yaw":0, "altitude": 0, "airspeed":0},
		"reset":False

		}

	globalDictionary = dotdict(globalDictionary) # 可以使用点操作符访问字典


	return globalDictionary

class dotdict(dict):
   """dot.notation access to dictionary attributes"""
   __getattr__ = dict.get
   __setattr__ = dict.__setitem__
   __delattr__ = dict.__delitem__
