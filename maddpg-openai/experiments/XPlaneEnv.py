## X-Plane Env Object
## Q-Learning Object
## Deep Q Learn AP (Auto Pilot)
## xplane environment for training the RL agent
## xPLANE and Reinforcement Learning - AI/RL based AutoPilot
## Deep Q AP
## Book -> Getting Started with Deep Learning: Programming and Methodologies using Python
## By Ricardo Calix
## www.rcalix.com
## Copyright (c) 2020, Ricardo A. Calix, Ph.D.

################################################################################

import imp
import time
import random
import scipy.spatial.distance as distance
import numpy as np
from math import radians, cos, sin, asin, sqrt

################################################################################

## positions in xplane are observations
## format [lat, long, alt, pitch, roll, true heading/yaw, gear]
## Palo Alto
## starting_position    = [37.524, -122.06899,  4000, 0.0, 0.0, 0.0, 1]
## Sunol Regional Wilderness (20 kms about east from Palo Alto)
## destination_position = [37.505, -121.843611, 4000, 0.0, 0.0, 0.0, 1]

################################################################################

class XPlaneEnv():

    def __init__(self, states, orig, dest, acts_bin, end_param):
        self.starting_position = orig
        self.destination_position = dest
        self.previous_position = orig
        self.actions_binary_n = acts_bin
        self.end_game_threshold = end_param
        self.n_states = states
        self.n_bins_state = int(        self.n_states**(1.0/3.0)        )        ## 9 <- 9x9x9 = 729 (cube root of n_states)
            
    ################################################################################
 
    def calculateDistance(self, a, b):
        Dist = 0
        p1 = (  float(a[0]), float(a[1]), float(a[2])   )
        p2 = (  float(b[0]), float(b[1]), float(b[2])   )
        Dist = distance.euclidean(p1, p2)
        return Dist
    
    ################################################################################
    ## calcs distance from 2 gps coordinates in kms
    
    def kms_calc_distance(self, a, b):
        
        lat1 = float(a[0])
        lon1 = float(a[1])
        alt1 = float(a[2])

        lat2 = float(b[0])
        lon2 = float(b[1])
        alt2 = float(b[2])
        
        lat1 = radians(lat1)
        lon1 = radians(lon1)
        
        lat2 = radians(lat2)
        lon2 = radians(lon2)
      
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
      
        c = 2 * asin(sqrt(a))
         
        # Radius of earth in kilometers. Use 3956 for miles
        r = 6371
           
        # calculate the result
        return(c * r)


    #######################################################
    
    def altitude_coords_calc(self, curr_posi, dest_posi):
        ## observation/position = [lat, long, alt, pitch, roll, yaw, gear]
        higher_than = 0
        lower_than = 0
        alt_curr = curr_posi[2]
        alt_dest = dest_posi[2]
        if alt_curr > alt_dest:
            higher_than = 1
        elif alt_curr < alt_dest:
            lower_than = 1
        return higher_than, lower_than

    ################################################################################################

    def latitude_coords_calc(self, curr_posi, dest_posi):
        ## observation = [lat, long, alt, pitch, roll, yaw, gear]
        lat_below = 0
        lat_above = 0
        lat_curr = curr_posi[0]
        lat_dest = dest_posi[0]
        if lat_curr < lat_dest:
            lat_below = 1
        elif lat_curr > lat_dest:
            lat_above = 1
        return lat_below, lat_above

    ##########################################################################################

    def longitude_coords_calc(self, curr_posi, dest_posi):
        # observation = [lat, long, alt, pitch, roll, yaw, gear]
        lon_right = 0
        lon_left = 0
        lon_curr = curr_posi[1]
        lon_dest = dest_posi[1]
        if lon_curr > lon_dest:
            lon_right = 1
        elif lon_curr < lon_dest:
            lon_left = 1
        return lon_right, lon_left

    ##########################################################################################
    ## convert values in range A=(0, 360) to values in range B=(-180, 180)

    def convert_rangeA_to_rangeB(self, old_value):
        old_min = 0.0
        old_max = 360.0
        new_min = -180.0
        new_max = 180.0
        new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
        return new_value

    ################################################################################
    
    def reward_distance_exponential(self, current_position):
        distance_max = self.calculateDistance(self.starting_position, self.destination_position)
        curr_dist    = self.calculateDistance(current_position, self.destination_position)
        
        distance_max = self.kms_calc_distance(self.starting_position, self.destination_position)* 3280.0
        curr_dist    = self.kms_calc_distance(current_position, self.destination_position)* 3280.0
        
        ## in feet
        reward_dist = 1 - (curr_dist/distance_max+1)**0.4
        return reward_dist
          

    #################################################################################
    #################################################################################
    #################################################################################
    #################################################################################

    def encode_state_xplane(self, pitch, roll, yaw):
        # 9x9x9 = 729
        # 2x2x2 =   8
        #print("***********************************")
        #print(pitch, roll, yaw)
     
        
        i = pitch
        i = i * self.n_bins_state
        i = i + roll
        i  = i * self.n_bins_state
        i = i + yaw
        return i
        
    ##########################################################

    def decode_state_xplane(self, i):
        #9x9x9 = 729
        #2x2x2 =   8
        out = []
        out.append(i % self.n_bins_state)
        i = i // self.n_bins_state
        out.append(i % self.n_bins_state)
        i = i // self.n_bins_state
        out.append(i % self.n_bins_state)
        i = i // self.n_bins_state
        result = [ out[2], out[1], out[0]  ]
        return result

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    ## i  -->>  [0,0,0,0,0,1,0,0,0]  -->>  5
    
    def range_to_vector_index_729(self, i):
        #print(i)
        if -180 <= i < -80:        
           return 0
        elif -80 <= i < -40:       
           return 1
        elif -40 <= i < -15:       
           return 2
        elif -15 <= i < 0:         
           return 3
        elif 0 <= i < 10:          
           return 4
        elif 10 <= i < 25:         
           return 5
        elif 25 <= i < 50:         
           return 6
        elif 50 <= i < 90:         
           return 7
        elif 90 <= i < 180:        
           return 8
        else:
           return 0

  
    ##########################################################################################
    ## select priority state
    
    def select_priority_state(self, pitch_ob, roll_ob, yaw_ob):
        yaw_ran = random.randint(1, 10)
        if (    (pitch_ob >= 0) and (abs(pitch_ob) > abs(roll_ob)) and (yaw_ran > 2)  ):        
            return 0, 0, 0
        elif ((pitch_ob < 0) and (abs(pitch_ob) > abs(roll_ob)) and (yaw_ran > 2)    ):
            return 0, 0, 1  
        elif ((roll_ob >=0) and (abs(roll_ob) > abs(pitch_ob)) and (yaw_ran > 2)     ):        
            return 0, 1, 0
        elif ((roll_ob < 0) and (abs(roll_ob) > abs(pitch_ob)) and (yaw_ran > 2)    ):
            return 0, 1, 1     
        elif (   (yaw_ob >= 0)  ):        
            return 1, 0, 0
        elif (   (yaw_ob < 0)   ):
            return 1, 0, 1     
        #elif (??):        
        #    return 1, 1, 0
        else:
            return 1, 1, 1 


    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    
    def six_pack_values_to_vectors(self, pitch_ob, roll_ob, yaw_ob):
      
        bin1, bin2, bin3 = self.select_priority_state(pitch_ob, roll_ob, yaw_ob)

        # (1, 1, 1)  -->>   8
        #state = self.encode_s
        state = self.encode_state_xplane(bin1, bin2, bin3)

        return state
        
    ##########################################################################################
    
    def get_state_from_observation(self, observation):
       # observation = [lat, long, alt, pitch, roll, yaw, gear]
       # states are pitch, roll, yaw readings from the six pack
       # states = 9x9x9 = 729 or 2x2x2 = 8
       
       state = 0
    
       pitch_ob = observation[3] #pitch
       roll_ob = observation[4] #roll
       yaw_ob = self.convert_rangeA_to_rangeB(    observation[5]    )      #yaw
       
       state = self.six_pack_values_to_vectors(pitch_ob, roll_ob, yaw_ob)
          
       return state

    
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    
    
    def reward_function(self, action, position_before_action, current_position):
        done = False
        reward = 0
        remaining_distance_in_kms = 0.0
        denominator = 90.0
        
      
        #######################################################################
        ## actions_binary = [pi+, pi-, ro+, ro-, ru+, ru-]
        
        ## if action == pitch up
        if (action == 0):
            pitch_level_sign = current_position[3]
            if (pitch_level_sign <= 0.0):
                reward = 10.0 #* (abs(pitch_level_sign)/denominator)
            else:
                reward = -10.0 #* (abs(pitch_level_sign)/denominator)
        
        #######################################################################
        
        ## if action == pitch down
        if (action == 1):
            pitch_level_sign = current_position[3]
            if (pitch_level_sign > 0.0):
                reward = 10.0 #* (abs(pitch_level_sign)/denominator)
            else:
                reward = -10.0 #* (abs(pitch_level_sign)/denominator)
        
        #######################################################################
        
        ## if action == roll right
        if (action == 2):
            roll_level_sign = current_position[4]
            if (roll_level_sign <= 0.0):
                reward = 10.0 #* (abs(roll_level_sign)/denominator)
            else:
                reward = -10.0 #* (abs(roll_level_sign)/denominator)
                
        #######################################################################
                
        ## if action == roll left
        if (action == 3):
            roll_level_sign = current_position[4]
            if (roll_level_sign > 0.0):
                reward = 10.0 #* (abs(roll_level_sign)/denominator)
            else:
                reward = -10.0 #* (abs(roll_level_sign)/denominator)
  
        ########################################################################

        ## if action == rudder +
        if (action == 4):
            rudder_level_sign = self.convert_rangeA_to_rangeB(  current_position[5]   )       
                      
            if (rudder_level_sign <= 0.0):
                reward = 10.0 #* (abs(rudder_level_sign)/denominator)
            else:
                reward = -10.0 #* (abs(rudder_level_sign)/denominator)
            
                
        #########################################################################
     
        ## if action == rudder -
        if (action == 5):
            
            rudder_level_sign = self.convert_rangeA_to_rangeB(   current_position[5]    )
            if (rudder_level_sign > 0.0):
                reward = 10.0 #* (abs(rudder_level_sign)/denominator)
            else:
                reward = -10.0 #* (abs(rudder_level_sign)/denominator)
          
     
        #########################################################################
        
        remaining_distance_in_kms = self.kms_calc_distance(current_position, self.destination_position)
        remaining_distance_in_feet = remaining_distance_in_kms * 3280.0 ## 1 km = 3280 feet
        
        if ( remaining_distance_in_feet <= self.end_game_threshold  ):
            done = True
        
        return reward, done, remaining_distance_in_kms
        
    ########################################################################
    
    def step(self, action, position_before_action, current_position):
        done = False
        reward = 0
        reward, done, kms_to_go = self.reward_function(action, position_before_action, current_position)
        return reward, done, kms_to_go
      

###############################################################################



