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
import scipy.spatial.distance as distance
import numpy as np
from math import radians, cos, sin, asin, sqrt

################################################################################

## positions in xplane are observations
## format [lat, long, alt, pitch, roll, true heading/yaw, gear]  状态量是7维的
## Palo Alto
## starting_position    = [37.524, -122.06899,  4000, 0.0, 0.0, 0.0, 1]  #初始状态
## Sunol Regional Wilderness (20 kms about east from Palo Alto)
## destination_position = [37.505, -121.843611, 4000, 0.0, 0.0, 0.0, 1] 目的地

################################################################################

class XPlaneEnv():

    def __init__(self, orig, dest, acts_bin, end_param):
        self.starting_position = orig
        self.destination_position = dest
        self.previous_position = orig
        self.actions_binary_n = acts_bin
        self.end_game_threshold = end_param
            
    ################################################################################
 
    def calculateDistance(self, a, b):
        Dist = 0
        p1 = (  float(a[0]), float(a[1]), float(a[2])   )
        p2 = (  float(b[0]), float(b[1]), float(b[2])   )
        Dist = distance.euclidean(p1, p2)  #计算两点欧氏距离
        return Dist
    
    ################################################################################
    ## calcs distance from 2 gps coordinates in kms
    
    def kms_calc_distance(self, a, b):  # 通过经纬度计算两地距离，单位为千米,a,b为postion
        
        lat1 = float(a[0])
        lon1 = float(a[1])
        alt1 = float(a[2])

        lat2 = float(b[0])
        lon2 = float(b[1])
        alt2 = float(b[2])
        
        lat1 = radians(lat1)  # 转为弧度制
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
          

    #################################################################################
 
    def encode_state_xplane(self, pitch, roll, yaw):  # 编码飞机状态，返回一个数
        # 9x9x9 = 729
        #print("***********************************")
        #print(pitch, roll, yaw)
     
        #print(pitch, roll, yaw)
        
        i = pitch
        i = i * 9
        i = i + roll
        i  = i * 9
        i = i + yaw
        return i
        
    ##########################################################

    def decode_state_xplane(self, i):  # 解码飞机状态，从数->[pitch,row,yaw]
        #9x9x9 = 729
        out = []
        out.append(i % 9)  # 取余数
        i = i // 9   # " // "表示整数除法。向下取整
        out.append(i % 9)
        i = i // 9
        out.append(i % 9)
        i = i // 9
        result = [ out[2], out[1], out[0]  ]
        return result

    #######################################################
    
    def altitude_coords_calc(self, curr_posi, dest_posi):  # 目的地与当前位置的高度比较
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

    def latitude_coords_calc(self, curr_posi, dest_posi):  # 目的地与当前位置的纬度比较
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

    def longitude_coords_calc(self, curr_posi, dest_posi):  # 目的地与当前位置的经度比较
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
    ## i  -->>  [0,0,0,0,0,1,0,0,0]  -->>  5
    
    def range_to_vector_index(self, i):
        '''

        Args:
            i: pitch/row/yaw

        Returns:
            将pitch/row/yaw的范围转化为9个档次
        '''
        #print(i)
        if -180 <= i < -35:
           return 0
        elif -35 <= i < -25:
           return 1
        elif -25 <= i < -15:
           return 2
        elif -15 <= i < -5:
           return 3
        elif -5 <= i < 5:
           return 4
        elif 5 <= i < 15:
           return 5
        elif 15 <= i < 25:
           return 6
        elif 25 <= i < 35:
           return 7
        elif 35 <= i < 180:
           return 8
        else:
           return 0
    
    ##########################################################################################
    
    def six_pack_values_to_vectors(self, pitch_ob, roll_ob, yaw_ob):
        pitch_vector_index = self.range_to_vector_index(pitch_ob) #
        roll_vector_index = self.range_to_vector_index(roll_ob)
        yaw_vector_index = self.range_to_vector_index(yaw_ob)
        return pitch_vector_index, roll_vector_index, yaw_vector_index
        
    ##########################################################################################
    
    def get_state_from_observation(self, observation):
       # observation = [lat, long, alt, pitch, roll, yaw, gear]
       # states are pitch, roll, yaw readings from the six pack
       # states = 9x9x9 = 729
       
       state = 0
    
       pitch_ob = observation[3] #pitch
       roll_ob = observation[4] #roll
       yaw_ob = observation[5] #yaw

       # 将pitch、row,yaw的值转化为9个档次中的一个
       pitch_index, roll_index, yaw_index = self.six_pack_values_to_vectors(pitch_ob, roll_ob, yaw_ob)
       
       # (8, 8, 8)  -->> 728，将pitch、row,yaw3个档次编码成一个状态值
       state = self.encode_state_xplane(pitch_index, roll_index, yaw_index)
       
       return state

    ################################################################################
    
    def reward_distance_exponential(self, current_position):  # 通过距离计算奖励
        distance_max = self.calculateDistance(self.starting_position, self.destination_position)  # 正常计算两点距离
        curr_dist    = self.calculateDistance(current_position, self.destination_position)
        
        distance_max = self.kms_calc_distance(self.starting_position, self.destination_position)* 3280.0  #通过经纬度计算两点距离
        curr_dist    = self.kms_calc_distance(current_position, self.destination_position)* 3280.0
        
        ## in feet #英尺
        reward_dist = 1 - (curr_dist/distance_max+1)**0.4
        return reward_dist
    
    ####################################################################
    
    def reward_function(self, action, position_before_action, current_position):
        done = False
        reward = 0
        remaining_distance_in_kms = 0.0
        
        ######################################################################
        '''
        reward = self.reward_distance_exponential(current_position)
        if (curr_dist > 1):
            reward = reward - 1 * (1 - (1/curr_dist))
        '''
        #######################################################################
        ## actions_binary = [pi+, pi-, ro+, ro-, ru+, ru-]
        
        ## if action == pitch up
        if (action == 0):
            pitch_level_sign = current_position[3]
            if (pitch_level_sign <= 0.0):
                reward = 10.0 * (abs(pitch_level_sign)/360.0)
            else:
                reward = -10.0 * (abs(pitch_level_sign)/360.0)
        
        #######################################################################
        
        ## if action == pitch down
        if (action == 1):
            pitch_level_sign = current_position[3]
            if (pitch_level_sign > 0.0):
                reward = 10.0 * (abs(pitch_level_sign)/360.0)
            else:
                reward = -10.0 * (abs(pitch_level_sign)/360.0)
        
        #######################################################################
        
        ## if action == roll right
        if (action == 2):
            roll_level_sign = current_position[4]
            if (roll_level_sign <= 0.0):
                reward = 10.0 * (abs(roll_level_sign)/360.0)
            else:
                reward = -10.0 * (abs(roll_level_sign)/360.0)
                
        #######################################################################
                
        ## if action == roll left
        if (action == 3):
            roll_level_sign = current_position[4]
            if (roll_level_sign > 0.0):
                reward = 10.0 * (abs(roll_level_sign)/360.0)
            else:
                reward = -10.0 * (abs(roll_level_sign)/360.0)
  
        ########################################################################
        
        ## if action == rudder +
        if (action == 4):
            rudder_level_sign = current_position[5]
            if (rudder_level_sign <= 360.0 and rudder_level_sign >= 180.0):
                reward = 10.0 * (1 - (abs(rudder_level_sign)/360.0))
            else:
                reward = -10.0 * (1 - (abs(rudder_level_sign)/360.0))
                
        #########################################################################
     
        ## if action == rudder -
        if (action == 5):
            rudder_level_sign = current_position[5]
            if (rudder_level_sign >= 0.0 and rudder_level_sign < 180.0):
                reward = 10 * (   abs(rudder_level_sign)/360.0   )
            else:
                reward = -10.0 * (   abs(rudder_level_sign)/360.0   )
     
        #########################################################################
        
        remaining_distance_in_kms = self.kms_calc_distance(current_position, self.destination_position)
        remaining_distance_in_feet = remaining_distance_in_kms * 3280.0 ## 1 km = 3280 feet
        
        if ( remaining_distance_in_feet <= self.end_game_threshold  ):
            done = True
        
        return reward, done, remaining_distance_in_kms
        
    ########################################################################
    
    def step(self, action, position_before_action, current_position):  # 这里的action为索引值
        done = False
        reward = 0
        reward, done, kms_to_go = self.reward_function(action, position_before_action, current_position)
        return reward, done, kms_to_go
      

###############################################################################



