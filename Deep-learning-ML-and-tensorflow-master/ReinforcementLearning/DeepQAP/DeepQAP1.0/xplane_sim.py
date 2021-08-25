## xplane env sim
## Q-Learning Object
## Deep Q Learn AP (Auto Pilot)
## xplane environment for training the RL agent
## xPLANE and Reinforcement Learning - AI/RL based AutoPilot
## Deep Q AP
## Book -> Getting Started with Deep Learning: Programming and Methodologies using Python
## By Ricardo Calix
## www.rcalix.com
## Copyright (c) 2020, Ricardo A. Calix, Ph.D.

##########################################################################

import imp   # import的缩写,比import方式更好
import numpy as np

##########################################################################

xpc = imp.load_source('xpc','xpc.py')  # 第一个参数为命名，重要的是第二个参数指定引用函数路径就行了。

##########################################################################

pi_d = 0.05
ro_d = 0.05
ru_d = 0.05

actions_n = 6

##########################################################################

drefs_position = ["sim/flightmodel/position/latitude",  # 纬度
                  "sim/flightmodel/position/longitude",  # 经度
                  "sim/flightmodel/position/elevation",  # 海拔
                  "sim/flightmodel/position/theta",  # The pitch relative to the plane normal to the Y axis in degrees
                  "sim/flightmodel/position/phi",  # The roll of the aircraft in degrees
                  "sim/flightmodel/position/psi",  # The true heading of the aircraft in degrees from the Z axis
                  "sim/cockpit/switches/gear_handle_status"]

###########################################################################

drefs_controls = ["sim/cockpit2/controls/yoke_pitch_ratio",   # This is how much the user input has deflected the yoke in the cockpit, in ratio, where -1.0 is full down, and 1.0 is full up.
                  "sim/cockpit2/controls/yoke_roll_ratio",  #  This is how much the user input has deflected the yoke in the cockpit, in ratio, where -1.0 is full left, and 1.0 is full right.
                  "sim/cockpit2/controls/yoke_heading_ratio",  # This is how much the user input has deflected the rudder in the cockpit, in ratio, where -1.0 is full left, and 1.0 is full right.
                  "sim/flightmodel/engine/ENGN_thro",  # Throttle (per engine) as set by user, 0 = idle, 1 = max
                  "sim/cockpit/switches/gear_handle_status",  # Gear handle is up or down?
                  "sim/flightmodel/controls/flaprqst"]  # Requested flap deployment, 0 = off, 1 = max

############################################################################

def send_posi(posi):
    client = xpc.XPlaneConnect()
    client.sendPOSI(posi)
    client.close()
    
##############################################################################

def send_ctrl(ctrl):  #take action
    client = xpc.XPlaneConnect()
    client.sendCTRL(ctrl)
    client.close()
 
##############################################################################

def get_posi():
    client = xpc.XPlaneConnect()
    #r = client.getDREFs(drefs_position)
    r = client.getPOSI(0)
    client.close()
    return r
   
##############################################################################

def get_ctrl():
    client = xpc.XPlaneConnect()
    #r = client.getDREFs(drefs_controls)
    r = client.getCTRL(0)
    client.close()
    return r
    
##############################################################################

def reset(posi):
    send_posi(posi)  #将位置发送给飞机
    new_posi = get_posi()  # 获取飞机当前位置
    return new_posi
 
##############################################################################

def convert_action_to_control(ctrl, action, reward):  #将训练得到的action转化为飞机的控制输入
    ## action is the selected index of the one-hot encoded vector
    ## ctrl = [-1.0, 0.8, -998, -998, 0, 0] # [pitch, roll, rudder, throttle, gear, throttle]
    ## actions is a one_hot encoded vector
    ## actions = [] ## 2**6 = 64种action组合
    ## action is the decimal representation of actions_binary # action是actiopn_binary的10进制表示形式
    ## actions_binary = [pi+, pi-, ro+, ro-, ru+, ru-]
    ## actions_binary = [1, 0, 0, 1, 0, 0] -> (up pitch, left roll, no rudder)
  
    pitch = ctrl[0]
    roll =  ctrl[1]
    rudder = ctrl[2]
    
    actions_binary = np.zeros(actions_n, dtype=int)  # actions_n=6
    
    actions_binary[action] = 1
    # pi_d ro_d ru_d =0.05 0.05 0.05 action_binary是one hot 编码格式
    pitch = pitch + actions_binary[0] * pi_d - actions_binary[1] * pi_d 
    roll = roll + actions_binary[2] * ro_d - actions_binary[3] * ro_d
    rudder = rudder + actions_binary[4] * ru_d - actions_binary[5] * ru_d
    
    '''
    pitch  = np.clip(pitch, -1.0, 1.0) #np.clip将范围限定到（-1，1），超出边界的用边界值替代
    roll   = np.clip(roll, -1.0, 1.0)
    rudder = np.clip(rudder, -1.0, 1.0)  # rudder舵 
    '''
    
    pitch  = np.clip(pitch, -0.10, 0.10)
    roll   = np.clip(roll, -0.15, 0.15)
    rudder = np.clip(rudder, -0.20, 0.20)
    
    ctrl = [pitch, roll, rudder, -998, 0, 0]
    return ctrl, actions_binary

##############################################################################
 
def update(action, reward):  # 更新，应该在env中的step中调用
    old_ctrl = get_ctrl()
    new_ctrl, actions_binary = convert_action_to_control(old_ctrl, action, reward)
    send_ctrl(new_ctrl)  ## set control surfaces e.g. pilot the plane
    posi = get_posi()  # 获取当前位置
    return posi, actions_binary, new_ctrl
 
#############################################################################


