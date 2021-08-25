import gym
import gym.spaces as spaces
from scipy.spatial.distance import pdist, squareform
import scipy.spatial.distance as distance

import xpc as xp
import parameters
import space_definition as envSpaces
import numpy as np
import itertools
from time import sleep #, clock
import tensorflow as tf

class initial:

    def connect(clientAddr, xpHost, xpPort, clientPort, timeout):
        return xp.XPlaneConnect(clientAddr, xpHost, xpPort, clientPort, timeout)


class XplaneEnv(gym.Env):

    def __init__(self, clientAddr='0.0.0.0', xpHost='127.0.0.1', xpPort=49009, clientPort=1, timeout=1000):
        # CLIENT = client
        XplaneEnv.CLIENT = None
        # print(parameters)
        # envSpace = envSpaces.xplane_space()

        self.ControlParameters = parameters.getParameters()
        self.obs_space=[]
        self.act_space=[]
        self.n=1
        # self.act_dim=5
        self.act_dim=4
        self.obs_dim=8
        self.crash_flag=False
        self.extreme_count = 0
        self.success_count = 1
        '''
        pitch是俯仰角，是“点头“
        yaw是偏航角，是‘摇头’      
        roll是旋转角，是“翻滚”
        elevator 升降舵 保持飞机的府仰平衡
        ailerons  控制飞机转弯的
        rudder 方向舵 可以左右旋转和上下推动
        throttle 油门
        '''
        # action:[elevator,aileron,rudder,throttle,speed_brake] or my action : [elevator,aileron,throttle]
        # self.action_space = spaces.Box(low=np.array([ -1, -1, -1,-1,-0.5]),high=np.array([1,1,1,1,1.5]),dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1,-1,-1,0]),high=np.array([1,1,1,1]),dtype=np.float32)
        # state:[roll_rate,pitch_rate,yaw_rate,row,pitch,yaw,altitude,airspeed]
        self.observation_space = spaces.Box(low=np.array([ -360, -360, -360 ,-180,-180,0,0,0]),high=np.array([360,360,360,180,180,360,5000,1000]), dtype=np.float32)
        self.act_space.append(self.action_space)
        self.obs_space.append(self.observation_space)
        # self.episode_steps = 0
        self.ControlParameters.episodeStep = 0
        self.actions = [0 for _ in range(self.act_dim)]
        try:
            XplaneEnv.CLIENT = initial.connect(clientAddr, xpHost, xpPort, clientPort, timeout)
        except:
            print("connection error. Check your paramters")
        print('I am client', XplaneEnv.CLIENT)

    def close(self):
        XplaneEnv.CLIENT.close()

    def rewardCalcul(self, targetState, xplaneState,action):
        '''
        input : target state (a list containing the target heading, altitude and runtime)
                xplaneState(a list containing the aircraft heading , altitude at present timestep, and the running time)
                Note: if the aircraft crashes then the run time is small, thus the running time captures crashes
        output: Gaussian kernel similarîty between the two inputs. A value between 0 and 1
        # targetState[yaw_rate=0,roll=0,pitch=0,yaw=0,altitude=2500,airspeed=100]
        # targetState[
        '''

        reward = 0
        target = [0,0,0]
        current_state = [1.2*xplaneState[0], 1.4*xplaneState[3],0.5*xplaneState[4]]
        dist = distance.euclidean(target,current_state)  # roll,pitch,yaw数值上的距离 不用yaw吧
        # print(dist,current_state)
        if dist<=10:
            self.success_count +=2
            self.extreme_count = 0
            # if abs(current_state[2]) < 5:
            #     reward += 8 * self.success_count / dist
            # elif abs(current_state[3]) < 15:
            #     reward += 3* self.success_count/dist
            # else:
            #     reward += 2*self.success_count/dist
            reward += 10*self.success_count/dist
            print(self.success_count)
            if self.success_count>120:  # 一个episode中有60steps处于平稳飞行状态时（不一定连续），认为success
                self.ControlParameters.flag = True
                reward += 100* self.success_count/dist
                self.success_count = 0
        elif 10<dist<=30:
            self.success_count += 1
            print(self.success_count)
            # if abs(current_state[2]) < 10:
            #     reward += 4 / dist
            # elif abs(current_state[3]) < 25:
            #     reward += 3 /dist
            # else:
            #     reward += 2/dist
            # reward +=3/dist
            reward += 10*self.success_count/dist
            self.extreme_count = 0
            # self.success_count = 0
        elif 30<dist<=70:
            reward += 2 / dist
            self.extreme_count = 0
        elif dist >70:
            reward +=-0.5/dist
            # self.extreme_count = 0
            # self.success_count = 0

        if xplaneState[4]<200 or 100<xplaneState[1]<=180 or -180<xplaneState[1]<-100 or 100<xplaneState[2]<=180 or -180<xplaneState[2]<-100:  # 如果坠机或者翻过来了
            reward += -10
            # self.success_count = 0
            self.extreme_count +=1
            if self.extreme_count>20 or XplaneEnv.CLIENT.getDREFs(self.ControlParameters.crash)[0][0]:  # 连续10steps处于极端状态时，reset
                self.crash_flag=True
                self.extreme_count = 0
                print('crash or extreme state!!!')
        print(reward)
        return reward

    def step(self, actions):
        """
          obs, rew, done, info = env.step(actions)
        """
        self.ControlParameters.flag = False  # for synchronisation of training:  done
        #############################################
        # 查看 pevious action 与controls是否一致
        print("previous action", actions)  # previous action
        # print("action on ctrl ...", XplaneEnv.CLIENT.getCTRL())  # action on control surface
        # 如果不一致，则要确保一致（通过频率调节）
        #############################################

        XplaneEnv.CLIENT.pauseSim(False)  # unpause x plane simulation
        '''
        sendCTRL():
          * Latitudinal Stick [-1,1]
          * Longitudinal Stick [-1,1]
          * Rudder Pedals [-1, 1]
          * Throttle [-1, 1] # 这里写错了 应该为[0,1]
          * Gear (0=up, 1=down)
          * Flaps [0, 1]
          * Speedbrakes [-0.5, 1.5]
        '''
        control = [actions[0],actions[1],actions[2],actions[3],-998,-998,-998]
        XplaneEnv.CLIENT.sendCTRL(control)  # send action
        sleep(0.01)


        self.actions = actions  # set the previous action to current action.
        # XplaneEnv.CLIENT.pauseSim(True)
        #################################################
        # temporary variable for holding state values
        # state = []  # 8维
        # state8 = []
        ################################################
        # stateVariableTemp: local_vx,local_vy,local_vz
        stateVariableTemp = XplaneEnv.CLIENT.getDREFs(self.ControlParameters.stateVariable)  # 获取空速
        '''getPOSI():
        *Latitude(deg)
        *Longitude(deg)
        *Altitude(m above MSL)
        *Pitch(deg)
        *Roll(deg)
        *True Heading(deg)
        *Gear(0 = up, 1 = down)
        '''
        self.ControlParameters.stateAircraftPosition = list(XplaneEnv.CLIENT.getPOSI())
        self.ControlParameters.stateVariableValue = [i[0] for i in stateVariableTemp]
        # combine the position and other state parameters in temporary variable here--8维
        state = self.ControlParameters.stateAircraftPosition + self.ControlParameters.stateVariableValue
        ########################################################
        # ****************************************************************************************
        # *******************************other training parameters ******************
        # P = XplaneEnv.CLIENT.getDREF("sim/flightmodel/position/P")[0]  # The roll rotation rates (relative to the flight)
        # Q = XplaneEnv.CLIENT.getDREF("sim/flightmodel/position/Q")[0]  # The pitch rotation rates (relative to the flight)
        # R = XplaneEnv.CLIENT.getDREF("sim/flightmodel/position/R")[0]  # The yaw rotation rates (relative to the flight)
        velocity_name = ["sim/flightmodel/position/P","sim/flightmodel/position/Q","sim/flightmodel/position/R"]
        [P,Q,R] = XplaneEnv.CLIENT.getDREFs(velocity_name)
        # print(P)
        ##############################################################################

        self.ControlParameters.state8['roll_rate'] = P[0]  # The roll rotation rates (relative to the flight)
        self.ControlParameters.state8['pitch_rate'] = Q[0] # The pitch rotation rates (relative to the flight)
        self.ControlParameters.state8['yaw_rate'] = R[0]  # The yaw rotation rates (relative to the flight)
        self.ControlParameters.state8['Roll'] = state[4]  # roll
        self.ControlParameters.state8['Pitch'] = state[3]  # pitch
        self.ControlParameters.state8['Yaw'] = state[5]  # heading
        self.ControlParameters.state8['altitude'] = state[2]  # Altitude
        self.ControlParameters.state8['airspeed'] = state[7]  # local velocity x  OpenGL coordinates

        state8 = [i for i in self.ControlParameters.state8.values()]
        # print('state8', state8)
        ###########################################################################
        # *******************************reward computation ******************
        # targetState = [0, 0, 4, 0, 2500, 100]  # targetState[yaw_rate=0,roll=0,pitch=0,yaw=0,altitude=2500,airspeed=100]
        targetState = [0,0,0,0,4,0,2500,100] # roll_rate,pitch_rate,yaw_rate,roll,pitch,altitude,airspeed
        xplaneState = [state8[0],state8[1],state8[2], state8[3],state8[4],state8[5],state8[6],state8[7]]  # present situation
        reward = self.rewardCalcul(targetState, xplaneState,actions)
        # print(reward)
        self.ControlParameters.episodeReward = reward
        ###########################################################################
        # 如果坠机
        # if XplaneEnv.CLIENT.getDREFs(self.ControlParameters.crash)[0][0]:
        # if state[2] <=200 or 120<state[3]<=180 or -180<state[3]<-120 or 120<state[4]<=180 or -180<state[4]<-120:  # 如果坠机或者翻过来了
        #     self.ControlParameters.flag = False
        #     self.crash_flag =True
        #     # self.ControlParameters.episodeReward =-1000
        #     print('crash or amazing!!!')
            # self.reset()
        # set flag to true if maximum steps has been achieved. Thus episode is finished.
        # set the maximum episode step to the value you want
        reward = self.ControlParameters.episodeReward
        ###########################################################################

        return np.array(state8), reward, self.ControlParameters.flag, self._get_info()  # self.ControlParameters.state8

    def _get_info(self):
        """Returns a dictionary contains debug info"""
        return {'control Parameters': self.ControlParameters, 'actions': self.action_space}

    def reset(self):
        """
        Reset environment and setup for new episode.
        Returns:
            initial state of reset environment.
        """
        XplaneEnv.CLIENT.sendCOMM('sim/operation/reset_flight')  # reset plane
        XplaneEnv.CLIENT.sendCOMM('sim/view/circle')  # set the external view
        self.actions = [0 for _ in range(self.act_dim)]
        init_position = ["sim/flightmodel/position/P",  # roll_rate
                          "sim/flightmodel/position/Q",  # pitch_rate
                          "sim/flightmodel/position/R",  # yaw_rate
                          "sim/flightmodel/position/phi",  # roll
                         "sim/flightmodel/position/theta",  # pitch
                          "sim/flightmodel/position/psi",  # heading
                         "sim/cockpit/autopilot/altitude",
                         "sim/flightmodel/position/indicated_airspeed"]
        init_vel = ['sim/flightmodel/position/local_vx',
                   'sim/flightmodel/position/local_vy',
                   'sim/flightmodel/position/local_vz',
                   'sim/flightmodel/position/P',
                   'sim/flightmodel/position/Q',
                   'sim/flightmodel/position/R']
        # values=[[0], [0], [0], [0], [0], [0],[2500],[100]]
        values=[np.random.uniform(-10,10), np.random.uniform(-10,10), np.random.uniform(-10,10),
                np.random.uniform(-90,90), np.random.uniform(-90,90), np.random.uniform(-90,90),
                2500, np.random.uniform(-20,20)+100]
        # XplaneEnv.CLIENT.sendDREFs(init_position, values)
        # stateTemp = XplaneEnv.CLIENT.getDREFs(init_position)
        # print(stateTemp)
        # self.ControlParameters.stateAircraftPosition = [i[0] for i in stateTemp]

        # 设置初始速度
        roll = values[3]/180*np.pi
        pitch = values[4]/180*np.pi
        heading = values[5]/180*np.pi
        Rr = np.array([[np.cos(roll), np.sin(roll), 0],
                       [-np.sin(roll), np.cos(roll), 0],
                       [0, 0, 1]])
        Rp = np.array([[1, 0, 0],
                       [0, np.cos(pitch), -np.sin(pitch)],
                       [0, np.sin(pitch), np.cos(pitch)]])
        Rh = np.array([[np.cos(heading), 0, -np.sin(heading)],
                       [0, 1, 0],
                       [np.sin(heading), 0, np.cos(heading)]])
        UnitVec = np.einsum('ij,jk,kl,l->i',Rh,Rp,Rr,np.array([0,0,-1]))*0.51444    # 0.51444是将konts转变为 m/s
        vel = [values[7]*UnitVec[0],values[7]*UnitVec[1],values[7]*UnitVec[2],values[0],values[1],values[2]]
        XplaneEnv.CLIENT.sendDREFs(init_vel, vel)  # reset the velocity

        pos = [-12.15,142.15, 2500, values[3], values[4], values[5], -998]
        self.ControlParameters.stateAircraftPosition = XplaneEnv.CLIENT.sendPOSI(pos)   # reset the position
        self.ControlParameters.stateVariableValue = []
        self.ControlParameters.episodeReward = 0.
        self.ControlParameters.flag = False
        self.ControlParameters.state8 = dict.fromkeys(self.ControlParameters.state8.keys(), 0)
        # state = self.ControlParameters.stateAircraftPosition
        # control = [0, 0, 0, 0, -998, -998, 0]
        # XplaneEnv.CLIENT.sendCTRL(control)  # send action
        state = values
        self.success_count =0

        XplaneEnv.CLIENT.sendCOMM('sim/instruments/DG_sync_mag')  # vacuum DG sync to magnetic north.
        # self.close()

        return state  # self.ControlParameters.state8

    def set_frame_rate(self,timestep):
        datarefs = ['sim/operation/misc/frame_rate_period', 'sim/operation/override/override_timestep']
        # enable override time step
        XplaneEnv.CLIENT.sendDREF(datarefs[1],0)
        # set time step
        XplaneEnv.CLIENT.sendDREF(datarefs[0], timestep)
        # client.sendDREF(datarefs[1], 0)
        # data = XplaneEnv.CLIENT.getDREFs(datarefs)
        # print("frame_rate_period:{}\ttime_step_enable:{}".format(data[0][0], data[1][0]))
