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
import time
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
        # action:[elevator,aileron,rudder,throttle,speed_brake]
        self.action_space = spaces.Box(low=np.array([ -1, -1, -1,0]),high=np.array([1,1,1,1]),dtype=np.float32)
        # state:[roll_rate,pitch_rate,yaw_rate,row,pitch,yaw,altitude,airspeed]
        self.observation_space = spaces.Box(low=np.array([ -360, -360, -360 ,-180,-180,0,0,0]),high=np.array([360,360,360,180,180,360,5000,150]), dtype=np.float32)
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

    def rewardCalcul(self, targetState, xplaneState, prev_xplaneState, action):
        '''
        input : target state (a list containing the target heading, altitude and runtime)
                xplaneState(a list containing the aircraft heading , altitude at present timestep, and the running time)
                Note: if the aircraft crashes then the run time is small, thus the running time captures crashes
        output: Gaussian kernel similarîty between the two inputs. A value between 0 and 1
        # targetState[yaw_rate=0,roll=0,pitch=0,yaw=0,altitude=2500,airspeed=100]
        '''

        reward = 0
        print(xplaneState)
        #####################################################
        data_tar_curr = np.array([targetState, xplaneState])
        data_tar_prev = np.array([targetState, prev_xplaneState])
        pairwise_dists_tar_curr = pdist(data_tar_curr, 'cosine')  # 计算向量targetState和xplaneState之间的余弦距离，1-pairwise_dists表示相似度,体现方向上的相对差距
        pairwise_dists_tar_prev = pdist(data_tar_prev, 'cosine')  # 计算向量targetState和prev_xplaneState之间的余弦距离，1-pairwise_dists表示相似度,体现方向上的相对差距
        # print('delta_dir',pairwise_dists_tar_curr-pairwise_dists_tar_prev)
        # print(pairwise_dists_tar_curr)
        if pairwise_dists_tar_curr < pairwise_dists_tar_prev:
            reward += 0.1
            if pairwise_dists_tar_curr<=0.006:
                self.success_count += 1
        else:
            reward -= 0.1
        start = time.time()
        # 势能reward, 谁距离小，表示谁的势能高
        dist_tar_curr = distance.euclidean(targetState,xplaneState)
        dist_tar_prev = distance.euclidean(targetState,prev_xplaneState)
        # print('delta_dis',dist_tar_prev-dist_tar_curr)
        # print(dist_tar_curr)
        if dist_tar_curr < dist_tar_prev :  # 势能从低-->高
            reward += 0.1
            if dist_tar_curr<=80:
                reward += 1
                self.success_count += 1
        else:
            reward -= 0.1
        if self.success_count >=50:
            print('fly safe!!!')
            reward += 10
            self.ControlParameters.flag =True
        if xplaneState[4] < 50:
            reward += -10
            print('crash!!!')
            self.crash_flag = True
        if 100 < xplaneState[1] <= 180 or -180 < xplaneState[1] < -100 or 100 < xplaneState[2] <= 180 or -180 < \
                xplaneState[2] < -100:  # 如果坠机或者翻过来了
            reward += -1
            self.success_count = 0
            self.extreme_count += 1
            if self.extreme_count > 10:  # 连续10steps处于极端状态时，reset
                self.crash_flag = True
                reward += -10
                print('extreme state!!!')
        else:
            self.extreme_count = 0
        #
        # # 额外给roll增加权重
        # print(xplaneState[1])
        if abs(xplaneState[1]) <=5:
            reward += 0.05
        elif 5 < abs(xplaneState[1])<=10:
            reward += 0.02

        # 5/13
        # delta = [xplaneState[i] - targetState[i] for i in range(len(targetState))]
        # delta = tf.nn.l2_normalize(delta)
        # delta = delta.eval()
        # print(delta)
        # reward += -(0.1*abs(delta[1])+0.1*abs(delta[2])+0.01*abs(delta[3])+0.05*abs(delta[4])+0.05*abs(delta[5]))

        #################################################
        # dist = distance.euclidean(targetState[1:4], xplaneState[1:4])  # roll,pitch,yaw数值上的距离
        # if dist <= 50:
        #     self.success_count += 1
        #     self.extreme_count = 0
        #     reward += 2 * self.success_count / dist
        #     print(self.success_count)
        #
        #     if self.success_count > 50:  # 一个episode中有50step连续处于平稳飞行状态时，认为success
        #         self.ControlParameters.flag = True
        #         reward += 10 * self.success_count / dist
        #         self.success_count = 0
        # elif 50 < dist <= 100:
        #     reward += 3 / dist
        #     self.extreme_count = 0
        #     self.success_count = 0
        # elif dist > 100:
        #     reward += -0.01
        #     # self.extreme_count = 0
        #     self.success_count = 0
        # if -20 < xplaneState[4] - targetState[4] < 100:
        #     reward += 0.01
        return reward

    def step(self, actions):
        """
          obs, rew, done, info = env.step(actions)
        """
        self.ControlParameters.flag = False  # for synchronisation of training:  done
        #############################################
        # 查看 pevious action 与controls是否一致
        # print("previous action", actions)  # previous action
        # print("action on ctrl ...", XplaneEnv.CLIENT.getCTRL())  # action on control surface
        # 如果不一致，则要确保一致（通过频率调节）
        #############################################

        XplaneEnv.CLIENT.pauseSim(False)  # unpause x plane simulation
        '''
        sendCTRL():
          * Latitudinal Stick [-1,1]
          * Longitudinal Stick [-1,1]
          * Rudder Pedals [-1, 1]
          * Throttle [-1, 1]
          * Gear (0=up, 1=down)
          * Flaps [0, 1]
          * Speedbrakes [-0.5, 1.5]
        '''
        control = [actions[0],actions[1],actions[2],actions[3],-998,-998,-998]
        XplaneEnv.CLIENT.sendCTRL(control)  # send action

        # drefs_controls = ["sim/cockpit2/controls/yoke_pitch_ratio",
        #                   "sim/cockpit2/controls/yoke_roll_ratio",
        #                   "sim/cockpit2/controls/yoke_heading_ratio",
        #                   "sim/flightmodel/engine/ENGN_thro",
        #                   "sim/flightmodel/controls/flaprqst"]
        # # control = [[actions[0]], [actions[1]],[actions[2]],[actions[3]],[actions[4]]]
        # control = [actions[0], actions[1],actions[2],actions[3],actions[4]]
        # XplaneEnv.CLIENT.sendDREFs(drefs_controls, control)


        # sleep(0.003)  # sleep for a while so that action is executed
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
        velocity_name = ["sim/flightmodel/position/P","sim/flightmodel/position/Q","sim/flightmodel/position/R"]
        [P,Q,R] = XplaneEnv.CLIENT.getDREFs(velocity_name)
        # print(P)
        ##############################################################################

        self.ControlParameters.state8['roll_rate'] = P[0]  # The roll rotation rates (relative to the flight)
        self.ControlParameters.state8['pitch_rate'] = Q[0] # The pitch rotation rates (relative to the flight)
        self.ControlParameters.state8['yaw_rate'] = R[0]  # The yaw rotation rates (relative to the flight)
        self.ControlParameters.state8['Roll'] = state[3]  # roll
        self.ControlParameters.state8['Pitch'] = state[4]  # pitch
        self.ControlParameters.state8['Yaw'] = state[5]  # pitch
        self.ControlParameters.state8['altitude'] = state[2]  # Altitude
        self.ControlParameters.state8['airspeed'] = state[7]  # local velocity x  OpenGL coordinates

        state8 = [i for i in self.ControlParameters.state8.values()]
        # print('state8', state8)
        ###########################################################################
        # *******************************reward computation ******************
        targetState = [0, 0, 0, 0, 2500, 100]  # targetState[yaw_rate=0,roll=0,pitch=0,yaw=0,altitude=2500,airspeed=100]
        xplaneState = [state8[2], state8[3],state8[4],state8[5],state8[6],state8[7]]  # present situation
        reward = self.rewardCalcul(targetState, xplaneState,self.prev_xplaneState, actions)
        self.prev_xplaneState = xplaneState
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
        XplaneEnv.CLIENT.sendCOMM('sim/instruments/DG_sync_mag')  # vacuum DG sync to magnetic north.
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
                np.random.uniform(-45,45), np.random.uniform(-45,45), np.random.uniform(-45,45),
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
        self.prev_xplaneState = [values[i] for i in range(2,len(values))]
        self.success_count =0
        # self.close()

        return state  # self.ControlParameters.state8
