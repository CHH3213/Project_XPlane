# import struct
import xpc_ as xpc
import numpy as np
import math
from time import sleep
from time import time


class XplaneEnv:
    def __init__(self, xpHost='127.0.0.1', xpPort=49009, clientPort=1, timeout=1000):
        self.states8 = []  # el,r,p,y,P,Q,R,KIAS
        self.veldefs = ['sim/flightmodel/position/local_vx',
                   'sim/flightmodel/position/local_vy',
                   'sim/flightmodel/position/local_vz',
                   'sim/flightmodel/position/P',
                   'sim/flightmodel/position/Q',
                   'sim/flightmodel/position/R']
        self.posidefs = []
        try:
            self.client = xpc.XPlaneConnect(xpHost, xpPort, clientPort, timeout)
        except:
            print('error parameter')
        print('I am client', self.client)

    def close(self):
        self.client.close()

    def reset(self,POSI,VEL):
        roll = POSI[3] / 180 * np.pi
        pitch = POSI[4] / 180 * np.pi
        heading = POSI[5] / 180 * np.pi
        Rr = np.array([[np.cos(roll), np.sin(roll), 0],
                       [-np.sin(roll), np.cos(roll), 0],
                       [0, 0, 1]])
        Rp = np.array([[1, 0, 0],
                       [0, np.cos(pitch), -np.sin(pitch)],
                       [0, np.sin(pitch), np.cos(pitch)]])
        Rh = np.array([[np.cos(heading), 0, -np.sin(heading)],
                       [0, 1, 0],
                       [np.sin(heading), 0, np.cos(heading)]])
        z_unit = np.array([0, 0, -1])
        UnitVec = np.einsum('ij,jk,kl,l->i', Rh, Rp, Rr, z_unit)*0.51444
        vel = [VEL[0] * UnitVec[0], VEL[0] * UnitVec[1], VEL[0] * UnitVec[2],
                VEL[1], VEL[2], VEL[3]]

        self.client.sendCOMM('sim/operation/reset_flight')  # reset plane
        self.client.sendCOMM('sim/view/circle')  # set the external view
        self.client.sendPOSI(POSI)
        self.client.sendDREFs(self.veldefs,vel)
        self.client.sendCOMM('sim/instruments/DG_sync_mag')  # vacuum DG sync to magnetic north.

    def random_reset(self):
        POSI = [0, 0, 2500, np.random.uniform(-180,180), np.random.uniform(-180,180), np.random.uniform(-180,180), -998]
        VEL = [np.random.uniform(-20,20)+100,np.random.uniform(-10,10),np.random.uniform(-10,10),np.random.uniform(-10,10)]
        self.reset(POSI,VEL)
        return POSI,VEL

    def get_states(self):
        position = self.client.getPOSI()
        [vx,vy,vz,P,Q,R,KIAS,has_crashed] = self.client.getDREFs(self.veldefs +
                                                                 ['sim/flightmodel/position/indicated_airspeed',
                                                                  'sim/flightmodel2/misc/has_crashed'])
        self.states8 = [P[0],Q[0],R[0],
                        position[4],position[3],position[5],position[2],
                        KIAS[0]]
        return self.states8,has_crashed[0]

    def set_frame_rate(self,timestep=0.05):
        datarefs = ['sim/operation/misc/frame_rate_period', 'sim/operation/override/override_timestep']
        # enable override time step
        self.client.sendDREF(datarefs[1], 0)
        # set time step
        self.client.sendDREF(datarefs[0], timestep)
        # client.sendDREF(datarefs[1], 0)
        data = self.client.getDREFs(datarefs)
        print("frame_rate_period:{}\ttime_step_enable:{}".format(data[0][0], data[1][0]))

    def step(self,actions):
        self.client.sendCTRL(actions)

# 位置式
class PID_posi:
    def __init__(self,kp,ki,kd,target,up = 1.,low = -1.):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.err = 0
        self.err_last = 0
        self.err_all = 0
        self.target = target
        self.up = up
        self.low = low
        self.value = 0

    def increase(self,state):
        self.err = self.target - state
        self.value = self.kp*self.err + self.ki * self.err_all + self.kd*(self.err-self.err_last)
        self.update()

    def update(self):
        self.err_last = self.err
        self.err_all = self.err_all + self.err
        if self.value > self.up:
            self.value = self.up
        elif self.value < self.low:
            self.value = self.low

    def auto_adjust(self,Kpc,Tc):
        self.kp = Kpc*0.6
        self.ki = self.kp/(0.5*Tc)
        self.kd = self.kp*(0.125*Tc)
        return self.kp,self.ki,self.kd

    def set_pid(self,kp,ki,kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def reset(self):
        self.err = 0
        self.err_last = 0
        self.err_all = 0

    def set_target(self,target):
        self.target = target

# 增量式
class PID_inc:
    def __init__(self,kp,ki,kd,target,up = 1.,low = -1.):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.err = 0
        self.err_last = 0
        self.err_ll = 0
        self.target = target
        self.up = up
        self.low = low
        self.value = 0
        self.inc = 0

    def increase(self,state):
        self.err = self.target - state
        self.inc = self.kp*(self.err - self.err_last) + self.ki * self.err + self.kd*(self.err-2*self.err_last+self.err_ll)
        self.update()

    def update(self):
        self.err_last = self.err
        self.err_ll = self.err_last
        self.value = self.value+self.inc
        if self.value > self.up:
            self.value = self.up
        elif self.value < self.low:
            self.value = self.low

    def auto_adjust(self,Kpc,Tc):
        self.kp = Kpc*0.6
        self.ki = self.kp/(0.5*Tc)
        self.kd = self.kp*(0.125*Tc)
        return self.kp,self.ki,self.kd

    def set_pid(self,kp,ki,kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def reset(self):
        self.err = 0
        self.err_last = 0
        self.err_ll = 0


class xplanePID:
    def __init__(self,target):
        self.pi_c = PID_posi(0.1, 0, 0.001, target[4], up=45, low=-45)
        self.el = PID_posi(0.03, 0., 0.002, 0, up=0.8, low=-0.5)  # control altitude
        self.ai = PID_posi(0.02, 0.001, 0.01, target[1])  # control roll
        self.ru = 0  # control heading
        self.th = PID_posi(0.05, 0.0001, 0.01, target[5], low=0)  # control KIAS
        self.gear = -998
        self.flaps = -998
        # sb = PID_inc(0.01,0,0.01,-target[5],up=1.5,low=-0.5)  # control KIAS

    def cal_actions(self,states8):
        self.pi_c.increase(states8[6])
        self.el.set_target(self.pi_c.value)
        self.el.increase(states8[4])
        self.ai.increase(states8[3])
        self.ru = 0 # 改为平稳飞行时的
        self.th.increase(states8[7])
        # sb.increase(-states8[7])
        return [self.el.value,self.ai.value,self.ru,self.th.value]

    def reset(self):
        self.pi_c.reset()
        self.el.reset()
        self.ai.reset()
        self.ru.reset()
        self.th.reset()
        # sb.reset()



if __name__ == "__main__":
    env = XplaneEnv()  # environment
    target = [0,0,0,0,2500,100]  # heading_rate roll pitch heading altitude KIAS

    # while True:
    #     posi = env.client.getPOSI()  # position
    #     print(posi)

    # actions = [elevator,aileron,rudder,throttle,gear,flaps,speed brake]
    # pi_c = PID_posi(0.1,0,0.001,target[4],up=45,low=-45)
    # el = PID_posi(0.03,0.,0.002,0,up=0.8,low=-0.5)  # control altitude
    # ai = PID_posi(0.02,0.001,0.01,target[1])  # control roll
    # ru = PID_posi(0.0001,0,0.001,target[3])  # control heading
    # th = PID_posi(0.05,0.0001,0.01,target[5],low=0)  # control KIAS
    # gear = -998
    # flaps = -998
    # sb = PID_inc(0.01,0,0.01,-target[5],up=1.5,low=-0.5)  # control KIAS
    xplane_pid = xplanePID(target)

    init_POSI = [0,0,1000,0,45,0,0]
    init_VEL = [100,0,0,0]
    # env.reset(init_POSI,init_VEL)
    env.random_reset()
    env.set_frame_rate(0.05)
    states8 = []
    while True:
        states8,has_crashed = env.get_states()  # P,Q,R,r,p,y,el,KIAS
        if has_crashed:
            # env.reset(init_POSI,init_VEL)
            env.random_reset()
            xplane_pid.reset()
            continue
        print('state:\n{}'.format(states8))
        action = xplane_pid.cal_actions(states8)
        env.client.sendCTRL(action+[-998,-998,-998])
        print('actons:\n {}'.format(action))
        sleep(0.05)

        # 改4维
