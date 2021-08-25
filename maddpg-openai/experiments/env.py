# !/usr/bin/python
# -*- coding: utf-8 -*-
import time
import random
import scipy.spatial.distance as distance
import numpy as np
import xplane_sim as sim
from math import radians, cos, sin, asin, sqrt
"""
test env
"""
import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import sys
import os
import math
import numpy as np
from numpy import pi
from numpy import random
import time
class XPlane:
    def __init__(self):
        self.env_type = 'continuous'
        self.name = 'xplane-v1.0'
        self.max_episode_steps = 1000
        self.obs_dim = 6  # 每个agent的observation的维度
        self.act_dim = 6  # action的维度(个数)
        self.n=1
        self.observation_space = []
        self.action_space = []
        self.action_space_shape = []
        self.u_range = 1
        self.observation_space.append(
            spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim,), dtype=np.float32))
        self.action_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.act_dim,), dtype=np.float32))

        self.obs = [0 for i in range(self.obs_dim)]
        self.prev_obs = [0 for i in range(self.obs_dim)]
        self.flight_origin = [37.524, -122.06899, 6000, -998, -998, -998]  # Palo Alto
        self.flight_destinaion = [37.505, -121.843611, 6000, -998, -998, -998]  # Sunol Valley

    def reset(self, pos):
        """
        Reset environment
        Usage:
            obs = env.reset()
        """
        self.obs=sim.reset(pos)
        return self.obs

    def step(self, action_n):
        """
        obs, rew, done, info = env.step(action_n)
        """
        print("\nStart environment step")
        self.prev_obs = self.obs  # make sure this happened after reward computing
        self._take_action(action_n)
        reward, done = self._compute_reward(action_n, self.prev_obs)
        self.obs = self._get_observation()
        info = 0

        return self.obs, reward, done,info

    def _get_observation(self):
        """
        Get observation of double_logger's state
        Args:
        Returns:
            obs: array([...pose+vel0...,pose+vell...pose+vel1...])
        """
        self.obs=sim.get_posi()
        return self.obs

    def _take_action(self, actions):

        actions[4]=0 if actions[4]<0.5 else 1  # gear起落架
        sim.send_ctrl(actions)


    def _compute_reward(self,action,obs):
        """
        Compute reward and done based on current status
        Return:
            reward
            done
        """
        reward, done = 0, False
        # if action == pitch up
        if action[0]>0:
            reward+=10.0 if obs[3]<0.0 else -10
        else:
            reward+=10.0 if obs[3]>0.0 else -10
        # if action == roll right
        if action[1]>0:
            reward+=10.0 if obs[4]<0.0 else -10
        else:
            reward+=10.0 if obs[4]>0.0 else -10
        # if action == rudder -
        obs[5] = sim.convert_action_to_control(obs[5])
        if action[2]>0:
            reward+=10.0 if obs[5]<0.0 else -10
        else:
            reward+=10.0 if obs[5]>0.0 else -10
        if obs==self.flight_destinaion:
            done=True

        return reward, done

if __name__ == "__main__":
    env = XPlane()
    num_steps = env.max_episode_steps
    obs = env.reset()
    ep, st = 0, 0
    o = env.reset()
    for t in range(num_steps):
        a = [np.random.randint(0, 4, size=5) for i in range(2)]
        o, r, d = env.step(a)
        st += 1
        if any(d):
            ep += 1
            st = 0
            obs = env.reset()