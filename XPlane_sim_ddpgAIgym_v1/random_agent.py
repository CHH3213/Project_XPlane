import argparse

import xpc
import gym
from env2 import XplaneEnv
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()
	
def __init__():
	pass

if __name__ == '__main__':
    env=XplaneEnv()
    
    agent = RandomAgent(env.action_space)
    
    
    episodes = 0
    while episodes < 100:
        obs = env.reset()
        done = False
        while not done:
            action = agent.act()
            obs, reward, done, _ = env.step(action) 

            print(obs, reward, done)
            
        episodes += 1

    env.close()
