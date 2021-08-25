#Implementation of Deep Deterministic Gradient with Tensor Flow"
# Author: Steven Spielberg Pon Kumar (github.com/stevenpjg)

import gym
from gym.spaces import Box, Discrete
import numpy as np
from ddpg import DDPG
from ou_noise import OUNoise
#specify parameters here:
episodes=10000
is_batch_norm = False #batch normalization switch

from env2 import XplaneEnv
import argparse
import os
import  tensorflow as tf
import time
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--episodes', default=1000, type=int)
    parser.add_argument("--max_steps", type=int, default=200, help="maximum max_steps length")  # 每个episode的步数为400步
    parser.add_argument('--saveData_dir', default="./save/xplane_ddpg/data",
                        help="directory to store all experiment data")
    parser.add_argument('--saveModel_dir', default='./save/xplane_ddpg',
                        help="where to store/load network weights")

    parser.add_argument('--checkpoint_frequency', default=100, type=int,
                        help="how often to checkpoint")
    parser.add_argument('--testing', default=False, action="store_true",
                        help="reduces exploration substantially")
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument('--load_dir', default='./save/xplane_ddpg',
                        help="where to load network weights")
    parser.add_argument('--memory_size', default=10000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--ou_mu', default=0, type=float,
                        help="OrnsteinUhlenbeckActionNoise mus for each action for each agent")
    parser.add_argument('--ou_theta', default=0.15, type=float,
                        help="OrnsteinUhlenbeckActionNoise theta for each agent")
    parser.add_argument('--ou_sigma', default=0.3, type=float,
                        help="OrnsteinUhlenbeckActionNoise sigma for each agent")
    args = parser.parse_args()

    session = tf.InteractiveSession()
    env= XplaneEnv()
    max_steps= args.max_steps #max_steps per episode
    assert isinstance(env.observation_space, Box), "observation space must be continuous"
    assert isinstance(env.action_space, Box), "action space must be continuous"
    
    #Randomly initialize critic,actor,target critic, target actor network  and replay buffer   
    session.run(tf.initialize_all_variables())
    agent = DDPG(env, is_batch_norm, args.batch_size, args.memory_size, sess=session)
    exploration_noise = OUNoise(env.action_space.shape[0],args.ou_mu,args.ou_theta,args.ou_sigma)
    counter=0
    reward_per_episode = 0    
    total_reward=0
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]    
    print("Number of States:", num_states)
    print("Number of Actions:", num_actions)
    print("Number of Steps per episode:", max_steps)
    #saving reward:
    reward_st = np.array([0])

    saver = tf.train.Saver(max_to_keep=1)
    if args.testing or args.restore:
        # os.path.join(args.load_dir, filename)
        saver.restore(session, args.load_dir)
    start_time = time.time()
    for episode in range(episodes):
        print("==== Starting episode no:",episode,"====","\n")
        observation = env.reset()
        reward_per_episode = 0
        steps = 0
        for t in range(max_steps):
            #rendering environmet (optional)
            x = observation
            action = agent.evaluate_actor(np.reshape(x,[1,num_states]))
            noise = exploration_noise.noise()
            action = action[0] + noise #Select action according to current policy and exploration noise
            print("Action at step", t ," :",action,"\n")
            
            observation,reward,done,info=env.step(action)
            
            #add s_t,s_t+1,action,reward to experience memory
            agent.add_experience(x,observation,action,reward,done)
            #train critic and actor network
            if counter > 64: 
                agent.train()
            reward_per_episode+=reward
            counter+=1
            #check if episode ends:
            if (done or (t == max_steps-1)):
                print('EPISODE: %d ',episode,' Steps:%d ',t,' Total Reward: %.3f',reward_per_episode)
                print("Printing reward to file")
                exploration_noise.reset() #reinitializing random noise for action exploration
                reward_st = np.append(reward_st,reward_per_episode)
                np.savetxt('episode_reward.txt',reward_st, newline="\n")
                print('\n\n')
                steps = t
                break
            if episode % args.checkpoint_frequency ==0:
                if not os.path.exists(args.saveModel_dir):
                    os.makedirs(args.saveModel_dir)
                save_path = saver.save(session,args.saveModel_dir,global_step=episode)
                print("saving model to {}".format(save_path))
        print("steps: {}, episodes: {}, episode reward: {}, time: {} \n".format(
            steps, episode, reward_per_episode, round(time.time() - start_time, 3)))

    total_reward+=reward_per_episode
    print("Average reward per episode {}".format(total_reward / episodes))
    print("Finished {} episodes in {} seconds".format(args.episodes,
                                                      time.time() - start_time))


if __name__ == '__main__':
    main()    