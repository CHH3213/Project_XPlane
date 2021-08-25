import gym
from gym.spaces import Box
import numpy as np
from DDPG_update2 import DDPG
from ou_noise import OUNoise
from env2 import XplaneEnv
from pid import xplanePID
import argparse
import os
import  tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()  # 可以用于从TensorFlow 1.x到2.x的复杂迁移项目的程序开头
import time
import scipy.io as sio


def norm(env, a):
    a_norm=[]
    for i in range(env.action_space.shape[0]):
        a_norm.append(-1. + (2 / (env.action_space.high[i] - env.action_space.low[i])) * (
                    a[i] - env.action_space.low[i]))
    return a_norm
def s_norm(env, obs):
    obs_norm=[]
    for i in range(env.observation_space.shape[0]):
        # print(obs[i])
        obs_norm.append(0. + (1) / (env.observation_space.high[i]-env.observation_space.low[i]) * (
                    obs[i] - env.observation_space.low[i]))
    return obs_norm
def decouple_norm(env,a_norm):
    a =[]
    for i in range(env.action_space.shape[0]):
        a.append((a_norm[i]-(-1))*((env.action_space.high[i]-env.action_space.low[i])/(1-(-1)))+env.action_space.low[i])
    return  a
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--episodes', default=400, type=int)
    parser.add_argument("--max_steps", type=int, default=800, help="maximum max_steps length")  # 每个episode的步数为400步
    parser.add_argument('--saveData_dir', default="./save/xplane_ddpg_3/data",
                        help="directory to store all experiment data")
    parser.add_argument('--saveModel_dir', default='./save/xplane_ddpg_3',
                        help="where to store/load network weights")
    parser.add_argument('--load_dir', default='./save/xplane_ddpg_3',
                        help="where to load network weights")

    parser.add_argument('--checkpoint_frequency', default=10, type=int,
                        help="how often to checkpoint")
    parser.add_argument('--testing', default=True, action="store_true",
                        help="reduces exploration substantially")
    parser.add_argument("--restore", default=False, action="store_true")
    parser.add_argument('--memory_size', default=100000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--batch_norm', default=True, action="store_true")
    parser.add_argument('--ou_mu', default=0, type=float,
                        help="OrnsteinUhlenbeckActionNoise mus for each action for each agent")
    parser.add_argument('--ou_theta', default=0.15, type=float,
                        help="OrnsteinUhlenbeckActionNoise theta for each agent")
    parser.add_argument('--ou_sigma', default=0.3, type=float,
                        help="OrnsteinUhlenbeckActionNoise sigma for each agent")
    args = parser.parse_args()

    # session = tf.Session()
    env= XplaneEnv()
    # env = gym.make('Pendulum-v0')
    max_steps= args.max_steps #max_steps per episode
    assert isinstance(env.observation_space, Box), "observation space must be continuous"
    assert isinstance(env.action_space, Box), "action space must be continuous"
    
    # Randomly initialize critic,actor,target critic, target actor network  and replay buffer
    # session.run(tf.initialize_all_variables())
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high
    agent = DDPG(a_dim, s_dim, a_bound, args.memory_size, args.batch_size, args.learning_rate)
    exploration_noise = OUNoise(a_dim,args.ou_mu,args.ou_theta,args.ou_sigma)

    reward_per_episode = 0    
    total_reward=0  
    print("Number of States:", s_dim)
    print("Number of Actions:", a_dim)
    print("Number of Steps per episode:", max_steps)
    #saving reward:
    reward_st = np.array([0])

    saver = agent.saver
    # Load previous results, if necessary
    if args.load_dir == "":
        args.load_dir = args.saveModel_dir
    if args.testing or args.restore:
        print("===============================")
        print('Loading previous state...')

        saver.restore(agent.sess, args.load_dir+'/xplane')
    start_time = time.time()
    # 存储数据列表
    reward_each_episode = []
    action_each_episode = []
    VAR = 3
    # 目标值
    target = [0, 0, 0, 0, 2500, 100]  # heading_rate roll pitch heading altitude KIAS
    Xplane_pid = xplanePID(target)  # heading_rate roll pitch heading altitude KIAS
    for episode in range(args.episodes):
        print("==== Starting episode no:",episode,"====","\n")
        observation = env.reset()
        observation = s_norm(env,observation)
        observation = np.array(observation)
        # observation = tf.nn.l2_normalize(observation)
        # observation = observation.eval()
        reward_per_episode = 0  # 用来存储一个episode的总和
        # 存储每个episode的数据
        reward_steps=[]
        action_steps =[]
        steps = 0
        for t in range(max_steps):
            #rendering environmet (optional)
            # env.render()  # test
            x = observation
            action = agent.choose_action(x)
            # print(type(action))
            # print(action)
            # action = Xplane_pid.cal_actions(observation)

            # print('action_beforenoise',action)
            noise = exploration_noise.noise()
            # print('noise',noise)
            # print('action[0]',action[0])
            # action = action[0] + 0.1*noise #Select action according to current policy and exploration noise


            action = np.clip(np.random.normal(action, VAR), -0.5, 0.5)
            # print(type(action))
            action = decouple_norm(env,action)
            print("Action at step", t ," :",action,"\n")
            start = time.time()
            observation,reward,done,info=env.step(action)

            observation = s_norm(env, observation)
            observation = np.array(observation)

            # observation = tf.nn.l2_normalize(observation)
            # observation = observation.eval()
            # print('observation', observation)
            # 每个episode、每一步的数据
            reward_steps.append(reward)
            action_steps.append(action)
            # print(reward)
            #add s_t,s_t+1,action,reward to experience memory
            agent.store_transition(x,action,reward, observation)
            #train critic and actor network
            if not args.testing:
                # print(agent.pointer)
                if agent.pointer > args.memory_size/10:
                    VAR *= .95  # decay the action randomness
                if agent.pointer > args.batch_size:
                    # print(agent.pointer)
                    agent.learn()
                    # print('train_action', action)
            reward_per_episode+=reward
            #check if episode ends:
            # print(t)
            # print(done)
            if (done or (t == max_steps-1) or env.crash_flag):
            # if (done or (t == max_steps-1) ):
                print('EPISODE:  ',episode,' Steps: ',t,' Total Reward: ',reward_per_episode)
                print("Printing reward to file")
                exploration_noise.reset() #reinitializing random noise for action exploration
                reward_st = np.append(reward_st,reward_per_episode)

                reward_each_episode.append(reward_steps)
                action_each_episode.append(action_steps)
                np.savetxt('episode_reward.txt',reward_st, newline="\n")
                print('\n\n')
                steps = t
                env.crash_flag = False
                break
            # print('a step time__', time.time() - start)
        if not args.testing:
            if (episode+1) % args.checkpoint_frequency ==0:
                if not os.path.exists(args.saveModel_dir):
                    os.makedirs(args.saveModel_dir)

                save_path = saver.save(agent.sess,args.saveModel_dir+'/xplane')
                print("saving model to {}".format(save_path))
        print("steps: {}, episodes: {}, episode reward: {}, time: {} \n".format(
            steps, episode, reward_per_episode, round(time.time() - start_time, 3)))
    # 保存数据

    if not os.path.exists(args.saveData_dir):
        os.makedirs(args.saveData_dir)
    sio.savemat(args.saveData_dir + '/data.mat', {'episode_reward': reward_each_episode,
                                                  'apisode_action':action_each_episode})
    total_reward+=reward_per_episode
    print("Average reward per episode {}".format(total_reward / args.episodes))
    print("Finished {} episodes in {} seconds".format(args.episodes,
                                                      time.time() - start_time))


if __name__ == '__main__':
    main()    