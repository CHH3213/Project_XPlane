import gym
from gym.spaces import Box
import numpy as np
from ddpg import DDPG
from ou_noise import OUNoise
#specify parameters here:
is_batch_norm = False #batch normalization switch
import test

from env2 import XplaneEnv
import argparse
import os
import  tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()  # 可以用于从TensorFlow 1.x到2.x的复杂迁移项目的程序开头
import time
import scipy.io as sio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--episodes', default=1000, type=int)
    parser.add_argument("--max_steps", type=int, default=200, help="maximum max_steps length")  # 每个episode的步数为400步
    parser.add_argument('--saveData_dir', default="./save/xplane_ddpg_4/data",
                        help="directory to store all experiment data")
    parser.add_argument('--saveModel_dir', default='./save/xplane_ddpg_4',
                        help="where to store/load network weights")
    parser.add_argument('--load_dir', default='./save/xplane_ddpg_4',
                        help="where to load network weights")

    parser.add_argument('--checkpoint_frequency', default=10, type=int,
                        help="how often to checkpoint")
    parser.add_argument('--testing', default=False, action="store_true",
                        help="reduces exploration substantially")
    parser.add_argument("--restore", default=True, action="store_true")
    parser.add_argument('--memory_size', default=10000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--batch_norm', default=True, action="store_true")
    parser.add_argument('--ou_mu', default=0, type=float,
                        help="OrnsteinUhlenbeckActionNoise mus for each action for each agent")
    parser.add_argument('--ou_theta', default=0.15, type=float,
                        help="OrnsteinUhlenbeckActionNoise theta for each agent")
    parser.add_argument('--ou_sigma', default=0.3, type=float,
                        help="OrnsteinUhlenbeckActionNoise sigma for each agent")
    args = parser.parse_args()

    session = tf.InteractiveSession()
    env= XplaneEnv()
    max_steps= args.max_steps  # max_steps per episode
    assert isinstance(env.observation_space, Box), "observation space must be continuous"
    assert isinstance(env.action_space, Box), "action space must be continuous"
    
    # Randomly initialize critic,actor,target critic, target actor network  and replay buffer
    # session.run(tf.initialize_all_variables())
    agent = DDPG(env, args.batch_norm, args.batch_size, args.memory_size, sess=session, lr =args.learning_rate)
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
    # Load previous results, if necessary
    if args.load_dir == "":
        args.load_dir = args.saveModel_dir
    if args.testing or args.restore:
        print("===============================")
        print('Loading previous state...')

        saver.restore(session, args.load_dir+'/xplane')
    start_time = time.time()
    # 存储数据列表
    reward_each_episode = []
    action_each_episode = []

    # 目标值
    target = [0, 0, 0, 0, 2500, 100]  # heading_rate roll pitch heading altitude KIAS
    Xplane_pid = test.xplanePID(target)  # heading_rate roll pitch heading altitude KIAS
    # env.set_frame_rate(0.05)
    for episode in range(args.episodes):
        print("==== Starting episode no:",episode,"====","\n")
        observation = env.reset()
        # observation = tf.nn.l2_normalize(observation)
        # observation = observation.eval()
        reward_per_episode = 0  # 用来存储一个episode的总和
        # 存储每个episode的数据
        reward_steps=[]
        action_steps =[]
        steps = 0
        for t in range(max_steps):
            #rendering environmet (optional)
            x = observation
            # action = agent.evaluate_actor(np.reshape(x,[1,num_states]))
            action = Xplane_pid.cal_actions(observation)
            action =[action]
            # print('action_beforenoise',action)
            noise = exploration_noise.noise()
            # print('noise',noise)
            # print('action[0]',action[0])

            action = action[0] + 0.1*noise #Select action according to current policy and exploration noise

            # action[0] = np.clip(action[0],-1,1)
            # action[1] = np.clip(action[1],-1,1)
            # action[2] = np.clip(action[2],-1,1)
            # action[3] = np.clip(action[3],0,1)
            # action[4] = np.clip(action[4],-0.5,1.5)

            VAR=1
            # action = np.clip(np.random.normal(action, VAR), -0.5, 0.5)

            # print("Action at step", t ," :",action,"\n")
            # print('action_pre',action)
            observation,reward,done,info=env.step(action)
            # observation = [1.0 / (1 + np.exp(-float(x))) for x in observation]  # 正则化，利用sigmoid函数将状态值映射到（0，1）
            # observation = tf.nn.l2_normalize(observation)
            # observation = observation.eval()
            # print('observation', observation)
            # 每个episode、每一步的数据
            reward_steps.append(reward)
            action_steps.append(action)
            #add s_t,s_t+1,action,reward to experience memory
            agent.add_experience(x,observation,action,reward,done)
            #train critic and actor network
            if not args.testing:
                if counter > 64:
                    agent.train()
                    # print('train_action', action)
            reward_per_episode+=reward
            counter+=1
            #check if episode ends:
            if (done or (t == max_steps-1) ):
                # print(done,t,env.crash_flag)
                print('EPISODE:  ',episode,' Steps: ',t,' Total Reward: ',reward_per_episode)
                print("Printing reward to file")
                exploration_noise.reset() #reinitializing random noise for action exploration
                reward_st = np.append(reward_st,reward_per_episode)
                
                reward_each_episode.append(reward_steps)
                action_each_episode.append(action_steps)
                # np.savetxt('episode_reward.txt',reward_st, newline="\n")
                print('\n\n')
                steps = t
                env.crash_flag = False
                break
        if not args.testing:
            if episode % args.checkpoint_frequency ==0:
                if not os.path.exists(args.saveModel_dir):
                    os.makedirs(args.saveModel_dir)

                save_path = saver.save(session,args.saveModel_dir+'/xplane')
                print("saving model to {}".format(save_path))
        # print("steps: {}, episodes: {}, episode reward: {}, time: {} \n".format(
        #     steps, episode, reward_per_episode, round(time.time() - start_time, 3)))
    # 保存数据
    if not os.path.exists(args.saveData_dir):
        os.makedirs(args.saveData_dir)
    sio.savemat(args.saveData_dir + '/data.mat', {'episode_reward': reward_each_episode,
                                                  'episode_action':action_each_episode})
    total_reward+=reward_per_episode
    print("Average reward per episode {}".format(total_reward / args.episodes))
    print("Finished {} episodes in {} seconds".format(args.episodes,
                                                      time.time() - start_time))


if __name__ == '__main__':
    main()