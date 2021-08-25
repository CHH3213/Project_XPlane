import gym
from gym.spaces import Box
import numpy as np
from TD3 import TD3
from TD3 import ReplayBuffer
from ou_noise import OUNoise
from env2_2 import XplaneEnv
from pid2 import xplanePID
import argparse
import os
import  tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()  # 可以用于从TensorFlow 1.x到2.x的复杂迁移项目的程序开头
import time
import scipy.io as sio

'''测试git'''
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
    parser.add_argument('--episodes', default=10, type=int)
    parser.add_argument("--max_steps", type=int, default=500, help="maximum max_steps length")  # 每个episode的步数为400步
    parser.add_argument('--saveData_dir', default="./save/xplane_TD3_pretrain6/data",
                        help="directory to store all experiment data")
    parser.add_argument('--saveModel_dir', default='./save/xplane_TD3_pretrain6',
                        help="where to store/load network weights")
    parser.add_argument('--load_dir', default='./save/xplane_TD3_pretrain6',
                        help="where to load network weights")

    parser.add_argument('--checkpoint_frequency', default=100, type=int,
                        help="how often to checkpoint")
    parser.add_argument('--testing', default=True, action="store_true",
                        help="reduces exploration substantially")
    parser.add_argument("--restore", default=True, action="store_true")
    parser.add_argument('--memory_size', default=100000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
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
    HIDDEN_DIM = 64
    POLICY_TARGET_UPDATE_INTERVAL = 3  # delayed steps for updating the policy network and target networks
    replay_buffer = ReplayBuffer(args.memory_size)
    agent = TD3(s_dim, a_dim, a_bound, HIDDEN_DIM, replay_buffer, POLICY_TARGET_UPDATE_INTERVAL, args.learning_rate, args.learning_rate)
    # exploration_noise = OUNoise(a_dim,args.ou_mu,args.ou_theta,args.ou_sigma)

    reward_per_episode = 0    
    total_reward=0  
    print("Number of States:", s_dim)
    print("Number of Actions:", a_dim)
    print("Number of Steps per episode:", max_steps)
    #saving reward:
    reward_st = np.array([0])

    start_time = time.time()
    # 存储数据列表
    reward_each_episode = []
    action_each_episode = []
    state_each_episode = []
    VAR = 1
    # 目标值
    target = [0,0,0, 0, 0, 0, 2500, 100]  # heading_rate roll pitch heading altitude KIAS
    Xplane_pid = xplanePID(target)  # heading_rate roll pitch heading altitude KIAS
    state = env.reset()
    state = np.array(state)
    state = state.astype(np.float32)
    agent.policy_net([state])
    agent.target_policy_net([state])
    frame_idx = 0
    ######################Hyperameters#########################
    EXPLORE_STEPS = 1000
    EXPLORE_NOISE_SCALE = 1.0  # range of action noise for exploration
    UPDATE_ITR = 3  # repeated updates for single step
    EVAL_NOISE_SCALE = 0.5  # range of action noise for evaluation of action value
    REWARD_SCALE = 1.  # value range of reward
    #################################################################
    t_state = time.time()
    if args.restore:
        agent.load(args.load_dir)
    if not args.testing:
        print('train!!!!!!!!!!!!!!!!')
        for episode in range(args.episodes):
            print("==== Starting episode no:",episode,"====","\n")
            observation = env.reset()
            observation = np.array(observation)
            observation = observation.astype(np.float32)
            # observation = s_norm(env,observation)
            # observation = np.array(observation)
            reward_per_episode = 0  # 用来存储一个episode的总和
            # 存储每个episode的数据
            reward_steps=[]
            action_steps =[]
            obs_steps = []
            steps = 0
            # reward_st = np.array([0])
            for t in range(max_steps):
                #rendering environmet (optional)
                # env.render()  # test
                state = observation
                # print(state)
                if frame_idx > EXPLORE_STEPS:
                    action = agent.policy_net.get_action(state, EXPLORE_NOISE_SCALE)

                    # print('train!!')

                else:
                    action = agent.policy_net.sample_action()
                # action = agent.policy_net.get_action(state, EXPLORE_NOISE_SCALE)

                # if episode >20:
                #     action = agent.policy_net.get_action(state, EXPLORE_NOISE_SCALE)
                # else:
                #     action = Xplane_pid.cal_actions(state)

                # action = np.clip(np.random.normal(action, VAR), -1, 1)
                # action[0] = np.clip(action[0], -1, 1)
                # action[1] = np.clip(action[1], -1, 1)
                # action[2] = np.clip(action[2], -1, 1)
                # action[3] = np.clip(action[3], 0, 1)
                # print(type(action))
                # action_env = decouple_norm(env,action)
                print("Action at step", t, " :", action, "\n")
                start = time.time()

                # observation,reward,done,info=env.step(action_env)
                observation,reward,done,info=env.step(action)

                # observation = s_norm(env, observation)
                # observation = np.array(observation)
                observation = np.array(observation)
                observation = observation.astype(np.float32)  # 转换数组的类型
                # print('observation', observation)
                # 每个episode、每一步的数据
                reward_steps.append(reward)
                action_steps.append(action)
                obs_steps.append(observation)
                # print(reward)
                #add s_t,s_t+1,action,reward to experience memory
                replay_buffer.push(state, action, reward, observation, done)
                if len(replay_buffer) > args.batch_size:
                    for i in range(UPDATE_ITR):
                        # train critic and actor network
                        agent.update(args.batch_size, EVAL_NOISE_SCALE, REWARD_SCALE)

                reward_per_episode+=reward
                frame_idx += 1
                #check if episode ends:
                # print(t)
                # print(done)
                if (done or (t == max_steps-1) or env.crash_flag):
                    # if (done or (t == max_steps-1) ):
                    print('EPISODE:  ',episode,' Steps: ',t,' Total Reward: ',reward_per_episode)
                    print("Printing reward to file")
                    # exploration_noise.reset() #reinitializing random noise for action exploration
                    reward_st = np.append(reward_st,reward_per_episode)

                    reward_each_episode.append(reward_steps)
                    action_each_episode.append(action_steps)
                    state_each_episode.append(obs_steps)
                    np.savetxt('xplane_TD3_pretrain6.txt',reward_st, newline="\n")
                    # np.savetxt('episode_reward.txt',reward_st, newline="\n")
                    print('\n\n')
                    steps = t
                    env.crash_flag = False
                    break
                # print('a step time__', time.time() - start)

            if (episode+1) % args.checkpoint_frequency ==0:
                if not os.path.exists(args.saveModel_dir):
                    os.makedirs(args.saveModel_dir)

                agent.save(args.saveModel_dir)
                print("saving model to {}".format(args.saveModel_dir))
            print("steps: {}, episodes: {}, episode reward: {}, time: {} \n".format(
                steps, episode, reward_per_episode, round(time.time() - start_time, 3)))
        if not os.path.exists(args.saveData_dir):
            os.makedirs(args.saveData_dir)
        sio.savemat(args.saveData_dir + '/data.mat', {'episode_reward': reward_each_episode,
                                                      'episode_action': action_each_episode,
                                                      'episode_state': state_each_episode})
        # 保存数据
        total_reward += reward_per_episode
        print("Average reward per episode {}".format(total_reward / args.episodes))
        print("Finished {} episodes in {} seconds".format(args.episodes,
                                                          time.time() - start_time))
    else:
        print('test!!!!!!!!!!!!!!!!')
        agent.load(args.load_dir)
        # state = env.reset()
        state = env.reset_spin()
        state = np.array(state)
        state = state.astype(np.float32)
        agent.policy_net([state])
        for episode in range(args.episodes):
            # state = env.reset()
            state = env.reset_spin()
            state = np.array(state)
            state = state.astype(np.float32)
            episode_reward = 0
            start = time.time()
            # 存储每个episode的数据
            reward_steps = []
            action_steps = []
            obs_steps = []
            steps = 0
            for step in range(max_steps):
                action = agent.policy_net.get_action(state, EXPLORE_NOISE_SCALE, greedy=True)
                # action = Xplane_pid.cal_actions(state)
                if step ==0:
                    action = [1,1,0,1]
                state, reward, done, info = env.step(action)
                state = state.astype(np.float32)
                print(state[6])
                episode_reward += reward
                # print('observation', observation)
                # 每个episode、每一步的数据
                reward_steps.append(reward)
                action_steps.append(action)
                obs_steps.append(state)
                # if (done or (step == max_steps - 1) or env.crash_flag):
                # if (done or (step == max_steps-1) ):
                time.sleep(0.01)
                if step == 0 :
                    time.sleep((6))
                if ((step == max_steps-1) ):
                    print("Printing reward to file")
                    reward_st = np.append(reward_st, reward_per_episode)
                    env.crash_flag = False
                    print(time.time()-start)
                    state_each_episode.append(obs_steps)
                    # print('aaa', obs_steps)
                    break
            print(
                'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode + 1, args.episodes, episode_reward,
                    time.time() - t_state
                )
            )


        sio.savemat(args.saveData_dir + '/pid_nws.mat', {'episode_state': state_each_episode})
        # sio.savemat(args.saveData_dir + '/TD3_w.mat', {'episode_state': state_each_episode})




if __name__ == '__main__':
    main()
