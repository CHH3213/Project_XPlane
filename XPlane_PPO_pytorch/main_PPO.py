import gym
from gym.spaces import Box
import numpy as np
from ppo import ppo
from env2_2 import XplaneEnv
from pid import xplanePID
import argparse
import os
import torch
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
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--layers', '-l', type=int, default=2)
    parser.add_argument('--pi_lr', '-pi', type=float, default=3e-4)
    parser.add_argument('--v_lr', '-v', type=float, default=1e-3)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--train_pi_iters', type=int, default=80)
    parser.add_argument('--train_v_iters', type=int, default=80)
    parser.add_argument('--target_kl', type=int, default=0.01)
    parser.add_argument('--saveData_dir', default="./save/xplane_PPO_1/data",
                        help="directory to store all experiment data")
    parser.add_argument('--saveModel_dir', default='./save/xplane_PPO_1',
                        help="where to store/load network weights")
    parser.add_argument('--load_dir', default='./save/xplane_PPO_1',
                        help="where to load network weights")

    parser.add_argument('--checkpoint_frequency', default=10, type=int,
                        help="how often to checkpoint")
    parser.add_argument('--testing', default=False, action="store_true",
                        help="reduces exploration substantially")
    parser.add_argument("--restore", default=False, action="store_true")
    parser.add_argument('--memory_size', default=100000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()

    # session = tf.Session()
    env= XplaneEnv()
    # env = gym.make('Pendulum-v0')
    max_steps= args.steps #max_steps per episode
    assert isinstance(env.observation_space, Box), "observation space must be continuous"
    assert isinstance(env.action_space, Box), "action space must be continuous"
    
    # Randomly initialize critic,actor,target critic, target actor network  and replay buffer
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    # exploration_noise = OUNoise(a_dim,args.ou_mu,args.ou_theta,args.ou_sigma)
    agent = ppo(lambda: env, hid=args.hid, layers=args.layers, seed=args.seed, lam=args.lam,
                steps_per_epoch=args.steps, pi_lr=args.pi_lr, v_lr=args.v_lr)
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

    t_state = time.time()
    if args.restore:
        agent.load_policy(args.load_dir)
    if not args.testing:
        for episode in range(args.epochs):
            print("==== Starting episode no:",episode,"====","\n")
            observation = env.reset()
            observation = torch.as_tensor(observation, dtype=torch.float32)

            reward_per_episode = 0  # 用来存储一个episode的总和
            # 存储每个episode的数据
            reward_steps=[]
            action_steps =[]
            obs_steps = []
            steps = 0
            for t in range(max_steps):
                #rendering environmet (optional)
                # env.render()  # test
                state = observation
                # print(state)
                state = torch.as_tensor(state, dtype=torch.float32)

                action = agent.act(state)
                # print(action)
                v = agent.v(state).detach().numpy()


                # if episode >20:
                #     action = agent.policy_net.get_action(state, EXPLORE_NOISE_SCALE)
                # else:
                #     action = Xplane_pid.cal_actions(state)

                # action = np.clip(np.random.normal(action, VAR), -1, 1)
                # print(action)
                action[0] = np.clip(action[0], -1, 1)
                action[1] = np.clip(action[1], -1, 1)
                action[2] = np.clip(action[2], -1, 1)
                action[3] = np.clip(action[3], 0, 1)
                # print(type(action))
                # action_env = decouple_norm(env,action)
                print("Action at step", t, " :", action, "\n")
                start = time.time()

                # observation,reward,done,info=env.step(action_env)
                observation,reward,done,info=env.step(action)


                # print('observation', observation)
                # 每个episode、每一步的数据
                reward_steps.append(reward)
                action_steps.append(action)
                obs_steps.append(observation)
                # print(reward)
                #add s_t,s_t+1,action,reward to experience memory
                agent.buf.store(state, action, reward,  v)
                reward_per_episode+=reward
                if (done or (t == max_steps-1) or env.crash_flag):
                # if (done or (t == max_steps-1) ):

                    if t == max_steps-1:
                        v = agent.v(torch.as_tensor(state, dtype=torch.float32)).detach().numpy()
                    else:
                        v=0
                    print('EPISODE:  ',episode,' Steps: ',t,' Total Reward: ',reward_per_episode)
                    print("Printing reward to file")
                    # exploration_noise.reset() #reinitializing random noise for action exploration
                    reward_st = np.append(reward_st,reward_per_episode)

                    reward_each_episode.append(reward_steps)
                    action_each_episode.append(action_steps)
                    state_each_episode.append(obs_steps)
                    np.savetxt('episode_reward.txt',reward_st, newline="\n")
                    print('\n\n')
                    steps = t
                    env.crash_flag = False
                    agent.buf.finish_path(v)
                    break
                # print('a step time__', time.time() - start)
            agent.buf.ptr = 0
            # if agent.buf.ptr == max_steps:
            #     data = agent.buf.get()
            #     loss_pi, kl = agent.update_pi(data, args.train_pi_iters, args.target_kl)
            #     loss_v = agent.update_v(data, args.train_v_iters)
            #     print('epoch: %3d \t loss of pi: %.3f \t loss of v: %.3f \t kl: %.3f' %
            #           (episode, loss_pi, loss_v, kl))
            if (episode+1) % args.checkpoint_frequency ==0:
                if not os.path.exists(args.saveModel_dir):
                    os.makedirs(args.saveModel_dir)

                torch.save(agent.pi, args.saveModel_dir+'/model.pth')
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

        obs_n, ep_ret, ep_len, episode = env.reset(), 0, 0, 0
        while episode <args.epochs:
            get_action = agent.load_policy(args.save_name + '/model.pth')
            action_n = get_action(obs_n)
            obs, rew, done, _ = env.step(action_n)
            ep_ret += np.sum(rew)
            ep_len += 1
            if done or (ep_len == args.steps):
                print('Episode %d \t EpRet %.3f \t EpLen %d \t Result %s' % (episode, ep_ret, ep_len, done))
                obs_n, ep_ret, ep_len = env.reset(), 0, 0
                episode += 1

            # for step in range(max_steps):
            #     action = agent.act(state)
            #     state, reward, done, info = env.step(action)
            #     state = state.astype(np.float32)
            #     episode_reward += reward
            #     # if (done or (step == max_steps - 1) or env.crash_flag):
            #     if (done or (step == max_steps-1) ):
            #         print("Printing reward to file")
            #         reward_st = np.append(reward_st, reward_per_episode)
            #         env.crash_flag = False
            #         print(time.time()-start)
            #         break
            # print(
            #     'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
            #         episode + 1, args.episodes, episode_reward,
            #         time.time() - t_state
            #     )
            # )




if __name__ == '__main__':
    main()    