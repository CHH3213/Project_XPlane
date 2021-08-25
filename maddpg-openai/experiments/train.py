# !/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import tensorflow as tf
import time
import os
from sys import path
import tf_util as U
from maddpg import MADDPGAgentTrainer
# from maddpg import MADDPGEnsembleAgentTrainer
import tensorflow.contrib.layers as layers
import tensorflow.contrib.rnn as rnn
import scipy.io as sio

# from env import XPlane
from env2_2 import XplaneEnv as XPlane

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_tag", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=400, help="maximum episode length")  #每个episode的步数为400步
    #episodes的回数，先前默认60000,现在改成5000
    parser.add_argument("--num-episodes", type=int, default=5000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")
    #这里切换ddpg和maddpg
    parser.add_argument("--adv-policy", type=str, default="ddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="../tmp/xplane_1/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="../tmp/xplane_1/", help="directory in which training state and model are loaded")

    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    # default=True出现图形界面
    parser.add_argument("--display", action="store_true", default=False)
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        # out = tf.nn.dropout(out, 0.8)  # dropout
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        # out = tf.nn.dropout(out, 0.6)  # dropout
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=tf.nn.tanh)
        # out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out



def mlp_model_q(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        # out = tf.nn.dropout(out, 0.8)  # dropout
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        # out = tf.nn.dropout(out, 0.8)  # dropout
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        # out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def lstm_mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions

    lstm_size = input.shape[1]
    input = tf.expand_dims(input, 0)  # [1,?,232]

    with tf.variable_scope(scope, reuse=reuse):
        # fully_connetcted: 全连接层
        out = input
        lstm = rnn.BasicLSTMCell(num_units, forget_bias=1.0, state_is_tuple=True)
        # GRU = tf.nn.rnn_cell.GRUCell(num_units)
        init_state = lstm.zero_state(1, dtype=tf.float32)
        # outputs, _states = rnn.static_rnn(lstm, X_split, dtype=tf.float32)
        outputs, _states = tf.nn.dynamic_rnn(lstm, out, time_major=False, initial_state=init_state)

        # outputs = tf.convert_to_tensor(np.array(outputs))
        out = layers.fully_connected(outputs[-1], num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=tf.nn.tanh)
        return out

def lstm_mlp_model_q(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions

    lstm_size = input.shape[1]
    input = tf.expand_dims(input, 0)  # [1,?,232]

    with tf.variable_scope(scope, reuse=reuse):
        # fully_connetcted: 全连接层
        out = input
        lstm = rnn.BasicLSTMCell(num_units, forget_bias=1.0, state_is_tuple=True)
        GRU = tf.nn.rnn_cell.GRUCell(num_units)
        init_state = lstm.zero_state(1, dtype=tf.float32)
        # outputs, _states = rnn.static_rnn(lstm, X_split, dtype=tf.float32)
        outputs, _states = tf.nn.dynamic_rnn(lstm, out, time_major=False, initial_state=init_state)
        # outputs = tf.convert_to_tensor(np.array(outputs))
        out = layers.fully_connected(outputs[-1], num_outputs=num_units, activation_fn=tf.nn.relu)
        # out = tf.nn.dropout(out, 0.8)  # dropout
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


# 修改
def get_trainers(env, obs_shape_n, arglist):
    '''
    从env中读取agent数量，再根据env中的动作空间action_space和obs_shape_n创建agent训练实例。
    make the List of trainers
    @Output: List of trainers 返回训练实例对象trainers
    '''
    trainers = []
    model1 = mlp_model
    model2 = mlp_model_q
    model_lstm1 = lstm_mlp_model
    model_lstm2 = lstm_mlp_model_q
    trainer = MADDPGAgentTrainer
    # trainer = MADDPGEnsembleAgentTrainer
    # print(arglist.adv_policy)
    trainers.append(trainer(
        "agent_xplane", model2, model1, obs_shape_n, env.action_space, 1, arglist,
        local_q_func=(arglist.adv_policy=='ddpg')))


    return trainers



#核心部分
def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = XPlane()
        # Create agent trainers
        # obs_shape_n = [env.observation_space_shape[i] for i in range(2)]
        obs_shape_n = [env.observation_space.shape ]
        # print('obs_shape_n',obs_shape_n) #('obs_shape_n', [(8,), (8,)])
        ###
        trainers = get_trainers(env, obs_shape_n, arglist)
        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore:
            print("===============================")
            print('Loading previous state...')
            print("===============================")
            filename = 'XPlaneSimulink'
            arglist.load_dir= os.path.join( arglist.load_dir, filename)
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [0.0]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        # print('asdasd', obs_n[0])
        episode_step = 0
        train_step = 0
        t_start = time.time()
        print('Starting iterations...')

        # chh ===============================
        # 数据保存
        episode_reward = []
        step_episode = []
        position_ = []
        volocity = []
        step = [i for i in range(arglist.max_episode_len+1)]
        action_save = []
        file_folder_name = "../save_data/训练数据/xplane_1"  # policy_continueTest
        if not os.path.exists(file_folder_name):
            os.makedirs(file_folder_name)
        ## -------------------------------
        while True:
            # get action
            # 在每一次迭代中，都会计算所有agent的动作
            # print('obs_n', obs_n) # *连续离散都是1*8维的，追与逃都是
            # for obs in obs_n:
            #     print('sdas', obs)
            # print('888888888888888888trainers', trainers)
            # print('gent.action', trainers[0].action(obs_n[0]))
            # action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            action_n = [trainers.action(obs_n)]


            # 调用环境执行n个智能体生成的动作，环境返回n个智能体的新状态、奖励值、智能体是否死亡、其他信息
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            # done = all(done_n)
            done = any(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
            # 保存训练数据到经验回放单元

            agent.experience(obs_n, action_n, rew_n, new_obs_n, done_n, terminal)
            # agent.experience(obs_n, action_n, rew_n, new_obs_n, done_n, terminal) # ====emsable

            # 更新状态
            obs_n = new_obs_n
            # chh注释
            episode_rewards[-1] += rew_n
            agent_rewards[-1] += rew_n

            # ## chh添加
            # # 数据保存
            step_episode.append(rew_n)  # 保存reward

            action_save.append(action_n)

            # print('episode_step',episode_step)

            # 如果智能体任务完成，或者智能体走完一个max_episode_len（默认25步）
            if done or terminal:

                episode_reward.append(step_episode) #将400个step保存进列表中

                # 重置
                step_episode = [] # 将每个episode的reward列表清空
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                agent_info.append([[]])

                # print("lenth of episode_rewards is :", len(episode_rewards))
                # print(f"finished no.{num_terminal} episode!") # chh 2020/10/20

            # increment global step counter
            train_step += 1



            # 更新所有trainers参数update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)
                # loss = agent.update(trainers)  # ====emsable

            # save model, display training output#保存模型

            # if terminal and (len(episode_rewards) % arglist.save_rate == 0) :
            # if (done or terminal) and (len(episode_rewards) % arglist.save_rate == 0) :  #modified
            if done or terminal:  #modified
                name='XplaneSimulink'
                if not os.path.exists(arglist.save_dir):
                    os.makedirs(arglist.save_dir)
                U.save_state(arglist.save_dir+name, saver=saver) # 保存模型！！
                # print statement depends on whether or not there are adversaries

                print("steps: {}, episodes: {}, mean episode reward: {}, time: {} \n".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))

                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                final_ep_ag_rewards.append(np.mean(agent_rewards[-arglist.save_rate:]))
            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                print("总共耗时：".format(round(time.time()-t_start, 3))) # chh 10/20
                # 保存所有的reward
                sio.savemat(file_folder_name + '/rewards.mat',
                            {'episode_reward': episode_reward})
                print('save_reward success !!!')
                break


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
