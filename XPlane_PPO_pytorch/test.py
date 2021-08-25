import time

import numpy as np
import torch
from torch.distributions.categorical import Categorical


def load_policy(fpath):
    net = torch.load(fpath, map_location='cpu')

    # make function for producing an action given a single state
    @torch.no_grad()
    def get_action(obs_n):
        logits = net(torch.Tensor(obs_n))
        pi = Categorical(logits=logits)
        action_n = pi.sample().numpy()

        return action_n

    return get_action

def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    obs_n, ep_ret, ep_len, episode = env.reset()[1], 0, 0, 0
    while episode < num_episodes:
        if render:
            env.render()
            time.sleep(1e-1)
        action_n = get_action(obs_n)
        obs_n, rew_n, done_n, _ = env.step(action_n)
        obs_n, rew_n = obs_n[1], rew_n[1]
        ep_ret += np.sum(rew_n)
        ep_len += 1
        done = any(done_n)

        if done or (ep_len == max_ep_len):
            print(done_n)
            print('Episode %d \t EpRet %.3f \t EpLen %d \t Result %s' % (episode, ep_ret, ep_len, done))
            obs_n, ep_ret, ep_len = env.reset()[1], 0, 0
            episode += 1

def make_env(scenario_name, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, done_callback=scenario.is_done)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, done_callback=scenario.is_done)
    return env

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', default='model.pt', type=str)
    parser.add_argument('--len', '-l', type=int, default=400)
    parser.add_argument("--scenario", default='simple_tag', type=str, help="name of the scenario script")
    parser.add_argument('--episodes', '-n', type=int, default=1)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--discrete_action_space', '-d', default=False, action='store_true')
    parser.add_argument('--final', default=False, action="store_true")
    args = parser.parse_args()
    env = make_env(args.scenario, args)
    get_action = load_policy(args.fpath)

    run_policy(env, get_action, args.len, args.episodes, not args.norender)