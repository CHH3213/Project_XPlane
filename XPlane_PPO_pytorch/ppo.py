import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from gym.spaces import Discrete
import gym
import os
import time
class GAE_buffer:
    def __init__(self, size, obs_dim, act_dim, gamma, lam):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim))
        self.act_buf = np.zeros(combined_shape(size, act_dim))
        self.rew_buf = np.zeros((size,))
        self.val_buf = np.zeros((size,))
        self.rtg_buf = np.zeros((size,))
        self.adv_buf = np.zeros((size,))

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        # print(np.size(act))  #4
        # print(act.shape) #(4,)
        # print(type(act)) #ndarray
        # print(act)
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.ptr += 1

    def finish_path(self, val):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], val)
        vals = np.append(self.val_buf[path_slice], val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]

        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.rtg_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size, 'You must fulfill buffer before getting data!'
        self.path_start_idx, self.ptr = 0, 0

        adv_mu, adv_std = statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mu) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, rtg=self.rtg_buf,
                    adv=self.adv_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

def discount_cumsum(xs, gamma):
    ys = np.zeros_like(xs, dtype=np.float32)
    cumsum = 0.
    for i, x in enumerate(xs[::-1]):
        cumsum *= gamma
        cumsum += x
        ys[-1 - i] = cumsum
    return ys

def statistics_scalar(X, with_min_and_max=False):
    mu = np.mean(X)
    std = np.std(X)
    if with_min_and_max:
        min = np.min(X)
        max = np.max(X)
        return mu, std, min, max
    return mu, std

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def orthogonal_init_(layer, gain=1.0, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, gain)
    torch.nn.init.constant_(layer.bias, bias_const)

def mlp(sizes, activation, output_activation=nn.Identity, orthogonal_init=False, final_gain=0.01):
    layers = []
    for j in range(len(sizes) - 1):
        act, gain = (activation, 1.0) if j < len(sizes) - 2 else (output_activation, final_gain)
        layer = nn.Linear(sizes[j], sizes[j + 1])
        if orthogonal_init:
            orthogonal_init_(layer, gain=gain)
        layers += [layer, act()]
    # print(nn.Sequential(*layers))
    return nn.Sequential(*layers)

class ppo:
    def __init__(self, env_fn, hid=256, layers=2, gamma=0.99, lam=0.97,
                 seed=0, steps_per_epoch=4000, pi_lr=1e-2, v_lr=1e-3, clip_ratio=0.2):
        super(ppo, self).__init__()

        # Instantiate environment
        self.env = env_fn()
        print(self.env.action_space)
        self.discrete = isinstance(self.env.action_space, Discrete)
        self.obs_dim = self.env.observation_space.shape[0]
        # print(len(self.env.action_space.shape))
        self.act_dim = self.env.action_space.shape[0]

        # random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ppo clip ratio
        self.clip_ratio = clip_ratio

        # create actor-critic
        self.mlp_sizes = [self.obs_dim] + [hid] * layers
        self.log_std = torch.nn.Parameter(-0.5 * torch.ones(self.act_dim, dtype=torch.float32))
        self.pi = mlp(sizes=self.mlp_sizes + [self.act_dim], activation=nn.Tanh)
        self.v = mlp(sizes=self.mlp_sizes + [1], activation=nn.Tanh)

        # Count variables
        var_counts = tuple(count_vars(module) for module in [self.pi, self.v])
        print('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

        # Discrete action in buf is of shape (N, )
        self.steps_per_epoch = steps_per_epoch
        # print(self.env.action_space.shape)  # 1
        self.buf = GAE_buffer(steps_per_epoch, self.obs_dim, self.env.action_space.shape, gamma, lam)

        # optimizers
        self.pi_optimizer = Adam(self.pi.parameters(), lr=pi_lr)  # it is wrong to optimize only parameters of mu !!!
        self.v_optimizer = Adam(self.v.parameters(), lr=v_lr)

    def act(self, obs):
        """Used for collecting trajectories or testing, which doesn't require tracking grads"""
        with torch.no_grad():
            logits = self.pi(obs)
            # print(logits)
            if self.discrete:
                pi = Categorical(logits=logits)
            else:
                pi = Normal(loc=logits, scale=self.log_std.exp())
            act = pi.sample()
            # print(act)
        return act.numpy()

    def log_prob(self, obs, act):
        act = act.squeeze(dim=-1)  # critical for discrete actions!
        logits = self.pi(obs)
        if self.discrete:
            pi = Categorical(logits=logits)
            return pi.log_prob(act)
        else:
            pi = Normal(loc=logits, scale=self.log_std.exp())
            # print(pi.log_prob(act).sum(dim=-1))
            return pi.log_prob(act).sum(dim=-1)

    def train(self, epochs=50, train_v_iters=80, train_pi_iters=80, max_ep_len=100, target_kl=0.01, save_name='./model/model.pth'):

        ret_stat, len_stat = [], []
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        for e in range(epochs):
            for t in range(self.steps_per_epoch):
                # env.render()
                o_torch = torch.as_tensor(o, dtype=torch.float32)
                a = self.act(o_torch)

                v = self.v(o_torch).detach().numpy()
                # print(a)
                next_o, r, d, _ = self.env.step(a)


                ep_ret += r
                ep_len += 1
                self.buf.store(o, a, r, v)
                o = next_o
                # print(d)
                timeout = ep_len == max_ep_len
                terminal = d or timeout
                epoch_ended = t == self.steps_per_epoch - 1

                if terminal or epoch_ended:
                    if timeout or epoch_ended:
                        v = self.v(torch.as_tensor(o, dtype=torch.float32)).detach().numpy()
                    else:
                        v = 0.
                    if terminal:
                        ret_stat.append(ep_ret)
                        len_stat.append(ep_len)
                    self.buf.finish_path(v)
                    o, ep_ret, ep_len = self.env.reset(), 0, 0
                    # break

            data = self.buf.get()
            loss_pi, kl = self.update_pi(data, train_pi_iters, target_kl)
            loss_v = self.update_v(data, train_v_iters)
            print('epoch: %3d \t loss of pi: %.3f \t loss of v: %.3f \t kl: %.3f \t return: %.3f \t ep_len: %.3f' %
                  (e, loss_pi, loss_v, kl, np.mean(ret_stat), np.mean(len_stat)))
            torch.save(self.pi, save_name)
            ret_stat, len_stat = [], []

    def update_pi(self, data, iter, target_kl):
        obs, act, adv = data['obs'], data['act'], data['adv']
        logp_old = self.log_prob(obs, act).detach()
        for i in range(iter):
            self.pi_optimizer.zero_grad()
            logp = self.log_prob(obs, act)
            appro_kl = (logp_old - logp).mean().item()
            if appro_kl > 1.5 * target_kl:
                print('PPO stops at iter %d' % i)
                break

            ratio = (logp - logp_old).exp()
            adv_clip = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            loss_pi = -torch.min(ratio * adv, adv_clip).mean()
            loss_pi.backward()
            self.pi_optimizer.step()
        return loss_pi.item(), appro_kl

    def update_v(self, data, iter):
        obs, rtg = data['obs'], data['rtg']
        for i in range(iter):
            self.v_optimizer.zero_grad()
            v = self.v(obs).squeeze()
            loss_v = ((v - rtg) ** 2).mean()
            loss_v.backward()
            self.v_optimizer.step()
        return loss_v.item()

    def load_policy(fpath):
        net = torch.load(fpath)

        # make function for producing an action given a single state
        @torch.no_grad()
        def get_action(obs_n):
            logits = net(torch.Tensor(obs_n))
            pi = Categorical(logits=logits)
            action_n = pi.sample().numpy()

            return action_n

        return get_action


if __name__ == '__main__':
    import argparse

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
    parser.add_argument('--save_name', type=str, default='./model')
    parser.add_argument('--testing', default=True, action="store_true",
                        help="reduces exploration substantially")
    args = parser.parse_args()


    env = gym.make('Pendulum-v0')
    if not os.path.exists(args.save_name):
        os.makedirs(args.save_name)
    agent = ppo(lambda: env, hid=args.hid, layers=args.layers, seed=args.seed, lam=args.lam,
                steps_per_epoch=args.steps, pi_lr=args.pi_lr, v_lr=args.v_lr)
    agent.train(epochs=args.epochs, max_ep_len=args.steps, save_name=args.save_name+'/model.pth')

    if args.testing:
        obs_n, ep_ret, ep_len, episode = env.reset(), 0, 0, 0
        while episode < args.epochs:
            env.render()
            time.sleep(1e-1)
            get_action = agent.load_policy(args.save_name+'/model.pth')
            action_n = get_action(obs_n)
            obs, rew, done, _ = env.step(action_n)
            ep_ret += np.sum(rew)
            ep_len += 1

            if done or (ep_len == args.steps):
                print('Episode %d \t EpRet %.3f \t EpLen %d \t Result %s' % (episode, ep_ret, ep_len, done))
                obs_n, ep_ret, ep_len = env.reset(), 0, 0
                episode += 1