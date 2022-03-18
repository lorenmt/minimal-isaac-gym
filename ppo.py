from env import Cartpole

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal


# define network architecture here
class Net(nn.Module):
    def __init__(self, num_inputs=4, num_outputs=1):
        super(Net, self).__init__()
        # we use a shared backbone for both actor and critic
        self.shared_net = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
        )

        # mean and variance for Actor Network
        self.to_mean = nn.Sequential(
            nn.Linear(256, num_outputs),
            nn.Tanh()
        )

        self.to_std = nn.Sequential(
            nn.Linear(256, num_outputs),
            nn.Softplus()
        )

        # value for Critic Network
        self.to_value = nn.Sequential(
            nn.Linear(256, 1)
        )

    def pi(self, x):
        x = self.shared_net(x)
        mu, std = self.to_mean(x), self.to_std(x)
        return mu, std

    def v(self, x):
        x = self.shared_net(x)
        x = self.to_value(x)
        return x


class PPO:
    def __init__(self, args):
        self.env = Cartpole(args)

        self.epoch = 5
        self.lr = 3e-4
        self.gamma = 0.9
        self.lmbda = 0.9
        self.clip = 0.2
        self.rollout_size = 128
        self.chunk_size = 32
        self.mini_chunk_size = self.rollout_size // self.chunk_size
        self.num_eval_freq = 100

        self.data = []
        self.score = 0
        self.run_step = 0
        self.optim_step = 0

        self.net = Net(self.env.num_obs, self.env.num_act).to(args.sim_device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def make_data(self):
        # organise data and make batch
        data = []
        for _ in range(self.chunk_size):
            s_lst, a_lst, r_lst, s_prime_lst, log_prob_lst, done_lst = [], [], [], [], [], []
            for _ in range(self.mini_chunk_size):
                rollout = self.data.pop()
                s, a, r, s_prime, log_prob, done = rollout

                s_lst.append(s)
                a_lst.append(a)
                r_lst.append(r)
                s_prime_lst.append(s_prime)
                log_prob_lst.append(log_prob)
                done_lst.append(done)

            s, a, r, s_prime, done = \
                torch.cat(s_lst), torch.cat(a_lst), torch.cat(r_lst), torch.cat(s_prime_lst), torch.cat(done_lst)

            # compute reward-to-go
            with torch.no_grad():
                target = r + self.gamma * self.net.v(s_prime) * done
                delta = target - self.net.v(s)

            # compute advantage
            advantage_lst = []
            advantage = 0.0
            for delta_t in reversed(delta):
                advantage = self.gamma * self.lmbda * advantage + delta_t
                advantage_lst.insert(0, advantage)

            advantage = torch.cat(advantage_lst)
            log_prob = torch.cat(log_prob_lst)

            mini_batch = (s, a, log_prob, target, advantage)
            data.append(mini_batch)
        return data

    def act(self, s):
        mu, std = self.net.pi(s)
        dist = Normal(mu, std)
        a = dist.sample()
        log_prob = dist.log_prob(a)
        return a, log_prob

    def update(self):
        # update actor and critic network
        data = self.make_data()

        for i in range(self.epoch):
            for mini_batch in data:
                s, a, old_log_prob, target, advantage = mini_batch

                _, log_prob = self.act(s)

                ratio = torch.exp(log_prob - old_log_prob)

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage
                loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.net.v(s), target)

                self.optim.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optim.step()

                self.optim_step += 1

    def run(self):
        # collect data
        s = self.env.obs_buf

        with torch.no_grad():
            a, log_prob = self.act(s)

        s_prime, r, done = self.env.step(a)
        self.data.append((s, a, r.unsqueeze(-1).float(), s_prime, log_prob, 1 - done.unsqueeze(-1).float()))

        self.score += r.float().mean().item()

        # update policy
        if len(self.data) == self.rollout_size:
            self.update()

        # evaluation
        if (self.run_step + 1) % self.num_eval_freq == 0:
            print('Steps: {:04d} | Opt Step: {:04d} | Reward {:.04f} |'
                  .format(self.run_step, self.optim_step, self.score / self.num_eval_freq))
            self.score = 0

        self.run_step += 1
