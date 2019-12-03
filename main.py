from Env import Env
from chainerrl.agents import dqn as DQN
from chainerrl import replay_buffer, explorers
from chainer import optimizers
import numpy as np
from chainerrl.action_value import DiscreteActionValue
import chainer.functions as F
import chainer.links as L
import chainer
import functools


def evaluate(env, agent, episode):
    actions = []
    state = env.reset("hhppphh")
    done = False
    while not done:
        action = agent.act(state)
        state, reward, done = env.step(action)
        actions.append(action)
        if reward < 0:
            break
    env.render(episode, reward)
    return reward, actions


class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions):
        super(QFunction, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, 64, 3, 1, 1, nobias=True)
            self.bn = L.BatchNormalization(64)
            self.l2 = L.Linear(None, n_actions)

    def forward(self, x):
        # print(x.shape)
        """Compute Q-values of actions for given observations."""
        h = F.relu(self.bn(self.conv(x)))
        h = self.l2(h)
        return DiscreteActionValue(h)


def create_agent(env):
    q_func = QFunction(env.grid_size ** 2, 4)

    start_epsilon = 1.
    end_epsilon = 0.8
    decay_steps = 20000

    explorer = explorers.LinearDecayEpsilonGreedy(start_epsilon, end_epsilon, decay_steps, env.random_action)

    opt = optimizers.Adam()
    opt.setup(q_func)

    rbuf_capacity = 5 * 10 ** 3
    minibatch_size = 16

    steps = 50000
    replay_start_size = 20
    update_interval = 10
    betasteps = (steps - replay_start_size) // update_interval
    rbuf = replay_buffer.PrioritizedReplayBuffer(rbuf_capacity)

    phi = lambda x: x.astype(np.float32, copy=False)

    agent = DQN.DQN(q_func, opt, rbuf, gamma=0.99,
                    explorer=explorer, replay_start_size=replay_start_size,
                    phi=phi, minibatch_size=minibatch_size)
    return agent


if __name__ == '__main__':
    grid_size = 7 * 2 + 1  # odd value
    max_episode = 100000
    collision_penalty = 4
    trap_penalty = 10
    min_learn_size = 10
    learn_step = 10
    seq = "hhppphh"

    env = Env(grid_size, collision_penalty, trap_penalty)
    agent = create_agent(env)

    steps = 0
    for episode in range(max_episode):
        state = env.reset(seq)
        done = False
        reward = 0
        while not done:
            action = agent.act_and_train(state, reward)
            state, reward, done = env.step(action)
            steps += 1
            # print(action,done)
        agent.stop_episode_and_train(state, reward, done)
        # print("episode {}".format(episode))
        if episode % 100 == 0:
            reward, _ = evaluate(env, agent, episode)
            print("episode:{}, reward = {}".format(episode, reward))
    print(reward)
