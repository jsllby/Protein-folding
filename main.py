from Env import Env
from chainerrl.agents import dqn as DQN
from chainerrl import replay_buffer, explorers
from chainer import optimizers, Variable
import numpy as np
from chainerrl.action_value import DiscreteActionValue
import chainer.functions as F
import chainer.links as L
import chainer
import functools

grid_size = 100
obs_size = 16
minibatch_size = 16


def evaluate(env, agent, episode):
    data = [("hhppppphhppphppphp", 4), ("hphphhhppphhhhpphh", 8), ("phpphphhhphhphhhhh", 9),
            ("hphpphhphpphphhpphph", 9),
            ("hhhpphphphpphphphpph", 10)]
    res = []
    loss = []
    for seq, opt in data:
        collision = False
        actions = []
        state = env.reset(seq)
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done = env.step(action)
            actions.append(action)
            if reward < 0:
                collision = True
                break
        res.append(reward)
        loss.append(opt - reward)
    loss = np.mean(np.square(loss))
    #     env.render(episode, reward)
    return res, loss


class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions):
        super(QFunction, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 3)
            self.bn1 = L.BatchNormalization(64)
            self.conv2 = L.Convolution2D(None, 64, 3)
            self.bn2 = L.BatchNormalization(64)
            self.conv3 = L.Convolution2D(None, 64, 3)
            self.bn3 = L.BatchNormalization(64)

            self.lstm = L.LSTM(None,128)

            self.l = L.Linear(None, 4)

    def forward(self, x):
        """Compute Q-values of actions for given observations."""
        x1 = x[:, 0, :, :].reshape((-1, 1, obs_size * 2 + 1, obs_size * 2 + 1))
        x2 = x[:, 1, :, :].reshape((-1, (obs_size * 2 + 1) ** 2))
        if x2.shape[0] == 1:
            x2 = np.tile(x2, (minibatch_size, 1))
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(x)))
        h = F.relu(self.bn3(self.conv3(x)))
        h = self.l(h)
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

    max_episode = 100000
    collision_penalty = 5
    trap_penalty = 10
    min_learn_size = 10
    learn_step = 10
    seq = "hhppphhh"

    env = Env(grid_size, collision_penalty, trap_penalty, obs_size)
    agent = create_agent(env)

    print("start trainning")
    steps = 0
    for episode in range(max_episode):
        print("episode {}".format(episode))
        state = env.reset()
        done = False
        reward = 0
        while not done:
            action = agent.act_and_train(state, reward)
            state, reward, done = env.step(action)
            steps += 1
            # print(action,done)
        agent.stop_episode_and_train(state, reward, done)
        # print("episode {}".format(episode))
        if episode % 10 == 0:
            reward, loss = evaluate(env, agent, episode)
            print("episode:{}, MSE = {}, rewards = {}".format(episode, loss, reward))
    #             if collision:
    #               print("episode:{}, reward = {}, collision".format(episode, reward))
    #             else:
    #               print("episode:{}, reward = {}".format(episode, reward))
    print(reward)
