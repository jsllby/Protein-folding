from Env import Env
from Q_Learning import Q_Learning
from chainerrl.agents import double_dqn as DDQN
from chainerrl import replay_buffer, explorers
from chainer import optimizers
import numpy as np
from chainerrl.action_value import DiscreteActionValue
import chainer.functions as F
import chainer.links as L
import chainer


def evaluate(env, agent):
    actions = []
    env.reset()
    done = False
    while not done:
        state = env.state_index
        action = agent.select_action(state, explore=False)
        actions.append(action)
        next_state, reward, done = env.step(action)
    return reward, actions


net_layers = [64, 32]


class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=None):
        super(QFunction, self).__init__()
        if n_hidden_channels is None:
            n_hidden_channels = net_layers
        net = []
        inpdim = obs_size
        for i, n_hid in enumerate(n_hidden_channels):
            net += [('l{}'.format(i), L.Linear(inpdim, n_hid))]
            # net += [('norm{}'.format(i), L.BatchNormalization(n_hid))]
            net += [('_act{}'.format(i), F.relu)]
            net += [('_dropout{}'.format(i), F.dropout)]
            inpdim = n_hid

        net += [('output', L.Linear(inpdim, n_actions))]

        with self.init_scope():
            for n in net:
                if not n[0].startswith('_'):
                    setattr(self, n[0], n[1])

        self.forward = net

    def __call__(self, x, test=False):
        """
        Args:
            x (ndarray or chainer.Variable): An observation
            test (bool): a flag indicating whether it is in test mode
        """
        for n, f in self.forward:
            if not n.startswith('_'):
                x = getattr(self, n)(x)
            elif n.startswith('_dropout'):
                x = f(x, 0.1)
            else:
                x = f(x)

        return DiscreteActionValue(x)


def create_agent(env):
    q_func = QFunction(env.grid_size ** 2, 4)

    start_epsilon = 1.
    end_epsilon = 0.3
    decay_steps = 20

    explorer = explorers.LinearDecayEpsilonGreedy(start_epsilon, end_epsilon, decay_steps, env.random_action)

    opt = optimizers.Adam()
    opt.setup(q_func)

    rbuf_capacity = 5 * 10 ** 3
    minibatch_size = 16

    steps = 1000
    replay_start_size = 20
    update_interval = 10
    betasteps = (steps - replay_start_size) // update_interval
    rbuf = replay_buffer.PrioritizedReplayBuffer(rbuf_capacity, betasteps=betasteps)

    phi = lambda x: x.astype(np.float32, copy=False)  # need to change

    agent = DDQN.DoubleDQN(q_func, opt, rbuf, gamma=0.99,
                           explorer=explorer, replay_start_size=replay_start_size,
                           target_update_interval=10,  # target q网络多久和q网络同步
                           update_interval=update_interval,
                           phi=phi, minibatch_size=minibatch_size)
    return agent


if __name__ == '__main__':
    grid_size = 21  # odd value
    max_episode = 10000
    collision_penalty = 1
    trap_penalty = 10
    min_learn_size = 10
    learn_step = 10
    seq = "HPHPHPHHPP"

    env = Env(grid_size, collision_penalty, trap_penalty)
    agent = create_agent(env)

    steps = 0
    for episode in range(max_episode):
        # print("episode:{}".format(episode))
        state = env.reset(seq)
        done = False
        reward = 0
        while not done:
            action = agent.act_and_train(state, reward)
            state, reward, done = env.step(action)
            steps += 1
            print(steps)
        agent.stop_episode_and_train(state, reward, done)

