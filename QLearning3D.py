import collections
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import extract_seq
import numpy as np
import seaborn as sns


class Q_Learning:
    def __init__(self):
        self.Q = collections.defaultdict(int)
        self.lr = 0.01
        self.max_epsilon = 1
        self.min_epsilon = 0.3
        self.decay_steps = 100000
        self.discount = 0.9
        self.actions = [0, 1, 2, 3, 4, 5]
        self.steps = 0

    def select_action(self, state, explore=True):

        if explore:
            pr = random.random()
            if self.steps > self.decay_steps:
                epsilon = self.min_epsilon
            else:
                epsilon = self.max_epsilon - (self.max_epsilon - self.min_epsilon) * (self.steps / self.decay_steps)
            if pr <= epsilon:
                return random.choice(self.actions)
        best_q = float('-inf')
        best_a = None
        for a in self.actions:
            if self.Q[(state, a)] > best_q:
                best_q = self.Q[(state, a)]
                best_a = a
        self.steps += 1
        return best_a

    def select_action_without_explore(self, state, actions):
        best_q = float('-inf')
        best_a = None
        for a in self.actions:
            if self.Q[(state, a)] > best_q and a not in actions:
                best_q = self.Q[(state, a)]
                best_a = a
        self.steps += 1
        return best_a

    def learn(self):
        pass

    def store_transition(self, state, action, next_state, reward):
        next_best_q = float('-inf')
        for a in self.actions:
            next_best_q = max(next_best_q, self.Q[(next_state, a)])
        sample = reward + self.discount * next_best_q
        self.Q[(state, action)] += self.lr * (sample - self.Q[(state, action)])


class Env:
    def __init__(self, grid_size, collision_penalty, trap_penalty):
        self.grid_size = grid_size
        self.dirs = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        self.collision_penalty = collision_penalty
        self.trap_penalty = trap_penalty

    def isTrapped(self):
        for dx, dy, dz in self.dirs:
            if (self.cur_position[0] + dx, self.cur_position[1] + dy, self.cur_position[2] + dz) not in self.state:
                return False
        return True

    def step(self, action):
        if self.cur_index <= 0:
            x, y, z = self.initial_positions[self.cur_index]
        else:
            x = self.dirs[action][0] + self.cur_position[0]
            y = self.dirs[action][1] + self.cur_position[1]
            z = self.dirs[action][2] + self.cur_position[2]

        reward = None
        if (x, y, z) not in self.state:
            self.append(x, y, z)
            if self.cur_index == len(self.seq):
                self.done = True
                reward = -self.free_energy()
            elif self.isTrapped():
                self.done = True
                reward = (-len(self.seq) + len(self.state)) * self.trap_penalty - self.free_energy()
            else:
                reward = 0.1
            if self.state_index == -1:
                self.state_index = 0
            else:
                self.state_index = 6 * self.state_index + 1 + action
        else:
            reward = -self.collision_penalty

        return self.state_index, reward, self.done

    def append(self, x, y, z):
        self.state[(x, y, z)] = self.seq[self.cur_index]
        self.cur_index += 1
        self.cur_position = [x, y, z]

    def generateSeq(self):
        n = random.randint(1, 20)
        seq = ""
        for i in range(n):
            p = random.random()
            if p < 0.5:
                seq += "H"
            else:
                seq += 'P'
        return seq

    def getSeq(self):
        seq = ""
        for i in self.seq:
            if i == -1:
                seq += "P"
            else:
                seq += "H"
        return seq

    def setSeq(self, seq, positions):
        self.labels = []
        self.seq = []
        cur = 1
        for i in seq:
            if i == 'H' or i == 'h':
                self.seq.append(1)
                self.labels.append(str(cur) + "-H")
            else:
                self.seq.append(-1)
                self.labels.append(str(cur) + "-P")
            cur += 1
        self.initial_positions = positions

    def getNext(self):
        return self.seq[self.cur_index]

    def reset(self):
        self.cur_index = 0
        self.cur_position = None  # the position of the last molecule added

        self.state = collections.OrderedDict()
        self.state_index = -1

        self.done = False

    def render_heatmap(self, predict=True):
        n = len(self.seq)
        dist = np.zeros((n, n))
        if not predict:
            positions = self.initial_positions
        else:
            positions = []
            for (x, y, z), value in self.state.items():
                positions.append([x, y, z])

        for i in range(n):
            for j in range(i + 1, n):
                temp = np.sqrt(np.sum(np.square(np.array(positions[i]) - np.array(positions[j]))))
                dist[i][j] = temp
                dist[j][i] = temp

        max, min = dist.max(), dist.min()
        dist = (dist - min) / (max - min)
        ax = sns.heatmap(dist, cmap="YlGnBu", square=True, xticklabels=self.labels, yticklabels=self.labels)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        if predict:
            plt.title("Prediction")
        else:
            plt.title("2D Grid")
        plt.show()

    def render_structure(self, reward, episode, marker):
        fig = plt.figure()
        axes3d = Axes3D(fig)

        x = []
        y = []
        z = []
        xrange = [self.grid_size, 0]
        yrange = [self.grid_size, 0]
        zrange = [self.grid_size, 0]
        for (i, j, k), value in self.state.items():
            x.append(i)
            y.append(j)
            z.append(k)
            xrange[0] = min(xrange[0], i)
            xrange[1] = max(xrange[1], i)
            yrange[0] = min(yrange[0], j)
            yrange[1] = max(yrange[1], j)
            zrange[0] = min(zrange[0], k)
            zrange[1] = max(zrange[1], k)
            if marker:
                if value == 1:
                    axes3d.scatter(i, j, k, c='b', s=90, zorder=2)
                    for dx, dy, dz in [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]:
                        cx, cy, cz = i + dx, j + dy, k + dz
                        if self.state.get((cx, cy, cz), 0) == 1:
                            plt.plot([i, cx], [j, cy], [k, cz], linewidth=3, color='r', zorder=1)
                else:
                    axes3d.scatter(i, j, k, c='g', s=90, zorder=2)

        axes3d.plot(x, y, z, linewidth=3, color='black', zorder=1, label="prediction")

        # x = []
        # y = []
        # z = []
        # for t in range(len(self.initial_positions)):
        #     i, j, k = self.initial_positions[t]
        #     value = self.seq[t]
        #     x.append(i)
        #     y.append(j)
        #     z.append(k)
        #     xrange[0] = min(xrange[0], i)
        #     xrange[1] = max(xrange[1], i)
        #     yrange[0] = min(yrange[0], j)
        #     yrange[1] = max(yrange[1], j)
        #     zrange[0] = min(zrange[0], k)
        #     zrange[1] = max(zrange[1], k)
        #     if marker:
        #         if value == 1:
        #             axes3d.scatter(i, j, k, c='deepskyblue', s=90, zorder=2)
        #         else:
        #             axes3d.scatter(i, j, k, c='lightgreen', s=90, zorder=2)
        #
        # axes3d.plot(x, y, z, linewidth=3, color='red', zorder=1, label="2D grid structure")
        #
        # size = 5
        # axes3d.set_xticks(
        #     range(min(xrange[0], self.grid_size // 2 - size), max(self.grid_size // 2 + size, xrange[1]) + 1, 1))
        # axes3d.set_yticks(
        #     range(min(yrange[0], self.grid_size // 2 - size), max(self.grid_size // 2 + size, yrange[1]) + 1, 1))
        # axes3d.set_zticks(
        #     range(min(zrange[0], self.grid_size // 2 - size), max(self.grid_size // 2 + size, zrange[1]) + 1, 1))
        # plt.title("episode: {}, reward: {}".format(episode, reward))
        # plt.legend()
        plt.show()

    def free_energy(self):
        consecutive_h = 0
        adjacent_h = 0
        pre = None
        for (x, y, z), value in self.state.items():
            if value == 1:
                if pre == 1:
                    consecutive_h += 1
                for dx, dy, dz in self.dirs:
                    cx, cy, cz = x + dx, y + dy, z + dz
                    if self.state.get((cx, cy, cz), 0) == 1:
                        adjacent_h += 1
            pre = value
        return consecutive_h - adjacent_h // 2


def evaluate(env, agent, episode, marker=True):
    # print("start evaluation")
    actions = []
    env.reset()
    done = False
    reward = None
    invalid = set()
    while not done:
        state = env.state_index
        action = agent.select_action_without_explore(state, invalid)
        actions.append(action)
        next_state, reward, done = env.step(action)
        if next_state == state:
            invalid.add(action)
        else:
            invalid = set()

    env.render_structure(reward, episode, marker)
    env.render_heatmap(predict=True)
    return reward, len(actions)


def generate_seq(max_length=20, prob=0.5):
    n = random.randint(1, max_length)
    seq = ""
    for i in range(n):
        p = random.random()
        if p < prob:
            seq += "H"
        else:
            seq += 'P'
    return seq


grid_size = 21 * 2 + 1  # odd value
max_episode = 10000000
collision_penalty = 1
trap_penalty = 5
min_learn_size = 10
learn_step = 5000

if __name__ == '__main__':
    # data = [("hhppppphhppphppphp", 4), ("hphphhhppphhhhpphh", 8), ("phpphphhhphhphhhhh", 9), ("hphpphhphpphphhpphph", 9),
    #         ("hhhpphphphpphphphpph", 10)]

    env = Env(grid_size, collision_penalty, trap_penalty)
    evaluate_interval = 100000

    seq, positions, real_positions = extract_seq.get_data(spacing=3.7, start=[env.grid_size // 2, env.grid_size // 2,
                                                                              env.grid_size // 2],
                                                          file='1fat.pdb')
    env.setSeq(seq[:20], positions[:20])
    agent = Q_Learning()

    step = 0
    for episode in range(max_episode):
        env.reset()
        done = False
        reward = None
        while not done:
            state = env.state_index
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, next_state, reward)
            step += 1
        if episode % evaluate_interval == 0 and done:
            reward, _ = evaluate(env, agent, episode, marker=False)
            print("episode {}, reward = {}".format(episode, reward))
    reward, _ = evaluate(env, agent, max_episode, marker=False)
    print("seq = {}, reward = {}".format(seq, reward))

    env.render_heatmap(predict=False)
