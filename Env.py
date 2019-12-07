import random
import collections
import matplotlib.pyplot as plt
import numpy as np


class Env:
    def __init__(self, grid_size, collision_penalty, trap_penalty, obs_size):
        self.grid_size = grid_size
        self.dirs = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        self.collision_penalty = collision_penalty
        self.trap_penalty = trap_penalty
        self.obs_size = obs_size

    def is_trapped(self):
        for dx, dy in self.dirs:
            x = self.cur_position[0] + dx
            y = self.cur_position[1] + dy
            if self.hp[x][y] == 0:
                return False
        return True

    def random_action(self):
        return random.randint(0, 3)

    def step(self, action):
        if not self.cur_position:
            x = y = self.grid_size // 2
        else:
            x = self.dirs[action][0] + self.cur_position[0]
            y = self.dirs[action][1] + self.cur_position[1]

        if self.hp[x][y] == 0:
            self.append(x, y)
            if self.cur_index == len(self.seq):
                self.done = True
                reward = -self.free_energy()
            elif self.is_trapped():
                self.done = True
                reward = -(len(self.seq) - self.cur_index) * self.trap_penalty - self.free_energy()
            else:
                reward = 0.1
        else:
            reward = -self.collision_penalty

        return self.get_state(), reward, self.done

    def append(self, x, y):
        if self.cur_position:
            self.pos[self.cur_position[0]][self.cur_position[1]] = 0
        self.hp[x][y] = self.seq[self.cur_index]
        self.pos[x][y] = self.seq[self.cur_index]
        self.order[(x, y)] = self.seq[self.cur_index]
        self.cur_index += 1
        self.cur_position = [x, y]

    def generate_seq(self):
        n = random.randint(15, 22)
        seq = ""
        for i in range(n):
            p = random.random()
            if p < 0.5:
                seq += "H"
            else:
                seq += 'P'
        return seq

    def get_seq(self):
        seq = ""
        for i in self.seq:
            if i == -1:
                seq += "P"
            else:
                seq += "H"
        return seq

    def set_seq(self, seq):
        self.seq = []
        pre = None
        self.consecutive_h = 0
        for i in seq:
            if i == 'H' or i == 'h':
                self.seq.append(1)
                if pre == 'H' or pre == 'h':
                    self.consecutive_h += 1
            else:
                self.seq.append(-1)
            pre = i

    def get_next(self):
        return self.seq[self.cur_index]

    def reset(self, seq=None):
        if not seq:
            seq = self.generate_seq()
        self.hp = np.zeros((self.grid_size, self.grid_size))
        self.order = collections.OrderedDict()
        self.pos = np.zeros((self.grid_size, self.grid_size))
        self.set_seq(seq)
        self.cur_index = 0
        self.cur_position = None  # the position of the last molecule added
        self.done = False
        return self.get_state()

    def get_state(self):
        if not self.cur_position:
            return np.zeros((1,self.obs_size * 2 + 1, self.obs_size * 2 + 1))
        x, y = self.cur_position
        return np.array([self.hp[x-self.obs_size:x+self.obs_size+1,y-self.obs_size:y+self.obs_size+1]])

    def render(self, episode, reward):
        x = []
        y = []
        xrange = [self.grid_size, 0]
        yrange = [self.grid_size, 0]
        for (i, j), value in self.order.items():
            x.append(i)
            y.append(j)
            xrange[0] = min(xrange[0], i)
            xrange[1] = max(xrange[1], i)
            yrange[0] = min(yrange[0], j)
            yrange[1] = max(yrange[1], j)
            if value == 1:
                plt.scatter(i, j, c='b', s=160, zorder=2)
                for dx, dy in [[-1, 0], [1, 0], [0, 1], [0, -1]]:
                    cx, cy = i + dx, j + dy
                    if self.order.get((cx, cy), 0) == 1:
                        plt.plot([i, cx], [j, cy], linewidth=3, color='r', zorder=1)
            else:
                plt.scatter(i, j, c='g', s=160, zorder=2)

        plt.plot(x, y, linewidth=3, color='black', zorder=1)
        plt.grid()
        plt.axis('scaled')
        size = 5
        plt.xticks(range(min(xrange[0], self.obs_size - size), max(self.obs_size + size, xrange[1]) + 1, 1))
        plt.yticks(range(min(yrange[0], self.obs_size - size), max(self.obs_size + size, yrange[1]) + 1, 1))
        plt.title("episode: {}, reward = {}".format(episode, reward))
        plt.show()

    def free_energy(self):
        adjacent_h = 0
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.hp[x][y] == 1:
                    for dx, dy in [[0, 1], [1, 0]]:
                        cx, cy = x + dx, y + dy
                        if 0 <= cx < self.grid_size and 0 <= cy < self.grid_size and self.hp[cx][cy] == 1:
                            adjacent_h += 1

        return self.consecutive_h - adjacent_h
