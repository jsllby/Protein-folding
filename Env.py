import random
import collections
import numpy as np


class Env:
    def __init__(self, grid_size, collision_penalty, trap_penalty):
        self.grid_size = grid_size
        self.dirs = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        self.collision_penalty = collision_penalty
        self.trap_penalty = trap_penalty
        self.state = np.zeros((self.grid_size, self.grid_size))

    def is_trapped(self):
        for dx, dy in self.dirs:
            x = self.cur_position[0] + dx
            y = self.cur_position[1] + dy
            if self.state[x][y] == 0:
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

        if self.state[x][y] == 0:
            self.append(x, y)
            if self.cur_index == len(self.seq):
                self.done = True
                reward = -self.free_energy()
            elif self.is_trapped():
                self.done = True
                reward = -(len(self.seq) - self.cur_index) * self.trap_penalty - self.free_energy()
            else:
                reward = 0
        else:
            reward = -self.collision_penalty

        return self.state.reshape((1, -1)).squeeze(), reward, self.done

    def append(self, x, y):
        self.state[x][y] = self.seq[self.cur_index]
        self.cur_index += 1
        self.cur_position = [x, y]

    def generate_seq(self):
        n = random.randint(1, 20)
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

    def reset(self, seq):
        self.set_seq(seq)
        self.cur_index = 0
        self.cur_position = None  # the position of the last molecule added
        self.done = False
        self.state = np.zeros((self.grid_size, self.grid_size))
        return self.state.reshape((1, -1)).squeeze()

    def render(self):
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.state[x][y] == 1:
                    print('H', end=' ')
                elif self.state[x][y] == -1:
                    print('P', end=' ')
                else:
                    print('*', end=' ')

            print()

    def free_energy(self):
        adjacent_h = 0
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.state[x][y] == 1:
                    for dx, dy in [[0, 1], [1, 0]]:
                        cx, cy = x + dx, y + dy
                        if 0 <= cx < self.grid_size and 0 <= cy < self.grid_size and self.state[cx][cy] == 1:
                            adjacent_h += 1

        return self.consecutive_h - adjacent_h
