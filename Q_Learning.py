import collections
import math
import random


class Q_Learning:
    def __init__(self):
        self.Q = collections.defaultdict(int)
        self.lr = 0.01
        self.max_epsilon = 1
        self.min_epsilon = 0.3
        self.decay_steps = 2000
        self.discount = 0.9
        self.actions = [0, 1, 2, 3]
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
        self.dirs = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        self.collision_penalty = collision_penalty
        self.trap_penalty = trap_penalty

    def isTrapped(self):
        for dx, dy in self.dirs:
            if (self.cur_position[0] + dx, self.cur_position[1] + dy) not in self.state:
                return False
        return True

    def step(self, action):
        if not self.cur_position:
            x = y = self.grid_size // 2
        else:
            x = self.dirs[action][0] + self.cur_position[0]
            y = self.dirs[action][1] + self.cur_position[1]

        reward = None
        if (x, y) not in self.state:
            self.append(x, y)
            if self.cur_index == len(self.seq):
                self.done = True
                reward = -self.free_energy()
            elif self.isTrapped():
                self.done = True
                reward = (-len(self.seq) + len(self.state)) * self.trap_penalty - self.free_energy()
            else:
                reward = 0
        else:
            reward = -self.collision_penalty

        if self.state_index == -1:
            self.state_index = 0
        else:
            self.state_index = 4 * self.state_index + 1 + action

        return self.state_index, reward, self.done

    def append(self, x, y):
        self.state[(x, y)] = self.seq[self.cur_index]
        self.cur_index += 1
        self.cur_position = [x, y]

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

    def setSeq(self, seq):
        self.seq = []
        for i in seq:
            if i == 'H' or i == 'h':
                self.seq.append(1)
            else:
                self.seq.append(-1)

    def getNext(self):
        return self.seq[self.cur_index]

    def reset(self):
        self.cur_index = 0
        self.cur_position = None  # the position of the last molecule added

        self.state = collections.OrderedDict()
        self.state_index = -1

        self.done = False

    def render(self):
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cur = self.state.get((x, y), 0)
                if cur == 1:
                    print('H', end=' ')
                elif cur == -1:
                    print('P', end=' ')
                else:
                    print('*', end=' ')

            print()

    def free_energy(self):
        consecutive_h = 0
        adjacent_h = 0
        pre = None
        for (x, y), value in self.state.items():
            if value == 1:
                if pre == 1:
                    consecutive_h += 1
                for dx, dy in self.dirs:
                    cx, cy = x + dx, y + dy
                    if self.state.get((cx, cy), 0) == 1:
                        adjacent_h += 1
            pre = value
        return consecutive_h - adjacent_h // 2


def evaluate(env, agent):
    actions = []
    env.reset()
    done = False
    reward = None
    while not done:
        state = env.state_index
        action = agent.select_action(state, explore=False)
        actions.append(action)
        next_state, reward, done = env.step(action)
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
max_episode = 1000000
collision_penalty = 1
trap_penalty = 5
min_learn_size = 10
learn_step = 5000

# initial enviroment
env = Env(grid_size, collision_penalty, trap_penalty)

env.setSeq("hhppppphhppphppphp")
# initial agent
agent = Q_Learning()

step = 0
evaluate_interval = 10000
for episode in range(max_episode):
    env.reset()  # env generates a random sequence
    done = False
    reward = None
    while not done:
        state = env.state_index
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.store_transition(state, action, next_state, reward)
        step += 1
    if episode % evaluate_interval == 0 and done:
        reward, _ = evaluate(env, agent)
        print("episode {}, reward = {}".format(episode, reward))
print(evaluate(env, agent))
