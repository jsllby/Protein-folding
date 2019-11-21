import collections
import random


class Q_Learning:
    def __init__(self):
        self.Q = collections.defaultdict(int)
        self.lr = 0.01
        self.epsilon = 0.2
        self.discount = 0.9
        self.actions = [0, 1, 2, 3]

    def select_action(self, state, explore=True):
        if explore:
            pr = random.random()
            if pr <= self.epsilon:
                return random.choice(self.actions)
        best_q = float('-inf')
        best_a = None
        for a in self.actions:
            if self.Q[(state, a)] > best_q:
                best_q = self.Q[(state, a)]
                best_a = a
        return best_a

    def learn(self):
        pass

    def store_transition(self, state, action, next_state, reward):
        next_best_q = float('-inf')
        for a in self.actions:
            next_best_q = max(next_best_q, self.Q[(next_state, a)])
        sample = reward + self.discount * next_best_q
        self.Q[(state, action)] += self.lr * (sample - self.Q[(state, action)])
