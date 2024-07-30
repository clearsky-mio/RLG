import numpy as np
from enum import Enum


class CliffWalkingEnv:
    def __init__(self, col, row) -> None:
        self.col = col
        self.row = row
        self.action_count = 4
        self.action = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.start = [0, 0]
        self.reward = np.array([[-1] * col] * row)
        self.reward[0][1 : col - 1] = -100

    @property
    def get_init_state(self):
        return self.start.copy()

    def check_action_valid(self, state, action):
        next_state = [
            state[0] + self.action[action][0],
            state[1] + self.action[action][1],
        ]
        if (
            next_state[0] >= 0
            and next_state[0] < self.row
            and next_state[1] >= 0
            and next_state[1] < self.col
        ):
            return True
        return False

    def step(self, state, action):
        next_state = [
            state[0] + self.action[action][0],
            state[1] + self.action[action][1],
        ]
        reward = self.reward[next_state[0]][next_state[1]]
        return next_state, reward
