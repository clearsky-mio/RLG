from CliffWalking.Env.CliffWalkingEnv import CliffWalkingEnv

import numpy as np


class ValueIterationSolver:
    def __init__(
        self, env: CliffWalkingEnv, theta: np.float32, gamma: np.float32
    ) -> None:
        self.env = env
        self.V = np.zeros((env.row, env.col))
        self.theta = theta
        self.gamma = gamma

    def value_iteration(self):
        iteration_time = 0
        while True:
            max_diff = 0
            for row in range(self.env.row):
                for col in range(self.env.col):
                    if row == 0:
                        if col == self.env.col - 1:
                            self.V[row][col] = 0
                            continue
                        elif col > 0:
                            self.V[row][col] = -1000
                            continue
                    qsa = np.zeros([self.env.action_count])
                    for action in range(self.env.action_count):
                        if not self.env.check_action_valid([row, col], action):
                            qsa[action] = -10000
                            continue
                        new_state, reward = self.env.step([row, col], action)
                        qsa[action] = (
                            reward + self.gamma * self.V[new_state[0]][new_state[1]]
                        )
                    new_v = np.max(qsa)
                    max_diff = max(max_diff, abs(new_v - self.V[row][col]))
                    self.V[row][col] = new_v
            if max_diff < self.theta:
                print("iteration times: ", iteration_time + 1)
                break
            else:
                iteration_time += 1

    def get_policy(self):
        for row in range(self.env.row):
            for col in range(self.env.col):
                print(self.V[row][col], end=" ")
            print()
        print()
        for row in range(self.env.row):
            for col in range(self.env.col):
                if row == 0 and col == self.env.col - 1:
                    print("G", end=" ")
                    continue
                elif row == 0 and col > 0:
                    for _ in range(4):
                        print("x", end=" ")
                    print(" ", end=" ")
                    continue
                best_V = -10000
                for action in range(self.env.action_count):
                    if not self.env.check_action_valid([row, col], action):
                        continue
                    new_state, _ = self.env.step([row, col], action)
                    best_V = max(best_V, self.V[new_state[0]][new_state[1]])
                for action in range(self.env.action_count):
                    if not self.env.check_action_valid([row, col], action):
                        print("*", end=" ")
                    else:
                        new_state, _ = self.env.step([row, col], action)
                        v = self.V[new_state[0]][new_state[1]]
                        if v == best_V:
                            if action == 0:
                                print("<", end=" ")
                            elif action == 1:
                                print(">", end=" ")
                            elif action == 2:
                                print("^", end=" ")
                            else:
                                print("v", end=" ")
                        else:
                            print("o", end=" ")
                print(" ", end=" ")
            print()


if __name__ == "__main__":
    env = CliffWalkingEnv(4, 12)
    solver = ValueIterationSolver(env, 0.0001, 0.9)
    solver.value_iteration()
    solver.get_policy()
