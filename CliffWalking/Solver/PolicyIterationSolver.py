from CliffWalking.Env.CliffWalkingEnv import CliffWalkingEnv

import numpy as np


class PolicyIterationSolver:
    def __init__(self, env: CliffWalkingEnv, gamma: float, theta: float):
        self.env = env
        self.gamma = gamma
        self.theta = theta

        self.policy = np.ones((env.row, env.col, len(env.action))) / len(env.action)
        self.V = np.zeros((env.row, env.col))

    def policy_evaluation(self):
        eval_times = 0
        while True:
            delta = 0
            for i in range(self.env.row):
                for j in range(self.env.col):
                    v = self.V[i][j]
                    if i == 0 and j == self.env.col - 1:
                        self.V[i][j] = self.env.reward[i][j]
                        continue
                    new_v = 0
                    for action in range(len(self.env.action)):
                        if not self.env.check_action_valid([i, j], action):
                            continue
                        next_state, reward = self.env.step([i, j], action)
                        new_v += self.policy[i][j][action] * (
                            reward + self.gamma * self.V[next_state[0]][next_state[1]]
                        )
                    self.V[i][j] = new_v
                    delta = max(delta, abs(v - self.V[i][j]))
            if delta < self.theta:
                print("eval_times: ", eval_times + 1)
                break
            else:
                eval_times += 1

    def policy_improvement(self):
        for row in range(self.env.row):
            for col in range(self.env.col):
                qsas = np.zeros(len(self.env.action))
                for action in range(len(self.env.action)):
                    if not self.env.check_action_valid([row, col], action):
                        qsas[action] = -10000
                    else:
                        next_state, reward = self.env.step([row, col], action)
                        qsas[action] = (
                            reward + self.gamma * self.V[next_state[0]][next_state[1]]
                        )
                best_action = np.argmax(qsas)
                cntq = 0
                for action in range(len(self.env.action)):
                    if qsas[action] == qsas[best_action]:
                        cntq += 1
                for action in range(len(self.env.action)):
                    if qsas[action] == qsas[best_action]:
                        self.policy[row][col][action] = 1 / cntq
                    else:
                        self.policy[row][col][action] = 0

    def policy_iteration(self):
        iter_times = 0
        while True:
            self.policy_evaluation()
            old_policy = self.policy.copy()
            self.policy_improvement()
            if np.sum(np.abs(self.policy - old_policy)) == 0:
                print("iter_times: ", iter_times + 1)
                break
            else:
                iter_times += 1

    def output(self):
        # unvalid action as *
        # valid action as o
        # best action as < > ^ v
        # goal as G
        # show as * o x o, mean left, right, up, down
        for i in range(self.env.row):
            for j in range(self.env.col):
                if i == 0 and j == self.env.col - 1:
                    print("G", end=" ")
                    continue
                elif i == 0 and j > 0:
                    for _ in range(4):
                        print("x", end=" ")
                    print(" ", end=" ")
                    continue
                best_action = np.argmax(self.policy[i][j])
                for action in range(len(self.env.action)):
                    if not self.env.check_action_valid([i, j], action):
                        print("*", end=" ")
                    else:
                        if self.policy[i][j][action] == self.policy[i][j][best_action]:
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
    solver = PolicyIterationSolver(env, 0.9, 0.0001)
    solver.policy_iteration()
    solver.output()
