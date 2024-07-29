from MAB.Game.MultiArmedBandit import MultiArmedBandit
from MAB.Solver.MABSolver import MABSolver
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import os
import time


class UpperConfidenceBoundSolver(MABSolver):
    def __init__(self, bandit: MultiArmedBandit, c: float):
        super().__init__(bandit)
        self.c = c  # exploration parameter
        self.step = 0
        self.name = "UpperConfidenceBoundSolver " + str(c)

    def select_action(self):
        ucb = self.values + self.c * np.sqrt(
            np.log(self.step + 1) / (2 * (self.counts + 1))
        )
        # print(f"values is {self.values}")
        # print(f"step is {self.step}")
        # print(f"n is {self.counts + 1}")
        # print(f"ucb is {ucb}")
        return np.argmax(ucb)

    def run_one_step(self):
        action = self.select_action()
        # print(f"select {action}")
        reward = self.bandit.step(action)
        self.regrect += self.values.max() - reward
        self.update_estimation(action, reward)
        self.step += 1
        self.action_log.append(action)
        self.reward_log.append(reward)
        self.regrect_log.append(self.regrect)
        return action, reward, self.regrect


def main():
    logger = SummaryWriter(
        os.path.join("logs", "UCB", time.strftime("%Y-%m-%d-%H-%M-%S"))
    )

    bandit = MultiArmedBandit(5)
    uncertainties = np.array([0.5, 1, 2, 4], dtype=np.float32)
    solvers = [
        UpperConfidenceBoundSolver(bandit, uncertaintie)
        for uncertaintie in uncertainties
    ]

    for solver in solvers:
        solver.learn(10000, logger)
        print("*" * 80)
        print(f"estimation is {solver.estimation}")
        print(f"real probs is {bandit._probs}")
        print(f"choise have been try {solver.counts}")

        print(
            f"estimate best choise is {np.argmax(solver.estimation)}, its value is {solver.values[solver.estimate_best_action]}"
        )
        print(
            f"real best choise is {np.argmax(bandit._probs)}, its value is {np.max(bandit._probs)}"
        )
        print("*" * 80)


if __name__ == "__main__":
    main()
