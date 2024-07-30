from MAB.Game.MultiArmedBandit import MultiArmedBandit
from MAB.Solver.MABSolver import MABSolver

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import os
import time


class ThompsonSamplingSolver(MABSolver):
    def __init__(self, bandit: MultiArmedBandit):
        super().__init__(bandit)
        self.alpha = np.ones(bandit.action_count)
        self.beta = np.ones(bandit.action_count)

    def select_action(self):
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def run_one_step(self):
        action = self.select_action()
        reward = self.bandit.step(action)
        self.regrect += self.values.max() - reward
        self.update_estimation(action, reward)
        self.alpha[action] += reward
        self.beta[action] += 1 - reward
        return action, reward, self.regrect


def main():
    logger = SummaryWriter(
        os.path.join("logs", "Thompson-Sampling", time.strftime("%Y-%m-%d-%H-%M-%S"))
    )

    bandit = MultiArmedBandit(3)
    solver = ThompsonSamplingSolver(bandit)
    solver.learn(100000, logger)

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
