from MAB.Game.MultiArmedBandit import MultiArmedBandit

import numpy as np


class MABSolver:
    def __init__(self, bandit: MultiArmedBandit):
        self.bandit = bandit
        self.counts = np.zeros([self.bandit.action_count], dtype=np.int32)
        self.values = np.zeros([self.bandit.action_count], dtype=np.float32)
        self.regrect = 0.0
        self.action_log = []
        self.reward_log = []
        self.regrect_log = []
        self.threshold = 0.8
        self.name = ""

    def select_action(self):
        """_summary_
        select action based on the current estimation of the action values

        Return:
            action: int, the selected action's index
        """
        raise NotImplementedError

    def update_estimation(self, action, reward):
        self.counts[action] += 1
        new_value = self.values[action] + (1 / self.counts[action]) * (
            reward - self.values[action]
        )
        self.values[action] = new_value

    def run_one_step(self):
        """_summary_
        select action, get reward, update estimation, and calculate regret for one step

        Return:
            action: int, the selected action's index
            reward: int, 0 means fail and 1 means success
            regrect: float, cumulative regret value
        """
        raise NotImplementedError

    def learn(self, n_steps: int = 10000):
        for _ in range(n_steps):
            _ = self.run_one_step()

    def learn(self, n_steps: int = 10000, logger=None):
        if logger is None:
            self.learn(n_steps)
        else:
            for i in range(n_steps):
                _ = self.run_one_step()
                if i % 100 == 0 or i == n_steps - 1:
                    for action in range(self.bandit.action_count):
                        logger.add_scalar(
                            self.name + "/action/" + str(action), self.counts[action], i
                        )
                        logger.add_scalar(
                            self.name + "/value/" + str(action), self.values[action], i
                        )
                    logger.add_scalar("regrect/" + self.name, self.regrect, i)

    @property
    def estimate_best_action(self):
        return np.argmax(self.values)

    @property
    def estimation(self):
        return self.values
