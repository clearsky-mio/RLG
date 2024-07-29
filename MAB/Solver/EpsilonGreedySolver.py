from MAB.Game.MultiArmedBandit import MultiArmedBandit
from Solver.MABSolver import MABSolver
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import os
import time

class EpsilonGreedySolver(MABSolver):
    def __init__(self, bandit: MultiArmedBandit, epsilon: float):
        super().__init__(bandit)
        self.epsilon = epsilon
        self.name = "EpsilonGreedySolver " + str(epsilon)
        
    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.bandit.action_count)
        return np.argmax(self.values)
    
    def run_one_step(self):
        action = self.select_action()
        reward = self.bandit.step(action)
        self.regrect += self.values.max() - reward
        self.update_estimation(action, reward)
        self.action_log.append(action)
        self.reward_log.append(reward)
        self.regrect_log.append(self.regrect)
        return action, reward, self.regrect
        

def main():
    k = np.random.randint(0, 20)
    bandit = MultiArmedBandit(10)
    epsilon = 0.2
    
    solver = EpsilonGreedySolver(bandit=bandit, epsilon=epsilon)
    solver.learn()
    
    print(f'estimation is {solver.estimation}')
    print(f'real probs is {bandit._probs}')
    print(f'choise have been try {solver.counts}')
    
    print(f'estimate best choise is {np.argmax(solver.estimation)}, its value is {solver.values[solver.estimate_best_action]}')
    print(f'real best choise is {np.argmax(bandit._probs)}, its value is {np.max(bandit._probs)}')
    

def epsilonCompare(n:int):
    logger = SummaryWriter(os.path.join("logs", "Epsilon-Greedy", time.strftime("%Y-%m-%d-%H-%M-%S")))
    
    bandit = MultiArmedBandit(5)
    epsilons = np.linspace(0, 1, n, dtype=np.float32)
    solvers = [EpsilonGreedySolver(bandit, epsilon) for epsilon in epsilons]
    for solver in solvers:
        solver.learn(10000, logger)
    

    
if __name__ == "__main__":
    # main()
    epsilonCompare(5)