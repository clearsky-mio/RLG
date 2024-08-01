import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

import numpy as np

import os


class PolicyIteration:
    def __init__(
        self, render_mode="human", theta: np.float32 = 0.001, gamma: np.float32 = 0.9
    ) -> None:
        self.env = gym.make(
            "FrozenLake-v1",
            desc=generate_random_map(size=4),
            is_slippery=True,
            render_mode=render_mode,
        )
        self.env.reset()
        self.probability = np.full((16, 4), 0.25)
        self.V = np.zeros([16])

        self.theta = theta
        self.gamma = gamma

        self.env = self.env.unwrapped
        self.holes = set()
        self.ends = set()
        for s in self.env.P:
            for a in self.env.P[s]:
                for s_ in self.env.P[s][a]:
                    if s_[2] == 1.0:
                        self.ends.add(s_[1])
                    if s_[3] == True:
                        self.holes.add(s_[1])
        self.holes = self.holes - self.ends

    def step(self, state, action):
        self.env.s = state
        new_state, reward, _, _, _ = self.env.step(action)
        return new_state, reward

    def policy_evaluation(self):
        for _ in range(16):
            max_diff = 0
            for state in range(16):
                if state in self.holes or state in self.ends:
                    self.V[state] = 0
                    continue
                new_v = 0
                for action in range(4):
                    for _ in range(256):
                        new_state, reward = self.step(state, action)
                        new_v += (
                            self.probability[state][action]
                            * (reward + self.gamma * self.V[new_state])
                            / 256
                        )
                max_diff = max(max_diff, abs(self.V[state] - new_v))
                self.V[state] = new_v

    def policy_improvement(self):
        for _ in range(32):
            max_diff = 0
            for state in range(16):
                if state in self.holes or state in self.ends:
                    continue
                qsa = np.zeros([4])
                for action in range(4):
                    v = 0
                    for _ in range(128):
                        new_state, reward = self.step(state, action)
                        v += reward + self.gamma * self.V[new_state]
                    qsa[action] = v / 128
                if np.sum(qsa) != 0:
                    max_diff = max(
                        max_diff, abs(max(self.probability[state] - qsa / np.sum(qsa)))
                    )
                    self.probability[state] = qsa / np.sum(qsa)

    def policy_iteration(self):
        for _ in range(16):
            self.policy_evaluation()
            self.policy_improvement()

    def output_policy(self):
        action_desc = ["<", "v", ">", "^"]
        for row in range(4):
            for col in range(4):
                state = row * 4 + col
                if state in self.holes:
                    for _ in range(4):
                        print("*", end=" ")
                elif state in self.ends:
                    for _ in range(4):
                        print("E", end=" ")
                else:
                    most_prob = 0
                    for action in range(4):
                        most_prob = max(most_prob, self.probability[state][action])
                    for action in range(4):
                        if (
                            abs(self.probability[state][action] - most_prob)
                            < self.theta
                        ):
                            print(action_desc[action], end=" ")
                        else:
                            print("o", end=" ")
                print(" ", end=" ")
            print()

    def eval(self):
        env = gym.make(
            "FrozenLake-v1",
            desc=self.env.desc,
            is_slippery=True,
            render_mode="human",
        )
        env.reset()
        os.system("pause")
        observation = 0
        terminated = False
        while not terminated:
            p = np.random.rand()
            action = np.argmax(self.probability[observation])
            observation, _, terminated, _, _ = env.step(action)
        os.system("pause")


if __name__ == "__main__":
    os.system("pause")
    p = PolicyIteration(render_mode=None)
    p.policy_iteration()
    p.output_policy()
    while True:
        p.eval()
