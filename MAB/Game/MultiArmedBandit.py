import numpy as np

class MultiArmedBandit:
    
    def __init__(self, n_actions=5):
        self._probs = np.random.random(n_actions)
        
    def step(self, action):
        return 1 if (np.random.random() < self._probs[action]) else 0
    
    @property
    def action_count(self):
        return len(self._probs)
    

def main():
    bandit = MultiArmedBandit()
    print(bandit.action_count)
    print(bandit._probs)
    
    
if __name__ == "__main__":
    main()