import numpy as np

class Agent:
    def __init__(self, P):
        self.P = P
        self.state = 0
        self.P_t = self.P[self.state]

    def step(self):
        self.P_t = self.P[self.state]
        rnum = np.random.uniform()
        self.state = 0 if rnum < self.P_t[0] else 1
        
def simulation(P, T, N, ns):
    G = [Agent(P[n]) for n in ns]
    Y = np.full((T, N), np.nan)
    for t in range(T):
        y = [G[n].state for n in ns]
        Y[t] = y
        for n in ns:
            G[n].step()
            
    return Y
