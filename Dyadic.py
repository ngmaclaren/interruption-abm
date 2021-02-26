import numpy as np
import scipy.stats as stats

class Agent:
    def __init__(self, P, others = None):
        self.P = P
        self.state = 0
        self.P_t = self.P[self.state]
        self.others = others
        self.otherstate = {}
        self.status = {}

    def find_neighbors(self, bunch):
        self.others = [x for x in bunch if x != self]

    def assign_status(self, scale):
        dist = stats.lognorm(loc = 0, s = .5, scale = scale)
        #dist = stats.norm(loc = 0, scale = scale) # scale is a free parameter
        for other in self.others:
            #interrupt = dist.rvs() - dist.median()
            #resist = 1 - dist.rvs() - (1 - dist.median())
            adjust = dist.rvs()
            interrupt = adjust - dist.median()
            resist = 1 - adjust - (1 - dist.median())
            d = {'interrupt': interrupt, 'resist': resist}
            # d = {}
            # d['interrupt'] = dist.rvs()
            # d['resist'] = dist.rvs()
            self.status[other] = d

    def listen(self):
        for other in self.others:
            self.otherstate[other] = other.state
        
    def apply_status(self):
        if self.state == 0:
            self.P_t[1] *= 1 + sum(
                [self.otherstate[other]*self.status[other]['interrupt'] for other in self.others])
            self.P_t[0] = 1 - self.P_t[1]
        else:
            self.P_t[0] *= 1 + sum(
                [self.otherstate[other]*self.status[other]['resist'] for other in self.others])
            self.P_t[1] = 1 - self.P_t[0]

    def step(self):
        self.P_t = self.P[self.state]
        self.apply_status()
        rnum = np.random.uniform()
        self.state = 0 if rnum < self.P_t[0] else 1
        
def simulation(P, T, N, ns, scale):
    G = [Agent(P[n]) for n in ns]
    for n in ns:
        G[n].find_neighbors(G)
    for n in ns:
        G[n].assign_status(scale)
    Y = np.full((T, N), np.nan)
    for t in range(T):
        y = [G[n].state for n in ns]
        Y[t] = y
        for n in ns:
            G[n].listen()
        for n in ns:
            G[n].step()
            
    return Y
