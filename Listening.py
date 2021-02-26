import numpy as np
import scipy.stats as stats

class Agent:
    def __init__(self, P, others = None):
        self.P = P
        self.state = 0
        self.P_t = self.P[self.state]
        self.others = others
        self.otherstate = 0
        self.politeness = {}
        
    def find_neighbors(self, bunch):
        self.others = [x for x in bunch if x != self]

    def assign_politeness(self, scale):
        dist = stats.lognorm(loc = 0, s = .5, scale = scale)
        # Independent option: not sure it's right.
        #interrupt = dist.rvs() - dist.median()
        #resist = 1 - dist.rvs() - (1 - dist.median())
        # Correlated option
        adjust = dist.rvs()
        interrupt = adjust - dist.median()
        resist = 1 - adjust - (1 - dist.median())
        d = {'interrupt': interrupt, 'resist': resist}
        #dist = stats.norm(loc = 0, scale = scale)
        #d = {}
        #d['interrupt'] = dist.rvs()
        #d['resist'] = dist.rvs()
        self.politeness = d
        
    def listen(self):
        listening = [other.state for other in self.others]
        self.otherstate = 1 if 1 in listening else 0
        
    def apply_politeness(self):
        if self.state == 0:
            # The probability that I will start
            self.P_t[1] *= 1 + self.otherstate*self.politeness['interrupt']
            self.P_t[0] = 1 - self.P_t[1]
        else:
            # The probability that I will stop
            self.P_t[0] *= 1 + self.otherstate*self.politeness['resist']
            self.P_t[1] = 1 - self.P_t[0]
    
    def step(self):
        self.P_t = self.P[self.state]
        self.apply_politeness()
        rnum = np.random.uniform()
        self.state = 0 if rnum < self.P_t[0] else 1
        
def simulation(P, T, N, ns, scale):
    G = [Agent(P[n]) for n in ns]
    for n in ns:
        G[n].find_neighbors(G)
    for n in ns:
        G[n].assign_politeness(scale)
    Y = np.full((T, N), np.nan)
    for t in range(T):
        y = [G[n].state for n in ns]
        Y[t] = y
        for n in ns:
            G[n].listen()
        for n in ns:
            G[n].step()
            
    return Y
