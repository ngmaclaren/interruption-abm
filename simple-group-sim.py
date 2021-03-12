import numpy as np
import pandas as pd
import InterruptionAnalysis as ia
import Independent as sim

#sim_choice = "indep"
#minutes = 10
#T = minutes*60*10

data = pd.read_csv('./data/timeseries.csv', index_col = 0)
numericcols = ['begin', 'end', 'dur', 'lat']
for col in numericcols:
    data[col] /= 100
Rs = pd.unique(data['gID'])

R = "YMY"#np.random.choice(Rs)
dat = data[data['gID'] == R] # leave it ~dat~ b/c col names reflect coding
r_ns = pd.unique(dat['pID'])
t_max = int(np.ceil(max(dat['end'])))
#T = range(t_max)
N = len(r_ns)
ns = list(range(N))

P_g = {n: ia.get_transition_matrix(dat, r_n) for n, r_n in zip(ns, r_ns)}

Y = sim.simulation(P_g, t_max, N, ns)#, basescale)
X = ia.Y_to_X(Y, ns)
X['scale'] = np.nan
X['gID'] = R
