# currently set to nsims = 250 and dirlimit = 25; reset to 2500 and 250 for production runs

import os
from math import ceil
from string import ascii_lowercase as letters

import numpy as np
import pandas as pd
import InterruptionAnalysis as ia
import Independent as sim

np.random.seed(12345)

# import reference data and convert time step to 1/10th second
data = pd.read_csv('./data/timeseries.csv', index_col = 0)
numeric_cols = ['begin', 'end', 'dur', 'lat']
for col in numeric_cols:
    data[col] = data[col]/100

# keep only those agents analyzed in the DHVg analysis: those with |x| >= 20
sample = list(data.groupby("pID")["dur"].count().loc[lambda x: x >= 20].index)

# estimate /p/ and /q/ for each r_i and collect it into a data frame indexed by pID with columns "p" and "q"
rows = {}
for pID in sample:
    P_i = ia.get_transition_matrix(data, pID)
    p = P_i[0, 1]
    q = P_i[1, 0]
    rows[pID] = [p, q]
P = pd.DataFrame.from_dict(rows, orient = "index", columns = ["p", "q"])

# sim parameters for all sims
nsims = 250

####
## Timer Code
## prints to stdout the time it takes to simulate one reference individual nsims times
## does not save any data
# import time
# time1 = time.time()
# pID = np.random.choice(sample)
# row = P[P.index.isin([pID])]
# P_i = np.array([[1 - row["p"], row["p"]], [row["q"], 1 - row["q"]]])
# gID = pID[:3]
# T = round(data[data["gID"] == gID]["end"].max())
# # these two are the same for all in this file, but would change otherwise
# N = 1
# ns = list(range(N))
# for run in range(nsims):
#     Y = sim.simulation(P_i, T, N, ns, oneagent = True)
#     X = ia.Y_to_X(Y, ns)
# time2 = time.time()
# print(time2 - time1)
## end timer code
####

# prepare subdirectory structure
# separate sims into subdirectories to speed file lookup
savepath = "./data/simulations/mimic-agents"
if not os.path.isdir(savepath):
    os.mkdir(savepath)
dirlimit = 25
nsubdirs = ceil(nsims/dirlimit)

# main loop
for pID in sample:
    print(pID)
    # select reference individual and associated transition matrix
    row = P[P.index.isin([pID])]
    P_i = np.array([[1 - row["p"], row["p"]], [row["q"], 1 - row["q"]]])
    
    # set sim parameters for this sim
    gID = pID[:3]
    T = round(data[data["gID"] == gID]["end"].max())
    # these two are the same for all in this file, but would change otherwise
    N = 1
    ns = list(range(N))

    # prepare subdirectories for this pID
    pidpath = savepath + f"/{pID}"
    if not os.path.isdir(pidpath):
        os.mkdir(pidpath)
    for i in range(nsubdirs):
        subpath = f"{pidpath}/{letters[i]}"
        if not os.path.isdir(subpath):
            os.mkdir(subpath)

    # run the sims
    pointer = 0
    subdir = letters[pointer]
    for run in range(nsims):
        if run > 0 and run % dirlimit == 0:
            pointer += 1
            subdir = letters[pointer]
        Y = sim.simulation(P_i, T, N, ns, oneagent = True)
        X = ia.Y_to_X(Y, ns)
        # and store them in the appropriate place
        X.to_csv(f"{pidpath}/{subdir}/{pID}-{run}.csv")

