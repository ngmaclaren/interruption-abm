import os
from string import ascii_lowercase as letters

import numpy as np
import pandas as pd
import InterruptionAnalysis as ia
import Independent

np.random.seed(12345)

# import reference data and convert time step to 1/10th second
data = pd.read_csv('./data/timeseries.csv', index_col = 0)
numeric_cols = ['begin', 'end', 'dur', 'lat']
for col in numeric_cols:
    data[col] = data[col]/100

# sim parameters for all sims
nsims = 100
gIDs = pd.unique(data["gID"])

# prepare subdirectory structure
# separate sims into subdirectories to speed file lookup
# if mimic-groups is level 1, there will be `nsims` level 2 directories with 33 level 3 directories each
savepath = "./data/simulations/mimic-groups"
if not os.path.isdir(savepath):
    os.mkdir(savepath)

# main loop
# The number of sims here means how many times through simulating each group once
for sim in range(nsims):
    print(sim)

    # prepare subdirectory for this sim
    simpath = savepath + f"/sim{sim:03d}" # sim number with leading zeros
    if not os.path.isdir(simpath):
        os.mkdir(simpath)
    
    # loop through the groups
    for gID in gIDs:
        print(gID)
        # filename for this gID
        filename = f"{simpath}/{gID}.csv"

        # set parameters for this group
        pIDs = pd.unique(data[data["gID"] == gID]["pID"])
        T = round(data[data["gID"] == gID]["end"].max())
        N = len(pIDs)
        ns = list(range(N))
        P_i = [ia.get_transition_matrix(data, pID) for pID in pIDs]

        # run sim itself
        Y = Independent.simulation(P_i, T, N, ns, oneagent = False)
        X = ia.Y_to_X(Y, ns)
        
        # clean up data frame for later ease of use
        X["gID"] = gID
        X["pID"] = X["pID"].apply(lambda x: pIDs[x])

        # save this sim in the right place
        X.to_csv(f"{filename}")
