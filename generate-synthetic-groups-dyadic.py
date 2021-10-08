import os
from string import ascii_lowercase as letters

import numpy as np
import pandas as pd
from scipy import stats

import InterruptionAnalysis as ia

import Dyadic

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
savepath = "./data/simulations/synthetic-groups-dyadic"
if not os.path.isdir(savepath):
    os.mkdir(savepath)

# prepare probability distributions for p and q
dists = pd.read_csv("./data/fits.csv", index_col = 0)

dist_choice = "theory"
if dist_choice == "theory":
    d1 = dists[(dists["transition"] == "AB") & (dists["dist"] == "weibull_min")]
    d2 = dists[(dists["transition"] == "BA") & (dists["dist"] == "lognorm")]
    chosen_dists = pd.concat([d1, d2])
elif dist_choice == "beta":
    chosen_dists = dists[dists["dist"] == "beta"]
else:
    chosen_dists = dists.loc[dists.groupby("transition")["ΔAIC"].idxmin()]

chosen_dists.set_index("transition", inplace = True)
fitrows = ["AB", "BA"]
fits = {}
for row in fitrows:
    dist = getattr(stats, chosen_dists.loc[row, "dist"])
    args = [chosen_dists.loc[row, "arg1"], chosen_dists.loc[row, "arg2"]]
    arg = [a for a in args if ~np.isnan(a)]
    loc = chosen_dists.loc[row, "loc"]
    scale = chosen_dists.loc[row, "scale"]
    if arg:
        fit = dist(loc = loc, scale = scale, *arg)
    else:
        fit = dist(loc = loc, scale = scale)
    fits[row] = fit
   
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

        p_i = fits["AB"].rvs(size = N)
        q_i = fits["BA"].rvs(size = N)
        P_i = [
            np.array([[1 - p, p],
                      [q, 1 - q]]) for p, q in zip(p_i, q_i)
        ]

        # run sim itself
        Y = Dyadic.simulation(P_i, T, N, ns, scale = 1e-5)
        X = ia.Y_to_X(Y, ns)
        
        # clean up data frame for later ease of use
        X["gID"] = gID

        # save this sim in the right place
        X.to_csv(f"{filename}")
