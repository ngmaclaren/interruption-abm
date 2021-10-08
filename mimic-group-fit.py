import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import InterruptionAnalysis as ia

# import base data for reference individuals
data = pd.read_csv('./data/timeseries.csv', index_col = 0)
numeric_cols = ['begin', 'end', 'dur', 'lat']
for col in numeric_cols:
    data[col] = data[col]/100 # converts to 1/10th seconds

rootpath = "./data/simulations/mimic-groups"
sims = os.listdir(rootpath)

gIDs = pd.unique(data["gID"])

for sim in sims:
    simpath = rootpath + "/" + sim

    for gID in gIDs:
        gidpath = simpath + "/" + gID
        for root, dirs, files in os.walk(pidpath):
            if not files:
                continue
            for f in files:
                X = pd.read_csv(os.path.join(root, f))




#####
## Pausing here: I don't need to redo these files before May 20, I need to generate the sims and make a new CCS figure before May 20.
#####
