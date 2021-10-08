import os
import numpy as np
import pandas as pd
import networkx as nx

import InterruptionAnalysis as ia

data = pd.read_csv("./data/timeseries.csv", index_col = 0)
numeric_cols = ["begin", "end", "dur", "lat"]
for col in numeric_cols:
    data[col] = data[col]/100 # converts to 1/10th seconds
# votedata = pd.read_csv("./data/vote-data.csv")
# N = len(votedata)
# group_sizes = votedata.groupby("gID")["pID"].count()
# effective_group_sizes = data.groupby("gID")["pID"].nunique()
gIDs = pd.unique(data["gID"])
maindir = "./data/networks"

## empirical
# does it need to be different?
# test for and create subdir in networks
empdir = maindir + "/emp"
if not os.path.isdir(empdir):
    os.mkdir(empdir)
for gID in gIDs:
    pIDs = pd.unique(data[data['gID'] == gID]['pID']) ###!!!!!! Ignoring silent group members!
    igb = ia.interruption_network_pandas(data, pIDs, use = 'both')
    # save the network to the right place
    nx.write_gml(igb, f"{empdir}/{gID}.gml", stringizer = str)
    
## mimic and synthetic
# can they follow the same dirwalk?
datapaths = ["./data/simulations/mimic-groups", "./data/simulations/synthetic-groups-independent",
             "./data/simulations/synthetic-groups-listening", "./data/simulations/synthetic-groups-dyadic"]

netpaths = ["./data/networks/mimic-groups", "./data/networks/synthetic-groups-independent",
            "./data/networks/synthetic-groups-listening", "./data/networks/synthetic-groups-dyadic"]

for datapath, netpath in zip(datapaths, netpaths):
    print(datapath)

    if not os.path.isdir(netpath):
        os.mkdir(netpath)
    
    for root, dirs, files in os.walk(datapath):
        if not files:
            continue
        sim = root.replace(datapath + "/", "")
        print(sim)
        simdir = netpath + f"/{sim}"
        if not os.path.isdir(simdir):
            os.mkdir(simdir)
        for f in files:
            X = pd.read_csv(os.path.join(root, f), index_col = 0)    
            X["sim"] = sim
            pIDs = pd.unique(X["pID"])
            sgb = ia.interruption_network_pandas(X, pIDs, use = "both")
            nx.write_gml(sgb, f"{simdir}/{f.replace('.csv', '')}.gml", stringizer = str)
            
