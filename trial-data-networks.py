import os
import numpy as np
import pandas as pd
import networkx as nx

import InterruptionAnalysis as ia

data = pd.read_csv("./data/timeseries.csv", index_col = 0)
numeric_cols = ["begin", "end", "dur", "lat"]
for col in numeric_cols:
    data[col] = data[col]/100 # converts to 1/10th seconds
gIDs = pd.unique(data["gID"])

maindir = "./data/networks"

subd = "/emp/"
refnets = {f.replace(".gml", ""): nx.read_gml(maindir + subd + f) for f in os.listdir(maindir + subd)}

subds = ["mimic-groups", "synthetic-groups-independent", "synthetic-groups-listening", "synthetic-groups-dyadic"]

def network_collector(maindir, subd): # probably a good place for recursion...
    x = {str(d): {f.replace(".gml", ""): nx.read_gml(f"{maindir}/{subd}/{d}/{f}") for f in os.listdir(f"{maindir}/{subd}/{d}")} for d in os.listdir(f"{maindir}/{subd}")}
    return x

mimicnets = network_collector(maindir, subds[0])
synthinets = network_collector(maindir, subds[1])
synthlnets = network_collector(maindir, subds[2])
synthdnets = network_collector(maindir, subds[3])

# I now want to collect density, pagerank centralization, and average clustering for each network
def networkstats_collector(g):
    dens = nx.density(g)
    cent = ia.pagerank_centralization(g, alpha = 0.99, weight = "weight")
    avg_clust = nx.average_clustering(g, weight = "weight")
    return {"dens": dens, "cent": cent, "avg_clust": avg_clust}

refnetstats = {gID: networkstats_collector(refnets[gID]) for gID in gIDs}
refnetstats = pd.DataFrame.from_dict(refnetstats, orient = "index")
#refnetstats.reset_index(inplace = True)

def networkstats_organizer(X, key = None):
    X = pd.DataFrame.from_dict(X, orient = "index")
    X.reset_index(inplace = True)
    X.rename(columns = {"index": "gID"}, inplace = True)
    if key:
        X["sim"] = key
    return X

mimicnetstats = {sim: {gID: networkstats_collector(mimicnets[sim][gID]) for gID in gIDs} for sim in os.listdir(f"{maindir}/{subds[0]}")}
mimicnetstats = {key: networkstats_organizer(val, key) for key, val in mimicnetstats.items()}
mimicnetstats = pd.concat(list(mimicnetstats.values()))

synthinetstats = {sim: {gID: networkstats_collector(synthinets[sim][gID]) for gID in gIDs} for sim in os.listdir(f"{maindir}/{subds[0]}")}
synthinetstats = {key: networkstats_organizer(val, key) for key, val in synthinetstats.items()}
synthinetstats = pd.concat(list(synthinetstats.values()))

synthlnetstats = {sim: {gID: networkstats_collector(synthlnets[sim][gID]) for gID in gIDs} for sim in os.listdir(f"{maindir}/{subds[0]}")}
synthlnetstats = {key: networkstats_organizer(val, key) for key, val in synthlnetstats.items()}
synthlnetstats = pd.concat(list(synthlnetstats.values()))

synthdnetstats = {sim: {gID: networkstats_collector(synthdnets[sim][gID]) for gID in gIDs} for sim in os.listdir(f"{maindir}/{subds[0]}")}
synthdnetstats = {key: networkstats_organizer(val, key) for key, val in synthdnetstats.items()}
synthdnetstats = pd.concat(list(synthdnetstats.values()))

from scipy import stats
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize = [7, 6])

colors = ["#5445b1", "#749dae", "#f3c483", "#5c1a33", "#cd3341","#f7dc6a"]
labels = ["Empirical", "Mimic", "Independent", "Listening", "Dyadic"]
#col = "dens"
#col = "cent"
col = "avg_clust"

xs = [refnetstats[col], mimicnetstats[col], synthinetstats[col], synthlnetstats[col], synthdnetstats[col]]

for color, label, x in zip(colors, labels, xs):
    kde = stats.gaussian_kde(x)
    newx = np.linspace(min(x), max(x), num = 100)
    newy = kde(newx)
    lwd = 3 if label == "Empirical" else 2
    zorder = 2.5 if label == "Empirical" else 2
    ax.plot(newx, newy, color = color, linewidth = lwd, label = label, zorder = zorder)

ax.legend(loc = "upper right")
ax.set_xlabel("Interruption Network Density")
ax.set_ylabel("Kernel Density Estimate")

fig.show()
