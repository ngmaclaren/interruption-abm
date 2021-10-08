import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import InterruptionAnalysis as ia

from math import comb
from scipy.stats import kstest
from scipy.stats import norm
from scipy.stats import lognorm

def dhvg_local_clustering(g, v, direction):
    """
    Ref: Donges et al (2013)
    """
    # this only works because the node labels are integers whose order makes sense
    if direction == "forward":
        nbr = sorted(list(g.successors(v)))
    elif direction == "backward":
        nbr = sorted(list(g.predecessors(v)))

    deg = len(nbr)

    if deg < 2:
        c = 0
        return c
    else:
        count = 0
        for j in nbr:
            jnbr = list(g.successors(j))
            for k in nbr:
                if k in jnbr:
                    count += 1

        denominator = comb(deg, 2)
        c = count/denominator
        return c
    
data = pd.read_csv('./data/timeseries.csv', index_col = 0)
gIDs = pd.unique(data['gID'])

## Group
gks = {}
gpvals = {}
for gID in gIDs:
    d = data.loc[data['gID'] == gID, ]
    g = ia.directed_horizontal_visibility_graph(d)
    cr = [dhvg_local_clustering(g, v, "backward") for v in g.nodes]
    ca = [dhvg_local_clustering(g, v, "forward") for v in g.nodes]
    gks[gID], gpvals[gID] = kstest(cr, ca)

## Individual
iks = {}
ipvals = {}
for gID in gIDs:
    pIDs = pd.unique(data[data.gID == gID]['pID'])
    for pID in pIDs:
        dat = data.loc[data['pID'] == pID, ]
        if len(dat) >= 20:
            g = ia.directed_horizontal_visibility_graph(dat)
            
            kr = list(dict(g.in_degree()).values())
            ka = list(dict(g.out_degree()).values())
            cr = [dhvg_local_clustering(g, v, "backward") for v in g.nodes]
            ca = [dhvg_local_clustering(g, v, "forward") for v in g.nodes]
            iks[pID], ipvals[pID] = kstest(cr, ca)

# pvallist = list(ipvals.values())
# sigs = {k: v for k, v in ipvals.items() if v < .05}
# sigdat = data.loc[data['pID'].isin(sigs.keys()), ]

## Goal is now a two panel figure that collects the size of the data vector on the x axis and the p-value on the y axis; individual data on the left, group data on the right

count = data.groupby(["gID", "pID"])["begin"].count().to_frame(name = "count")
count.reset_index(inplace = True)

gdata = count.groupby("gID").agg(group_count = ("count", "sum"))
gks = pd.Series(gks)
gpvals = pd.Series(gpvals)
gdata["ks"] = gks
gdata["pval"] = gpvals

idata = count.set_index("pID")
iks = pd.Series(iks)
ipvals = pd.Series(ipvals)
idata["ks"] = iks
idata["pval"] = ipvals

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
# Indiv
ax1.scatter(x = idata["count"], y = idata["pval"],
            marker = ".", color = "k")
ax1.set_xlabel("Number of Events")
ax1.set_ylabel("K-S Test p-Value")
ax1.set_ylim(-0.05, 1.05)
ax1.axhline(y = 0.05, color = "red")
ax1.text(0.01, 0.99, "A", horizontalalignment = "left", verticalalignment = "top", transform = ax1.transAxes)
# Group
ax2.scatter(x = gdata["group_count"], y = gdata["pval"],
            marker = "o", color = "k")
ax2.set_xlabel("Number of Events")
ax2.set_ylabel("K-S Test p-Value")
ax2.set_ylim(-0.05, 1.05)
ax2.axhline(y = 0.05, color = "red")
ax2.text(0.01, 0.99, "B", horizontalalignment = "left", verticalalignment = "top", transform = ax2.transAxes)
fig.tight_layout()

fig.savefig("./img/equilibrium.pdf")
print("Done")
