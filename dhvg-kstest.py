## Largely getting the same results with the KS test, but it will be easier to explain and the results are cleaner.
## Try: simulate data on the same scale
##      changing t---shouldn't matter for a horizontal visibility graph
##      making a regular visibility graph instead? I don't think it's appropriate, necessarily, but I can't remember right now why I thought that.

# Saturday, January 30, 2021, 0913
# can replicate the data from the book.
## for autoregressive, should be able to scale by using the mean of all 'dur' as the starting value
## and scale using lognorm.rvs() fit to the empirical data
##
## I'm not sure how to scale the chaotic data yet.

import numpy as np
import pandas as pd
import networkx as nx
import InterruptionAnalysis as ia

from math import comb
from scipy.stats import kstest
from scipy.stats import norm
from scipy.stats import lognorm

def dhvg_local_clustering(g, v, direction):
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
pvals = {}
for gID in gIDs:
    d = data.loc[data['gID'] == gID, ]
    g = ia.directed_horizontal_visibility_graph(d)
    cr = [dhvg_local_clustering(g, v, "backward") for v in g.nodes]
    ca = [dhvg_local_clustering(g, v, "forward") for v in g.nodes]
    pvals[gID] = kstest(cr, ca)[1]

## Individual
pvalues = {}
for gID in gIDs:
    pIDs = pd.unique(data[data.gID == gID]['pID'])
    for pID in pIDs:
        dat = data.loc[data['pID'] == pID, ]
        if len(dat) > 20:
            g = ia.directed_horizontal_visibility_graph(dat)
            
            kr = list(dict(g.in_degree()).values())
            ka = list(dict(g.out_degree()).values())
            cr = [dhvg_local_clustering(g, v, "backward") for v in g.nodes]
            ca = [dhvg_local_clustering(g, v, "forward") for v in g.nodes]
            pvalues[pID] = kstest(cr, ca)[1]

pvallist = list(pvalues.values())
sigs = {k: v for k, v in pvalues.items() if v < .05}
sigdat = data.loc[data['pID'].isin(sigs.keys()), ]

# Can maybe do the lognorm random draws as a reference: same distribution but essentially perfectly reversible. So we can see that the groups do vary in that they are not perfect realizations of a lognormal process. 
scale = np.exp(data['dur'].mean())
s = data['dur'].std()
loc = data['dur'].mean()
ξ = lognorm(s = s, loc = loc, scale = scale) #### This isn't really autoregressive...

d = pd.DataFrame({'begin': np.arange(0, 200, 1, int), 'dur': ξ.rvs(200)})
g = ia.directed_horizontal_visibility_graph(d)
kr = list(dict(g.in_degree()).values())
ka = list(dict(g.out_degree()).values())
cr = [dhvg_local_clustering(g, v, "backward") for v in g.nodes]
ca = [dhvg_local_clustering(g, v, "forward") for v in g.nodes]
print()
print("Degrees")
print(kstest(kr, ka))
print("Clustering")
print(kstest(cr, ca))


###############
## STOP HERE ##
###############

    
g = ia.directed_horizontal_visibility_graph(data[data['gID'] == 'MTU'])#'YMY'])
kr = list(dict(g.in_degree()).values())
ka = list(dict(g.out_degree()).values())
cr = [dhvg_local_clustering(g, v, "backward") for v in g.nodes]
ca = [dhvg_local_clustering(g, v, "forward") for v in g.nodes]

print(kstest(cr, ca))
print(kstest(kr, ka))

## add linear noise:
dat = data.copy()
d = dat.loc[dat['gID'] == 'YMY',].sort_values('begin').reset_index()
def noise(x, t, p):
    newx = x + t*p
    return newx
d['dur'] = noise(d['dur'], d.index, 100) # somewhere between 10 and 100 this produces a sig value
g = ia.directed_horizontal_visibility_graph(d)

kr = list(dict(g.in_degree()).values())
ka = list(dict(g.out_degree()).values())
cr = [dhvg_local_clustering(g, v, "backward") for v in g.nodes]
ca = [dhvg_local_clustering(g, v, "forward") for v in g.nodes]

print()
print("Degrees")
print(kstest(kr, ka))
print("Clustering")
print(kstest(cr, ca))



# add chaotic noise
# this will be the "nonlinear-deterministic Hénon map" from the reference (Donges et al 2013)
# xt = A − x2 t−1 + Byt−1 , yt = xt−1
# A = 1.4 and B = 0.3
# I'm not going to do the burn in because I want x to be
d = data.loc[dat['gID'] == 'YMY',].sort_values('begin').reset_index()

scale = np.full(len(d), np.nan)
x = np.random.random()
y = np.random.random()
a = 1.4
b = 0.3
runs = 2000
for r in range(runs):
    newx = a - x**2 + b*y
    newy = x
    x, y = newx, newy
    check = r - (runs - len(scale))
    if check >= 0:
        scale[check] = x

d['dur'] = d['dur']*scale
g = ia.directed_horizontal_visibility_graph(d)

kr = list(dict(g.in_degree()).values())
ka = list(dict(g.out_degree()).values())
cr = [dhvg_local_clustering(g, v, "backward") for v in g.nodes]
ca = [dhvg_local_clustering(g, v, "forward") for v in g.nodes]

print()
print("Degrees")
print(kstest(kr, ka))
print("Clustering")
print(kstest(cr, ca))

# try the logistic map #### not working
d = data.loc[dat['gID'] == 'YMY',].sort_values('begin').reset_index()
scale = np.full(len(d), np.nan)
x = np.random.random()
r = 3.7
runs = 2000
for r in range(runs):
    newx = r*x*(1 - x)
    x = newx
    check = r - (runs - len(scale))
    if check >= 0:
        scale[check] = x
d['dur'] = d['dur']*(scale)
g = ia.directed_horizontal_visibility_graph(d)

kr = list(dict(g.in_degree()).values())
ka = list(dict(g.out_degree()).values())
cr = [dhvg_local_clustering(g, v, "backward") for v in g.nodes]
ca = [dhvg_local_clustering(g, v, "forward") for v in g.nodes]

print()
print("Degrees")
print(kstest(kr, ka))
print("Clustering")
print(kstest(cr, ca))

dat = data.loc[data['gID'] == 'YMY',].sort_values('begin').reset_index()
longest = dat.sort_values('dur').tail(20)
shortest = dat.sort_values('dur').head(20)
first = dat.sort_values('begin').head(20)
last = dat.sort_values('begin').tail(20)

#d = dat[~(dat['begin'].isin(longest['begin']))]
#d.loc[d['begin'].isin(longest['begin']), 'dur'] *= 2
marker = d['end'].max() * .5
d.loc[d['begin']  > marker, 'dur'] *= 10
g = ia.directed_horizontal_visibility_graph(d)

kr = list(dict(g.in_degree()).values())
ka = list(dict(g.out_degree()).values())
cr = [dhvg_local_clustering(g, v, "backward") for v in g.nodes]
ca = [dhvg_local_clustering(g, v, "forward") for v in g.nodes]

print()
print("Degrees")
print(kstest(kr, ka))
print("Clustering")
print(kstest(cr, ca))


pvalues = {}
for gID in gIDs:
    pIDs = pd.unique(data[data.gID == gID]['pID'])
    for pID in pIDs:
        dat = data.loc[data['pID'] == pID, ]
        if len(dat) > 20:
            g = ia.directed_horizontal_visibility_graph(dat)
            
            kr = list(dict(g.in_degree()).values())
            ka = list(dict(g.out_degree()).values())
            cr = [dhvg_local_clustering(g, v, "backward") for v in g.nodes]
            ca = [dhvg_local_clustering(g, v, "forward") for v in g.nodes]
            pvalues[pID] = kstest(cr, ca)[1]

pvallist = list(pvalues.values())
sigs = {k: v for k, v in pvalues.items() if v < .05}
sigdat = data.loc[data['pID'].isin(sigs.keys()), ]

# Autoregressive data
# xt = αxt−1 + ξt
# α = 0.5, ξ = scipy.stats.norm()
α = 0.5
ξ = norm.rvs
# chaotic data
# xt = A − x2 t−1 + Byt−1 , yt = xt−1
# A = 1.4 and B = 0.3
a = 1.4
b = 0.3

discard = 1000
ld = 200
t = np.arange(0, ld, 1, int)
runs = discard + ld

def arf(x, α, ξ):
    newx = α*x + ξ()
    return newx
def chf(x, y, a, b):
    newx = a - x**2 + b*y
    newy = x
    x, y = newx, newy
    return (x, y)

ard = np.full(ld, np.nan)
x = np.random.random()
for r in range(runs):
    x = arf(x, α, ξ)
    check = r - discard
    if check >= 0:
        ard[check] = x
d = pd.DataFrame({'begin': t, 'dur': ard})
g = ia.directed_horizontal_visibility_graph(d)
kr = list(dict(g.in_degree()).values())
ka = list(dict(g.out_degree()).values())
cr = [dhvg_local_clustering(g, v, "backward") for v in g.nodes]
ca = [dhvg_local_clustering(g, v, "forward") for v in g.nodes]
print()
print("Degrees")
print(kstest(kr, ka))
print("Clustering")
print(kstest(cr, ca))

chd = np.full(ld, np.nan)
x = np.random.random()
y = np.random.random()
for r in range(runs):
    x, y = chf(x, y, a, b)
    check = r - discard
    if check >= 0:
        chd[check] = x
d = pd.DataFrame({'begin': t, 'dur': chd})
g = ia.directed_horizontal_visibility_graph(d)
kr = list(dict(g.in_degree()).values())
ka = list(dict(g.out_degree()).values())
cr = [dhvg_local_clustering(g, v, "backward") for v in g.nodes]
ca = [dhvg_local_clustering(g, v, "forward") for v in g.nodes]
print()
print("Degrees")
print(kstest(kr, ka))
print("Clustering")
print(kstest(cr, ca))



## Autoregressive, scaled to the empirical data
α = data['dur'].mean()
scale = np.exp(data['dur'].mean())
s = data['dur'].std()
loc = data['dur'].mean()
ξ = lognorm(s = s, loc = loc, scale = scale) #### This isn't really autoregressive...

d = pd.DataFrame({'begin': np.arange(0, 200, 1, int), 'dur': ξ.rvs(200)})
g = ia.directed_horizontal_visibility_graph(d)
kr = list(dict(g.in_degree()).values())
ka = list(dict(g.out_degree()).values())
cr = [dhvg_local_clustering(g, v, "backward") for v in g.nodes]
ca = [dhvg_local_clustering(g, v, "forward") for v in g.nodes]
print()
print("Degrees")
print(kstest(kr, ka))
print("Clustering")
print(kstest(cr, ca))

def lmap(x, r) :
    newx = r*x*(1 - x)
    return newx

a = 1.4
b = 0.3
chd = np.full(ld, np.nan)
#x = ξ.rvs()
#y = ξ.rvs()
x = np.random.random()
r = 3.7
t = np.arange(0, ld, 1, int)
for r in range(runs):
    x = lmap(x, r)
    #x, y = chf(x, y, a, b)
    check = r - discard
    if check >= 0:
        chd[check] = x
x *= data['dur'].mean()
d = pd.DataFrame({'begin': t, 'dur': chd})
g = ia.directed_horizontal_visibility_graph(d)
kr = list(dict(g.in_degree()).values())
ka = list(dict(g.out_degree()).values())
cr = [dhvg_local_clustering(g, v, "backward") for v in g.nodes]
ca = [dhvg_local_clustering(g, v, "forward") for v in g.nodes]
print()
print("Degrees")
print(kstest(kr, ka))
print("Clustering")
print(kstest(cr, ca))











# reset /t/ to an integer series
gID = 'UID'#np.random.choice(gIDs)
d_orig = data.loc[data['gID'] == gID, ].sort_values('begin').reset_index()
d_alt = data.loc[data['gID'] == gID, ].sort_values('begin').reset_index()
d_alt['begin'] = np.arange(start = 0, stop = len(d_alt), step = 1, dtype = int)

for d in [d_orig, d_alt]: # differences are probably due to separating speaking events that happened at the same time
    g = ia.directed_horizontal_visibility_graph(d)
    #kr = list(dict(g.in_degree()).values())
    #ka = list(dict(g.out_degree()).values())
    cr = [dhvg_local_clustering(g, v, "backward") for v in g.nodes]
    ca = [dhvg_local_clustering(g, v, "forward") for v in g.nodes]
    print(sorted(cr))
    print(sorted(ca))
    print(kstest(cr, ca))
    #print(kstest(kr, ka))

