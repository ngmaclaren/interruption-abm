import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats as stats
import matplotlib.pyplot as plt
import InterruptionAnalysis as ia

import Independent
import Listening
import Dyadic

black = ia.whiteboard["Black"]
gray = ia.whiteboard["Gray50"]
blue = ia.whiteboard["Blue4"]
skyblue = ia.whiteboard["SkyBlue1"]
green = ia.whiteboard["Green4"]
red = ia.whiteboard["Red"]

data = pd.read_csv("./data/timeseries.csv", index_col = 0)
numeric_cols = ["begin", "end", "dur", "lat"]
for col in numeric_cols:
    data[col] = data[col]/100 # converts to 1/10th seconds
votedata = pd.read_csv("./data/vote-data.csv")
N = len(votedata)
group_sizes = votedata.groupby("gID")["pID"].count()
effective_group_sizes = data.groupby("gID")["pID"].nunique()

gIDs = list(effective_group_sizes.index)
pIDs = pd.unique(data["pID"])
rows = {}
for pID in pIDs:
    # estimate p and q here
    P_i = ia.get_transition_matrix(data, pID)
    p = P_i[0, 1]
    q = P_i[1, 0]
    rows[pID] = [p, q]
    
P = pd.DataFrame.from_dict(rows, orient = "index", columns = ["p", "q"])
P["gID"] = [x[:3] for x in list(P.index)]

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

refdat = data.groupby("gID").agg(sum_tst = ("dur", "sum"), count_tst = ("dur", "count"))

iss_dfs = []
nss_dfs = []
for gID in gIDs:
    dat = data.loc[data["gID"] == gID, ]
    pIDs = pd.unique(data["pID"])

    iss = ia.interruptive_simultaneous_speech(dat, pIDs)
    sorter = iss.groupby(["i", "begin"])["dur"].count()
    sorter = list(sorter[sorter > 1].index)
    for row in sorter:
        temp = iss.loc[(iss["i"] == row[0]) & (iss["begin"] == row[1]), ]
        drop = temp.loc[temp["dur"] != max(temp["dur"]), ]
        iss.drop(list(drop.index), inplace = True)

    nss = ia.non_interruptive_simultaneous_speech(dat, pIDs)
    sorter = nss.groupby(["j", "begin"])["dur"].count()
    sorter = list(sorter[sorter > 1].index)
    for row in sorter:
        temp = nss.loc[(nss["j"] == row[0]) & (nss["begin"] == row[1]), ]
        drop = temp.loc[temp["dur"] != max(temp["dur"]), ]
        nss.drop(list(drop.index), inplace = True)

    isss = iss.groupby("i").agg(iss_sum = ("dur", "sum"), iss_count = ("dur", "count"))
    iss_dfs.append(isss)
    nsss = nss.groupby("j").agg(nss_sum = ("dur", "sum"), nss_count = ("dur", "count"))
    nss_dfs.append(nsss)

iss = pd.concat(iss_dfs)
iss.index.rename("pID", inplace = True)
iss.reset_index(inplace = True)
nss = pd.concat(nss_dfs)
nss.index.rename("pID", inplace = True)
nss.reset_index(inplace = True)
iss["gID"] = iss["pID"].apply(lambda x: x[:3])
nss["gID"] = nss["pID"].apply(lambda x: x[:3])
iss = iss.groupby("gID").agg(iss_sum = ("iss_sum", "sum"), iss_count = ("iss_count", "sum"))
nss = nss.groupby("gID").agg(nss_sum = ("nss_sum", "sum"), nss_count = ("nss_count", "sum"))

refdat = pd.concat([refdat, iss, nss], axis = 1)
sumcols = ["sum_tst", "iss_sum", "nss_sum"]
refdat[sumcols] = refdat[sumcols]/10

# now I need to make all the interruption networks. I have of course completely forgotten how to do that.
refnets = {}
refprs = {}
for gID in gIDs:
    pIDs = pd.unique(data[data['gID'] == gID]['pID']) ###!!!!!! Ignoring silent group members!
    igb = ia.interruption_network_pandas(data, pIDs, use = 'both')
    refnets[gID] = igb
    pr = nx.pagerank_numpy(igb, weight = "weight", alpha = 0.99)
    refprs = {**refprs, **pr}
refprs_df = pd.DataFrame.from_dict(refprs, orient = "index", columns = ["pr"])
refprs_df.rename_axis(index = "pID", inplace = True)
refprs_df.reset_index(inplace = True)
refprs_df["gID"] = refprs_df["pID"].apply(lambda x: x[:3])
rls = [ia.find_leader(refnets[gID], 0.99, "weight") for gID in gIDs]

refnetstats = {}
for gID in gIDs:
    g = refnets[gID]
    dens = nx.density(g)
    cent = ia.pagerank_centralization(g, alpha = 0.99, weight = "weight")
    avg_clust = nx.average_clustering(g, weight = "weight")
    refnetstats[gID] = {"dens": dens, "cent": cent, "avg_clust": avg_clust}
rnsdf = pd.DataFrame.from_dict(refnetstats, orient = "index")

refdat = pd.concat([refdat, rnsdf], axis = 1)

nsims = 100
kind =  "synthetic" # "mimic" "synthetic"
assumption = "dyadic" # "independent" "listening" "dyadic"
print(kind)
print(assumption)
all_simdats = []
all_pvals = {}
if kind == "mimic":
    all_prs = []
    all_sls = []
    
for sim in range(nsims):
    print(sim)
    results = []
    for gID in gIDs:
        pIDs = pd.unique(data[data["gID"] == gID]["pID"])
        T = round(data[data["gID"] == gID]["end"].max())
        N = len(pIDs)
        ns = list(range(N))

        if kind == "mimic":
            P_i = [ia.get_transition_matrix(data, pID) for pID in pIDs]
        else:
            p_i = fits["AB"].rvs(size = N)
            q_i = fits["BA"].rvs(size = N)
            P_i = [
                np.array([[1 - p, p],
                          [q, 1 - q]]) for p, q in zip(p_i, q_i)
            ]

        if assumption == "independent":
            Y = Independent.simulation(P_i, T, N, ns, oneagent = False)
        elif assumption == "listening":
            Y = Listening.simulation(P_i, T, N, ns, scale = 1e-5)
        elif assumption == "dyadic":
            Y = Dyadic.simulation(P_i, T, N, ns, scale = 1e-5)

        X = ia.Y_to_X(Y, ns)        
        X["gID"] = gID
        #X["sim"] = sim
        if kind == "mimic":
            X["pID"] = X["pID"].apply(lambda x: pIDs[x])
        results.append(X)
    rdf = pd.concat(results)
    simdat = rdf.groupby("gID").agg(sum_tst = ("dur", "sum"), count_tst = ("dur", "count"))

    iss_dfs = []
    nss_dfs = []
    for gID in gIDs:
        dat = rdf.loc[rdf["gID"] == gID, ]
        pIDs = pd.unique(rdf["pID"])

        iss = ia.interruptive_simultaneous_speech(dat, pIDs)
        sorter = iss.groupby(["i", "begin"])["dur"].count()
        sorter = list(sorter[sorter > 1].index)
        for row in sorter:
            temp = iss.loc[(iss["i"] == row[0]) & (iss["begin"] == row[1]), ]
            drop = temp.loc[temp["dur"] != max(temp["dur"]), ]
            iss.drop(list(drop.index), inplace = True)

        nss = ia.non_interruptive_simultaneous_speech(dat, pIDs)
        sorter = nss.groupby(["j", "begin"])["dur"].count()
        sorter = list(sorter[sorter > 1].index)
        for row in sorter:
            temp = nss.loc[(nss["j"] == row[0]) & (nss["begin"] == row[1]), ]
            drop = temp.loc[temp["dur"] != max(temp["dur"]), ]
            nss.drop(list(drop.index), inplace = True)

        isss = iss.groupby("i").agg(iss_sum = ("dur", "sum"), iss_count = ("dur", "count"))
        isss["gID"] = gID
        iss_dfs.append(isss)
        nsss = nss.groupby("j").agg(nss_sum = ("dur", "sum"), nss_count = ("dur", "count"))
        nsss["gID"] = gID
        nss_dfs.append(nsss)

    iss = pd.concat(iss_dfs)
    iss.index.rename("pID", inplace = True)
    iss.reset_index(inplace = True)
    nss = pd.concat(nss_dfs)
    nss.index.rename("pID", inplace = True)
    nss.reset_index(inplace = True)
    #iss["gID"] = iss["pID"].apply(lambda x: x[:3])
    #nss["gID"] = nss["pID"].apply(lambda x: x[:3])
    iss = iss.groupby("gID").agg(iss_sum = ("iss_sum", "sum"), iss_count = ("iss_count", "sum"))
    nss = nss.groupby("gID").agg(nss_sum = ("nss_sum", "sum"), nss_count = ("nss_count", "sum"))

    simdat = pd.concat([simdat, iss, nss], axis = 1)
    sumcols = ["sum_tst", "iss_sum", "nss_sum"]
    simdat[sumcols] = simdat[sumcols]/10

    simnets = {}
    for gID in gIDs:
        dat = rdf[rdf["gID"] == gID]
        pIDs = pd.unique(dat['pID']) ###!!!!!! Ignoring silent group members!
        igb = ia.interruption_network_pandas(dat, pIDs, use = 'both')
        refpids = pd.unique(data[data["gID"] == gID]["pID"])
        missing = [x for x in refpids if x not in list(igb.nodes)]
        igb.add_nodes_from(missing)
        simnets[gID] = igb

    if kind == "mimic":
        simprs = {}
        for gID in gIDs:
            pr = nx.pagerank_numpy(simnets[gID], weight = "weight", alpha = 0.99)
            simprs = {**simprs, **pr}
        simprs_df = pd.DataFrame.from_dict(simprs, orient = "index", columns = ["pr"])
        simprs_df.rename_axis(index = "pID", inplace = True)
        simprs_df["sim"] = sim
        simprs_df.reset_index(inplace = True)
        simprs_df["gID"] = simprs_df["pID"].apply(lambda x: x[:3])
        all_prs.append(simprs_df)
        sls = {gID: ia.find_leader(simnets[gID], 0.99, "weights") for gID in gIDs}
        sls_df = pd.DataFrame.from_dict(sls, orient = "index", columns = ["leader"])
        sls_df.rename_axis(index = "pID", inplace = True)
        sls_df["sim"] = sim
        sls_df.reset_index(inplace = True)
        sls_df["gID"] = sls_df["pID"].apply(lambda x: x[:3])
        all_sls.append(sls_df)

    simnetstats = {}
    for gID in gIDs:
        g = simnets[gID]
        dens = nx.density(g)
        cent = ia.pagerank_centralization(g, alpha = 0.99, weight = "weight")
        avg_clust = nx.average_clustering(g, weight = "weight")
        simnetstats[gID] = {"dens": dens, "cent": cent, "avg_clust": avg_clust}
    snsdf = pd.DataFrame.from_dict(simnetstats, orient = "index")

    simdat = pd.concat([simdat, snsdf], axis = 1)
    simdat["sim"] = sim
    all_simdats.append(simdat)

    pvals = {}
    for col in list(refdat):
        pval = stats.wilcoxon(refdat[col], simdat[col])[1]
        pvals[col] = pval
    all_pvals[sim] = pvals
    
all_pvals_df = pd.DataFrame.from_dict(all_pvals, orient = "index")
all_simdats_df = pd.concat(all_simdats)

if kind == "mimic":
    all_prs_df = pd.concat(all_prs)
    all_sls_df = pd.concat(all_sls)
    
    p_match = []
    for sim in pd.unique(all_sls_df["sim"]):
        dat = all_sls_df.loc[all_sls_df["sim"] == sim, ]
        prop = len([x for x in list(dat["leader"]) if x in rls])/len(dat)
        p_match.append(prop)

    fig, ax = plt.subplots()
    ax.plot(list(range(nsims)), p_match)
    ax.set_xlabel("Simulation Run")
    ax.set_ylabel("Proportion of Groups for which Leaders Match")
    #ax.axhline(1/effective_group_sizes.mean(), color = green)
    fig.savefig("./img/mimic-groups-leadermatch.svg")
    fig.show()
    
fig, axs = plt.subplots(3, 3)
current_size = fig.get_size_inches()
new_size = [x*2 for x in current_size]
fig.set_size_inches(new_size)

figdat = all_pvals_df.copy()

for ax, col in zip(axs.flatten(), list(figdat)):
    ax.hist(figdat[col], bins = 50, color = gray, label = r"$p$-value")
    #ax.set_title(col)
    # check to see if 0.05 is within the current axes limits
    # if it is, plot the axvline
    ax.axvline(figdat[col].median(), color = green, label = "Median")
    if 0.05 < figdat[col].max():
        ax.axvline(0.05, color = blue, label = r"$p$ = 0.05")
    # if col in ["iss_sum", "nss_sum", "nss_count"]:
    #     ax.set_xscale("log")
    ax.legend()

LETTERS = list(map(chr, range(ord("A"), ord("Z"))))
for ax, LETTER in zip(axs.flatten(), LETTERS):
    ax.text(0.01, 0.99, LETTER, horizontalalignment = "left", verticalalignment = "top", transform = ax.transAxes)
    
fig.tight_layout()
if kind == "mimic":
    fig.savefig("./img/mimic-groups-independent.svg")
elif kind == "synthetic":
    fig.savefig(f"./img/synthetic-groups-{assumption}.svg")
fig.show()

# get some fresh data, make sure everything works above, including storing the sim results
# once have the sim results df, make boxplots. Boxplot should compare the reference data on 1 to the combined simdata on 2 for each of the columns, just like for the wilcoxon results above.
fig, axs = plt.subplots(3, 3)
current_size = fig.get_size_inches()
new_size = [x*2 for x in current_size]
fig.set_size_inches(new_size)

figdat = all_simdats_df.copy()
figdat.rename_axis(index = "gID", inplace = True)
figdat.reset_index(inplace = True)

for ax, col in zip(axs.flatten(), list(refdat)):
    ax.boxplot(refdat[col], positions = [1], labels = ["Empirical"], medianprops = dict(color = green))
    ax.boxplot(figdat[~figdat[col].isna()][col], positions = [2], labels = ["Simulated"], medianprops = dict(color = green))

for ax, LETTER in zip(axs.flatten(), LETTERS):
    ax.text(0.01, 0.99, LETTER, horizontalalignment = "left", verticalalignment = "top", transform = ax.transAxes)

fig.tight_layout()
if kind == "mimic":
    fig.savefig("./img/mimic-groups-independent-boxplots.svg")
elif kind == "synthetic":
    fig.savefig(f"./img/synthetic-groups-{assumption}-boxplots.svg")

fig.show()
print("Done.")
