### I don't think this is answering the question I want to ask, somehow. Maybe it will come out in the wash when I calculate the total cumulative TST in the synthetic groups
### regardless, point of this analysis was to decide whether or not to keep .rvs(), and I think the answer is yes as long as I do what I did here, which was to stratify within group sizes

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import InterruptionAnalysis as ia
import Independent as sim

data = pd.read_csv("./data/timeseries.csv", index_col = 0)
numeric_cols = ["begin", "end", "dur", "lat"]
for col in numeric_cols:
    data[col] = data[col]/100 # converts to 1/10th seconds
votedata = pd.read_csv("./data/vote-data.csv")
N = len(votedata)
group_sizes = votedata.groupby("gID")["pID"].count()
effective_group_sizes = data.groupby("gID")["pID"].nunique()
#plt.plot(group_sizes, effective_group_sizes, "ko"); plt.show()

#idata = data.groupby("pID")["lat"].agg([ia.bursty_coef, ia.memory_coef, "count"])
    
#sample = list(data.groupby("pID")["dur"].count().loc[lambda x: x > 20].index)
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

nsims = 2500
max_ab = []; min_ab = []
max_ba = []; min_ba = []
for _ in range(nsims):
    n = len(P)
    ab = fits["AB"].rvs(n)
    ba = fits["BA"].rvs(n)
    max_ab.append(max(ab))
    max_ba.append(max(ba))
    min_ab.append(min(ab))
    min_ba.append(min(ba))

rdf = pd.DataFrame({"min_p": min_ab, "max_p": max_ab, "min_q": min_ba, "max_q": max_ba})

fig, axs = plt.subplots(2, 2)
current_size = fig.get_size_inches()
new_size = [x*2 for x in current_size]
fig.set_size_inches(new_size)

black = ia.whiteboard["Black"]
gray = ia.whiteboard["Gray50"]
blue = ia.whiteboard["Blue4"]
skyblue = ia.whiteboard["SkyBlue1"]
green = ia.whiteboard["Green4"]
red = ia.whiteboard["Red"]

titles = [r"Min $p$", r"Max $p$", r"Min $q$", r"Max $q$"]
cols = list(rdf)
refs = [P["p"].min(), P["p"].max(), P["q"].min(), P["q"].max()]

for ax, col, ref, title in zip(axs.flatten(), cols, refs, titles):
    ax.set_title(title)
    ax.hist(rdf[col], bins = 50, color = gray)
    ax.axvline(ref, color = blue)

for ax, letter in zip(axs.flatten(), ["A", "B", "C", "D"]):
    ax.text(0.01, 0.99, letter, horizontalalignment = "left", verticalalignment = "top", transform = ax.transAxes)
    
#len([b for b in Bs if abs(b - np.mean(Bs)) >= abs(refB - np.mean(Bs))])/len(Bs)
pvals = {}
for col, ref in zip(cols, refs):
    pvals[col] = len([x for x in rdf[col] if abs(x - rdf[col].mean()) >= abs(ref - rdf[col].mean())])/len(rdf[col])

for ax, col in zip(axs.flatten(), cols):
    ax.text(0.5, 0.5, r"$p$ = " + f"{pvals[col]}", horizontalalignment = "left", transform = ax.transAxes)

fig.savefig("./img/pq-overall-distribution.svg")
fig.show()

###
### As a curiousity, let's do p vs q
fig, ax = plt.subplots()
current_size = fig.get_size_inches()
new_size = [x*1.2 for x in current_size]
fig.set_size_inches(new_size)

ax.scatter(x = P["p"], y = P["q"], marker = ".", color = gray)
ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$q$")
ax.set_xscale("log")
ax.set_yscale("log")
fig.show()

# Now, I need to do a within groups version of the same analysis.

# choose a group size by randomly selecting an index in group_sizes
# choose an effective group size by choosing the same index in effective_group_sizes
# independently draw from P["p"] and P["q"] that many times
nsims = 2500
gIDs = list(effective_group_sizes.index)
results = []
refdat = P.groupby("gID")
for gID in gIDs:
    N = effective_group_sizes[gID]
    for sim in range(nsims):
        ## the below is the code for bootstrap sampling---not what I want to try first
        ## P_g = pd.DataFrame({"p": np.random.choice(P["p"]), "q": np.random.choice(P["q"])})
        P_g = pd.DataFrame({"p": fits["AB"].rvs(size = N), "q": fits["BA"].rvs(size = N)})

        wn_grp_min_p = P_g["p"].min()
        wn_grp_max_p = P_g["p"].max()
        wn_grp_min_q = P_g["q"].min()
        wn_grp_max_q = P_g["q"].max()

        ref_min_p = refdat["p"].min()[gID]
        ref_max_p = refdat["p"].max()[gID]
        ref_min_q = refdat["q"].min()[gID]
        ref_max_q = refdat["q"].max()[gID]
        
        ts_min_p = ref_min_p - wn_grp_min_p
        ts_max_p = ref_max_p - wn_grp_max_p
        ts_min_q = ref_min_q - wn_grp_min_q
        ts_max_q = ref_max_q - wn_grp_max_q

        row = {"N": N, "ts_min_p": ts_min_q, "ts_max_p": ts_max_p, "ts_min_q": ts_min_q, "ts_max_q": ts_max_q}
        results.append(row)
rdf = pd.DataFrame(results)

Ns = sorted(list(pd.unique(rdf["N"])))
fig, axs = plt.subplots(len(Ns), 4, figsize = [10, 15])

titles = [r"Min $p$", r"Max $p$", r"Min $q$", r"Max $q$"]
cols = ["ts_min_p", "ts_max_p", "ts_min_q", "ts_max_q"]
colors = [ia.whiteboard["Green4"], ia.whiteboard["DarkOliveGreen4"], ia.whiteboard["Burlywood4"],
          ia.whiteboard["DarkOrange3"], ia.whiteboard["Gold3"], ia.whiteboard["Blue4"],
          ia.whiteboard["SkyBlue1"], ia.whiteboard["Red"]]

for i in range(len(Ns)):
    for j, title in zip(range(len(cols)), titles):
        x = sorted(list(rdf.loc[rdf["N"] == Ns[i], cols[j]]))
        maj = round(.975*len(x))
        ll = x[-maj]
        ul = x[maj]
        axs[i, j].hist(x, bins = 50, color = gray)
        axs[i, j].axvline(ll, color = green)
        axs[i, j].axvline(ul, color = green)

        if j == 0:
            axs[i, j].set_ylabel(r"$N$" + f" = {Ns[i]}")

        if i == 0:
            axs[i, j].set_title(title)

fig.tight_layout(rect=[0.01, 0.03, 1, 0.95]) #left bottom right top
fig.savefig("./img/pq-wngroup-distribution.svg")
fig.show()



# try a scatter  ## I think this is the same information, but less of it and harder to make sense of
# fig, axs = plt.subplots(1, 2)
# figdat = rdf[rdf["N"] == 6]
# axs[0].scatter(figdat["ts_min_p"], figdat["ts_max_p"], color = gray, marker = ".")
# axs[0].set_title(r"$p$")
# axs[1].scatter(figdat["ts_min_q"], figdat["ts_max_q"], color = gray, marker = ".")
# axs[1].set_title(r"$q$")
# for ax in axs.flatten():
#     ax.set_xlabel("Min")
#     ax.set_ylabel("Max")
# fig.show()
