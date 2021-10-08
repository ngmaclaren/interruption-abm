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
    
# keep only those agents that I analyzed in the DHVg analysis: those with |x| >= 20
sample = list(data.groupby("pID")["dur"].count().loc[lambda x: x >= 20].index)

rootpath = "./data/simulations/mimic-agents"

results = {}
for pID in sample:
    # collect reference data
    refdat = data.loc[data["pID"] == pID, ] # so, refdat["dur"] and refdat["lat"]
    refB = ia.bursty_coef(refdat["lat"])
    refM = ia.memory_coef(refdat.sort_values("begin")["lat"])

    # storage for summary stats from each sim run
    durs = []
    lats = []
    Bs = []
    Ms = []

    # search through directory structure
    # a loaded sim should be called `X`
    pidpath = rootpath + "/" + pID
    for root, dirs, files in os.walk(pidpath):
        if not files:
            continue
        for f in files:
            X = pd.read_csv(os.path.join(root, f))

            dur = list(X["dur"])
            lat = list(X["lat"])
            B = ia.bursty_coef(X["lat"])
            M = ia.memory_coef(X["lat"]) # already sorted on "begin"
            durs.extend(dur)
            lats.extend(lat)
            Bs.append(B)
            Ms.append(M)
        lat_ksp = stats.kstest(refdat["lat"], lats)[1]
        dur_ksp = stats.kstest(refdat["dur"], durs)[1]
        B_ptp = len([b for b in Bs if abs(b - np.mean(Bs)) >= abs(refB - np.mean(Bs))])/len(Bs)
        M_ptp = len([m for m in Ms if abs(m - np.mean(Ms)) >= abs(refM - np.mean(Ms))])/len(Ms)
        results[pID] = {"lat_ksp": lat_ksp, "dur_ksp": dur_ksp, "B_ptp": B_ptp, "M_ptp": M_ptp}
rdf = pd.DataFrame.from_dict(results, orient = "index")

fig, axs = plt.subplots(2, 2)
current_size = fig.get_size_inches()
new_size = [x*2 for x in current_size]
fig.set_size_inches(new_size)
fig.tight_layout()

black = ia.whiteboard["Black"]
gray = ia.whiteboard["Gray50"]
blue = ia.whiteboard["Blue4"]
skyblue = ia.whiteboard["SkyBlue1"]
green = ia.whiteboard["Green4"]
red = ia.whiteboard["Red"]

titles = ["Latency", "Duration", "Burstiness", "Memory"]
cols = list(rdf)
for ax, title, col in zip(axs.flatten(), titles, cols):
    #ax.set_title(title)
    ax.hist(rdf[col], bins = 20, color = gray, label = r"$p$-value")
    ax.axvline(0.05, color = red, linewidth = 2, label = r"$p = 0.05$")
    #ax.axvline(rdf[col].mean(), color = blue, linewidth = 2, label = "Mean")
    ax.axvline(rdf[col].median(), color = blue, linewidth = 2, label = "Median")
    ax.legend()
    
for ax, letter in zip(axs.flatten(), ["A", "B", "C", "D"]):
    ax.text(0.01, 0.99, letter, fontsize = 12,# fontweight = "bold",
            horizontalalignment = "left", verticalalignment = "top", transform = ax.transAxes)
    
fig.savefig("./img/individual-sims-results.pdf")
fig.show()
