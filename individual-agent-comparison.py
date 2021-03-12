############################
###       WARNING        ###
### Long Running Program ###
############################
  
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import InterruptionAnalysis as ia
import Independent as sim

data = pd.read_csv('./data/timeseries.csv', index_col = 0)
numeric_cols = ['begin', 'end', 'dur', 'lat']
for col in numeric_cols:
    data[col] = data[col]/100 # converts to 1/10th seconds

idata = data.groupby("pID")["lat"].agg([ia.bursty_coef, ia.memory_coef, "count"])
    
# keep only those agents that I analyzed in the DHVg analysis: those with |x| > 20
sample = list(data.groupby("pID")["dur"].count().loc[lambda x: x > 20].index)

# estimate /p/ and /q/ for each r_i and collect it into a data frame indexed by pID with columns "p" and "q"
rows = {}
for pID in sample:
    # estimate p and q here
    P_i = ia.get_transition_matrix(data, pID)
    p = P_i[0, 1]
    q = P_i[1, 0]
    rows[pID] = [p, q]
    
P = pd.DataFrame.from_dict(rows, orient = "index", columns = ["p", "q"])

pID = np.random.choice(sample)
results = {}
for pID in sample:
    gID = pID[:3]
    print(pID)
    row = P[P.index.isin([pID])]
    P_i = np.array([[1 - row["p"], row["p"]], [row["q"], 1 - row["q"]]])

    refdat = data.loc[data["pID"] == pID, ] # so, refdat["dur"] and refdat["lat"]
    refB = ia.bursty_coef(refdat["lat"])
    refM = ia.memory_coef(refdat.sort_values("begin")["lat"])

    nsims = 2500
    T = round(data[data["gID"] == gID]["end"].max())
    N = 1
    ns = list(range(N))
    durs = []
    lats = []
    Bs = []
    Ms = []
    for _ in range(nsims):
        Y = sim.simulation(P_i, T, N, ns, oneagent = True)
        X = ia.Y_to_X(Y, ns)
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
print(rdf)

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

titles = ["Latency", "Duration", "Burstiness", "Memory"]
cols = list(rdf)
for ax, title, col in zip(axs.flatten(), titles, cols):
    ax.set_title(title)
    ax.hist(rdf[col], bins = 20, color = gray, label = r"$p$-value")
    ax.axvline(0.05, color = blue, linewidth = 2, label = r"$p = 0.05$")
    #ax.axvline(rdf[col].mean(), color = blue, linewidth = 2, label = "Mean")
    ax.axvline(rdf[col].median(), color = green, linewidth = 2, label = "Median")
    ax.legend()
    
for ax, letter in zip(axs.flatten(), ["A", "B", "C", "D"]):
    ax.text(0.01, 0.99, letter, horizontalalignment = "left", verticalalignment = "top", transform = ax.transAxes)
    
fig.savefig("./img/individual-sims-results.svg")
fig.show()




###
### Old code testing out features

# pID = np.random.choice(sample)#pd.unique(data["pID"]))
# gID = pID[:3]#data[data["pID"] == pID].loc[0, "gID"]
# print(pID)
# row = P[P.index.isin([pID])]
# print(row)
# P_i = np.array([
#     [1 - row["p"], row["p"]],
#     [row["q"], 1 - row["q"]]
#     ])
# #g_i = sim.Agent(P_i)
# #minutes = 10
# #T = minutes*60*10
# T = round(data[data["gID"] == gID]["end"].max())
# N = 1
# ns = list(range(N))

# nsims = 500
# cols = ["dur", "lat", "B", "M"]
# durs = []
# lats = []
# Bs = []
# Ms = []
# for _ in range(nsims):
#     Y = sim.simulation(P_i, T, N, ns, oneagent = True)
#     X = ia.Y_to_X(Y, ns)
#     dur = list(X["dur"])
#     lat = list(X["lat"])
#     B = ia.bursty_coef(X["lat"])
#     M = ia.memory_coef(X["lat"]) # already sorted on "begin"
#     durs.extend(dur)
#     lats.extend(lat)
#     Bs.append(B)
#     Ms.append(M)
# fig, axs = plt.subplots(2, 2)
# current_size = fig.get_size_inches()
# new_size = [x*2 for x in current_size]
# fig.set_size_inches(new_size)

# black = ia.whiteboard["Black"]
# gray = ia.whiteboard["Gray75"]
# blue = ia.whiteboard["Blue4"]
# skyblue = ia.whiteboard["SkyBlue1"]
# green = ia.whiteboard["Green4"]


# refdat = data.loc[data["pID"] == pID, ]
# fig.suptitle(pID)

# ax1 = axs[0, 0] # dur
# ax2 = axs[0, 1] # lat
# ax3 = axs[1, 0] # B
# ax4 = axs[1, 1] # M

# ax1.set_title("Latency")
# ax1.hist(lats, bins = 50, color = blue)
# ax1.axvline(np.mean(lats), color = skyblue, linewidth = 8)
# ax1.axvline(refdat["lat"].mean(), color = green, linewidth = 2)
# ax2.set_title("Duration")
# ax2.hist(durs, bins = 50, color = blue)
# ax2.axvline(np.mean(durs), color = skyblue, linewidth = 8)
# ax2.axvline(refdat["dur"].mean(), color = green, linewidth = 2)
# ax3.set_title("Burstiness")
# ax3.hist(Bs, bins = 25, color = blue)
# ax3.axvline(np.mean(Bs), color = skyblue, linewidth = 8)
# ax3.axvline(ia.bursty_coef(refdat.sort_values("begin")["lat"]), color = green, linewidth = 2)
# ax4.set_title("Memory")
# ax4.hist(Ms, bins = 25, color = blue)
# ax4.axvline(np.mean(Ms), color = skyblue, linewidth = 8)
# ax4.axvline(ia.memory_coef(refdat.sort_values("begin")["lat"]), color = green, linewidth = 2)

# for ax, letter in zip(axs.flatten(), ["A", "B", "C", "D"]):
#     ax.text(0.01, 0.99, letter, horizontalalignment = "left", verticalalignment = "top", transform = ax.transAxes)
    
# fig.show()


###
### For confirming the sim works

# results = []
# for _ in range(250):
#     Y = sim.simulation(P_i, T, N, ns, oneagent = True)
#     results.append(Y.sum()/10)
# print({"mean": np.mean(results), "sd": np.std(results), "ref": data.loc[data["pID"] == pID, ]["dur"].sum()/10})

###
### For visualizing the simulations with respect to the reference data 

# fig, axs = plt.subplots(6, 1, figsize = (15, 10))
# #fig.tight_layout()

# ia.visualize_speaking_data(data, pID = pID, ax = axs[0], colors = ia.solarized["base01"])

# for run in range(1, 6):
#     Y = sim.simulation(P_i, T, N, ns, oneagent = True)
#     X = ia.Y_to_X(Y, ns)
#     ia.visualize_speaking_data(X, pID = ns[0], ax = axs[run], colors = ia.solarized["base1"])

# fig.show()



###
### just plot the histograms
# fig, axs = plt.subplots(2, 2)
# current_size = fig.get_size_inches()
# new_size = [x*2 for x in current_size]
# fig.set_size_inches(new_size)

# dat = data.loc[data["pID"].isin(sample)]
# idat = dat.groupby("pID")["lat"].agg([ia.bursty_coef, ia.memory_coef, "count"])

# ax1 = axs[0, 0] # dur
# ax2 = axs[0, 1] # lat
# ax3 = axs[1, 0] # B
# ax4 = axs[1, 1] # M

# ax1.set_title("Latency")
# ax1.hist(dat["lat"], bins = 50, color = ia.solarized["green"])
# ax2.set_title("Duration")
# ax2.hist(dat["dur"], bins = 50, color = ia.solarized["cyan"])
# ax3.set_title("Burstiness")
# ax3.hist(idat["bursty_coef"], bins = 25, color = ia.solarized["blue"])
# ax4.set_title("Memory")
# ax4.hist(idat["memory_coef"], bins = 25, color = ia.solarized["violet"])


# for ax, letter in zip(axs.flatten(), ["A", "B", "C", "D"]):
#     ax.text(0.01, 0.99, letter, horizontalalignment = "left", verticalalignment = "top", transform = ax.transAxes)
    
# fig.show()
