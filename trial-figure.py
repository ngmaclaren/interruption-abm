## Take out all the calculations etc., just load in the tabulated data and make the figure.

#import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import InterruptionAnalysis as ia

emp = pd.read_csv("./data/emp-tst-iss.csv")
mimic = pd.read_csv("./data/mimic-groups-tst-iss.csv")
synthi = pd.read_csv("./data/synthetic-groups-independent-tst-iss.csv")
synthl = pd.read_csv("./data/synthetic-groups-listening-tst-iss.csv")
synthd = pd.read_csv("./data/synthetic-groups-dyadic-tst-iss.csv")

## try selecting a random test sim and comparing the distributions there
#testsim = np.random.choice(pd.unique(mimic["sim"]))

# plot

## KDE plot version
fig, ax = plt.subplots(figsize = [7, 6])
# https://github.com/ciannabp/inauguration/blob/main/R/colors.R
colors = ["#5445b1", "#749dae", "#f3c483"]#, "#5c1a33", "#cd3341","#f7dc6a"]
labels = ["Empirical", "Mimic", "Independent"]#, "Listening", "Dyadic"]
col = "count_iss" # "sum_tst"
xs = [emp[col], mimic[col], synthi[col]]#, synthl[col], synthd[col]]
#xs = [emp[col], mimic.loc[mimic["sim"] == testsim, col], synthi.loc[synthi["sim"] == testsim, col]]
for color, label, x in zip(colors, labels, xs):
    kde = stats.gaussian_kde(x)
    newx = np.linspace(min(x), max(x), num = 100)
    newy = kde(newx)
    lwd = 3 if label == "Empirical" else 2
    zorder = 2.5 if label == "Empirical" else 2
    ax.plot(newx, newy, color = color, linewidth = lwd, label = label, zorder = zorder)
    # if label == "Empirical":
    #     alpha = .75
    #     ax.fill_between(newx, newy, color = color, alpha = alpha)
ax.legend(loc = "upper right")
ax.set_xlabel("Total Speaking Time")
ax.set_ylabel("Kernel Density Estimate")

axin = ax.inset_axes([.5, .5, .25, .25])
for color, label, x in zip(colors, labels, xs):
    kde = stats.gaussian_kde(x)
    newx = np.linspace(min(x), max(x), num = 100)
    newy = kde(newx)
    zorder = 2.5 if label == "Empirical" else 2
    axin.plot(newx, newy, color = color, linewidth = lwd, label = label, zorder = zorder)
axin.set_xscale("log")

fig.show()

fig, ax = plt.subplots()

ax.boxplot(xs, positions = [1, 2, 3])

plt.show()


#ax1.tick_params(axis = "both", labelcolor = b1)
#ax2 = ax1.twiny().twinx()
#ax2.legend(loc = "center right")



# fig.show()


# whiteboard color theme
# black = ia.whiteboard["Black"]
# gray = ia.whiteboard["Gray50"]
# blue = ia.whiteboard["Blue4"]
# green = ia.whiteboard["Green4"]
# red = ia.whiteboard["Red"]
# orange = ia.whiteboard["DarkOrange3"]

# datas = pd.concat([data_tst, data_iss], axis = 1)
# mimics = pd.concat([mimic_tst, mimic_iss], axis = 1)
# synths = pd.concat([synth_tst, synth_iss], axis = 1)
# synths = synths.loc[synths["count_iss"] > 0, ]

# fig, ax = plt.subplots()
# fig.tight_layout()

# colors = [green, orange, blue]
# alphas = [.5, .5, 1]
# labels = ["Mimic", "Synthetic", "Empirical"]
# dfs = [mimics, synths, datas]
# for color, alpha, label, df in zip(colors, alphas, labels, dfs):
#     ax.scatter(df["sum_tst"], df["count_iss"], marker = ".", color = color, alpha = alpha, label = label)
# # ax.scatter(mimic_tst, mimic_iss, marker = ".", color = green, alpha = .5, label = "Mimic")
# # ax.scatter(synth_tst, synth_iss, marker = ".", color = orange, alpha = .5, label = "Synthetic")
# # ax.scatter(data_tst, data_iss, marker = ".", color = blue, label = "Empirical")

# for color, df in zip(colors, dfs):
#     x = np.log(df["sum_tst"])
#     y = np.log(df["count_iss"])
#     coefs = np.polyfit(x, y, 1)
#     fit = np.poly1d(coefs)
#     newy = fit(x)
#     ax.plot(x, newy, color = color)
    
# ax.set_xscale("log")
# ax.set_yscale("log")

# ax.legend()

# fig.show()
# fig.savefig("./img/ccs2021-figure.pdf") 





## colors could be shaded instead of from different color families. Like the blues and greens from whiteboard.
# b1 = tuple(c/255 for c in (0, 0, 139))
# b2 = tuple(c/255 for c in (28, 134, 238))
# b3 = tuple(c/255 for c in (0, 191, 255))
# g1 = tuple(c/255 for c in (0, 139, 0))
# g2 = tuple(c/255 for c in (134, 238, 28))
# g3 = tuple(c/255 for c in (191, 255, 0))

# greys = ["#000000", "#666666", "#bfbfbf"]
# colors = ["#008b00", "#cd6600", "#cdad00"]


## try fig, ax1 = plt.subplots(); ax2 = ax.twiny --> to plot both TST and ISS on the same Axes object.

## violinplot version
# fig, ax = plt.subplots(figsize = [6, 4.5])
# colors = ["#5445b1", "#749dae", "#f3c483", "#5c1a33", "#cd3341","#f7dc6a"]
# labels = ["Empirical", "Mimic", "Independent", "Listening", "Dyadic"]
# #col = "sum_tst"
# col = "count_iss"
# xs = [emp[col], mimic[col], synthi[col], synthl[col], synthd[col]]

# parts = ax.violinplot(xs, showextrema = False)
# for pc in parts["bodies"]:
#     pc.set_facecolor(colors[3])
#     pc.set_alpha(1)
# #for c in ["cmaxes", "cmins", "cbars"]:
# #    parts[c].set_edgecolor(colors[4])
# ax.set_xticklabels([""] + labels)
# #ax.set_yscale("log")
# fig.show()
