import numpy as np
import pandas as pd

import InterruptionAnalysis as ia

data = pd.read_csv("./data/timeseries.csv", index_col = 0)
votedata = pd.read_csv("./data/vote-data.csv")
numericcols = ["begin", "end", "dur", "lat"]
data[numericcols] = data[numericcols]/1000

summary_funcs = [np.mean, np.std, np.median, np.min, np.max]

## Event Level
print("Event Level")
print(data[["dur", "lat"]].agg(summary_funcs).T)
print(data[["dur", "lat"]].corr("pearson"))

print("ICC")
gbs = ["pID", "gID"]
varcols = ["dur", "lat"]
for gb in gbs:
    print(gb)
    for var in varcols:
        X = data[[var, gb]]
        model = f"{var} ~ {gb}"
        k = X.groupby(gb).count().mean()
        print(var)
        print(ia.icc1(X, model, k))

## Individual Level
print("Individual Level")

idata = data.groupby("pID")["lat"].agg([ia.bursty_coef, ia.memory_coef, "count"])

idata = data.groupby("pID").agg(
    B = pd.NamedAgg(column = "lat", aggfunc = ia.bursty_coef),
    M = pd.NamedAgg(column = "lat", aggfunc = ia.memory_coef),
    count = pd.NamedAgg(column = "lat", aggfunc = "count"),
    tst = pd.NamedAgg(column = "dur", aggfunc = "sum")
)

#### Turn this into a function so I can reuse it on the synthetic groups
gIDs = pd.unique(votedata["gID"])

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
nss = pd.concat(nss_dfs)
nss.index.rename("pID", inplace = True)

idata = pd.concat([idata, iss, nss], axis = 1)

# for adding back in the zero valued participants; not completed
# missing_pID = votedata.loc[~(votedata["pID"].isin(data["pID"])), "pID"]
# rowdict = {"B": np.nan, "M": np.nan, "count": 0, "tst": 0, "iss_sum": 0, "iss_count": 0, "nss_sum": 0, "nss_count": 0}

print("summary stats")
print(idata.agg(summary_funcs).T)
print("correlation")
print(idata.corr("pearson"))

idata["gID"] = [x[:3] for x in list(idata.index)]

for col in list(idata)[:-1]:
    X = idata[[col, "gID"]]
    model = f"{col} ~ gID"
    k = X.groupby("gID").count().mean()
    print(col)
    print(ia.icc1(X, model, k))

## Group Level
print("Group Level")

# recalculate burstiness and memory for the undifferentiated group data
# remaining vars are the sums of the individual level variables
gdata_temp = data.sort_values(["gID", "begin"])
gdata = gdata_temp.groupby("gID").agg(
    B = pd.NamedAgg(column = "lat", aggfunc = ia.bursty_coef),
    M = pd.NamedAgg(column = "lat", aggfunc = ia.memory_coef)
    )

idata_sums = idata.groupby("gID").agg(
    count = ("count", "sum"),
    tst = ("tst", "sum"),
    iss_sum = ("iss_sum", "sum"),
    iss_count = ("iss_count", "sum"),
    nss_sum = ("nss_sum", "sum"),
    nss_count = ("nss_count", "sum")
    )

gdata = pd.concat([gdata, idata_sums], axis = 1)

print("summary stats")
print(gdata.agg(summary_funcs).T)
print("correlations")
print(gdata.corr("pearson"))
