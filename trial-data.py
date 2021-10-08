import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import InterruptionAnalysis as ia

def count_iss(dat): # sort to gID first
    pIDs = pd.unique(dat["pID"])
    iss = ia.interruptive_simultaneous_speech(dat, pIDs)
    if iss.empty:
        iss = pd.DataFrame({"iss_count": 0}, index = pIDs)
        iss.index.rename("i", inplace = True)
        return iss
    sorter = iss.groupby(["i", "begin"])["dur"].count()
    sorter = list(sorter[sorter > 1].index)
    for row in sorter:
        temp = iss.loc[(iss["i"] == row[0]) & (iss["begin"] == row[1]), ]
        drop = temp.loc[temp["dur"] != max(temp["dur"]), ]
        iss.drop(list(drop.index), inplace = True)
    iss = iss.groupby("i").agg(iss_count = ("dur", "count"))
    return iss

## empirical
print("empirical")
print("...")
data = pd.read_csv("./data/timeseries.csv", index_col = 0)
numeric_cols = ["begin", "end", "dur", "lat"]
for col in numeric_cols:
    data[col] = data[col]/100 # converts to 1/10th seconds
gIDs = pd.unique(data["gID"])

data_tst = data.groupby("gID").agg(sum_tst = ("dur", "sum"))
data_iss = [count_iss(data.loc[data["gID"] == gID, ]) for gID in gIDs]
data_iss = pd.concat(data_iss)
data_iss["gID"] = data_iss.index.map(lambda x: x[:3])
data_iss = data_iss.groupby("gID").agg(count_iss = ("iss_count", "sum"))

emp = pd.concat([data_tst, data_iss], axis = 1)
emp["sim"] = "emp"
emp.reset_index(inplace = True)
emp.to_csv("./data/emp-tst-iss.csv")

## simulated
print("simulated")
paths = ["./data/simulations/mimic-groups", "./data/simulations/synthetic-groups-independent",
         "./data/simulations/synthetic-groups-listening", "./data/simulations/synthetic-groups-dyadic"]

for path in paths:
    print(path)
    dat = []
    for root, dirs, files in os.walk(path):
        if not files:
            continue
        for f in files:
            X = pd.read_csv(os.path.join(root, f), index_col = 0)
            X["sim"] = root.replace(path + "/", "")
            dat.append(X)
    dat = pd.concat(dat)

    dat_tst = dat.groupby(["sim", "gID"]).agg(sum_tst = ("dur", "sum"))

    dat_iss = []
    for sim in os.listdir(path):
        print(sim)
        for gID in gIDs:
            d = dat.loc[(dat["gID"] == gID) & (dat["sim"] == sim), ]
            iss = count_iss(d)
            iss["sim"] = sim
            iss["gID"] = gID
            dat_iss.append(iss)
    dat_iss = pd.concat(dat_iss)
    dat_iss = dat_iss.groupby(["sim", "gID"]).agg(count_iss = ("iss_count", "sum"))

    simul = pd.concat([dat_tst, dat_iss], axis = 1)
    simul.reset_index(inplace = True)

    nomen = path.replace("./data/simulations/", "")
    simul.to_csv(f"./data/{nomen}-tst-iss.csv")



print("Done.")
