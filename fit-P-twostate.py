import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats._continuous_distns import _distn_names
from scipy.special import rel_entr

import InterruptionAnalysis as ia

ignore = False
if ignore:
    savepath = './data/fits-nosimul.csv'
else:
    savepath = './data/fits.csv'

data = pd.read_csv('./data/timeseries.csv', index_col = 0)

numeric_cols = ['begin', 'end', 'dur', 'lat']
for col in numeric_cols:
    data[col] = data[col]/100 # converts to 1/10th seconds

gIDs = pd.unique(data['gID'])#['YMY']

pdata = []

for gID in gIDs:
    dat = data[data['gID'] == gID]
    pIDs = pd.unique(dat['pID'])
    t_max = max(dat['end'])
    T = range(int(np.ceil(t_max)))

    P = {pID: ia.get_transition_matrix(dat, pID, ignore_simultaneous = ignore) for pID in pIDs}

    for pID in pIDs:
        row = {#(gID, pID, P[pID][0, 0], P[pID][1, 0], P[pID][0, 1], P[pID][1, 1]) #### This is wrong.
            'gID': gID, 'pID': pID,
            'AA': P[pID][0, 0], 'AB': P[pID][0, 1],
            'BA': P[pID][1, 0], 'BB': P[pID][1, 1]
        }
        pdata.append(row)

# keep in mind A is Pr('o') and B is Pr('e')
pdata = pd.DataFrame(pdata)#, columns = ['gID', 'pID', 'AA', 'AB', 'BA', 'BB'])

cols = ['AA', 'AB', 'BA', 'BB']
results = []
AICresults = {}

#distributions = [getattr(stats, distname) for distname in _distn_names if distname != 'levy_stable']
testdists = ['norm', 'lognorm', 'weibull_max', 'weibull_min', 'pareto', 'beta', 'gumbel_l', 'gumbel_r', 'johnsonsb', 'johnsonsu']
distributions = [getattr(stats, distname) for distname in testdists]

for col in cols:

    dat = pdata[col].dropna()
    klds = {}
    AICs = {}
    n = len(dat)
    bins = 75

    y, x = np.histogram(dat, bins = bins, density = True)
    x = (x + np.roll(x, -1))[:-1] / 2.0 # makes x the midpoint of the bin
    
    for dist in distributions:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore') # not sure what this does either

                params = dist.fit(dat)
                arg = params[:-2]#; print([dist.name, arg])
                loc = params[-2]
                scale = params[-1]
                k = len(params)

                logLik = np.sum(dist.logpdf(dat, loc = loc, scale = scale, *arg))
                AIC = 2*k - 2*(logLik)

                AICs[dist.name] = AIC

                if arg:
                    arg1 = arg[0]
                    if len(arg) > 1:
                        arg2 = arg[1]
                    else:
                        arg2 = np.nan
                else:
                    arg1, arg2 = np.nan, np.nan

                result = (col, dist.name, AIC, arg1, arg2, loc, scale)
                results.append(result)
        except Exception:
            klds[dist.name] = np.nan
        
    
    AICresults[col] = AICs

results_cols = ['transition', 'dist', 'AIC', 'arg1', 'arg2', 'loc', 'scale']
results = pd.DataFrame(results, columns = results_cols)
minAIC = {col: min(results[results['transition'] == col]['AIC']) for col in cols}
results['ΔAIC'] = np.nan
for i in range(len(results)):
    results.loc[i, 'ΔAIC'] = results.loc[i, 'AIC'] - minAIC[results.loc[i, 'transition']]

AICresults = pd.DataFrame(AICresults)
AICresults = AICresults.replace(np.inf, np.nan)
#AICresults = AICresults.drop('vonmises')
for c in list(AICresults):
    AICresults[c] = AICresults[c].apply(lambda x: x - min(AICresults[c]))

#dist_choice = {'AA': 'weibull_max', 'AB': 'lognorm', 'BA': 'weibull_min', 'BB': 'beta'}
dist_choice = {col: AICresults[col][AICresults[col] == min(AICresults[col])].index.tolist()[0] for col in cols}

solarized = {'base03': '#002b36', 'base02': '#073642', 'base01': '#586e75', 'base00': '#657b83', 'base0': '#839496', 'base1': '#93a1a1', 'base2': '#eee8d5', 'base3': '#fdf6e3', 'yellow': '#b58900', 'orange': '#cb4b16', 'red': '#dc322f', 'magenta': '#d33682', 'violet': '#6c71c4', 'blue': '#268bd2', 'cyan': '#2aa198', 'green': '#859900'}

fig, ax = plt.subplots(1, 2, tight_layout = True, figsize = (10, 5))
fig.tight_layout()

color_choices = ['yellow', 'orange', 'red', 'magenta', 'violet', 'blue', 'cyan', 'green', 'base03', 'base1']
colors = [solarized[choice] for choice in color_choices]
colors = colors[:len(testdists)]

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')

    for a, col in zip(ax.flatten(), ["AB", "BA"]):
        dat = pdata[col].dropna()
        y, x, patches = a.hist(dat, density = True, bins = 75, alpha = .5)
        x = (x + np.roll(x, -1))[:-1] / 2.0
        #dist_name = dist_choice[col]
        for distribution, color in zip(testdists, colors):
            dist = getattr(stats, distribution)
            params = dist.fit(dat)
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]
            pdf = dist.pdf(x, loc = loc, scale = scale, *arg)
            a.plot(x, pdf, color = color, linewidth = 3,
                   label = dist.name)
        #a.set_title(f'{col}')
        a.legend()

ax[0].text(0.01, 0.99, "A", horizontalalignment = "left", verticalalignment = "top", transform = ax[0].transAxes)
ax[1].text(0.01, 0.99, "B", horizontalalignment = "left", verticalalignment = "top", transform = ax[1].transAxes)
        
if ignore:
    fig.savefig('./img/distribution-fits-nosimul.svg')
else:
    fig.savefig('./img/distribution-fits.svg')
plt.show()

results.to_csv(savepath)
