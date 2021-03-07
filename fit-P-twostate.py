import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats._continuous_distns import _distn_names
from scipy.special import rel_entr

import InterruptionAnalysis as ia

ignore = True
if ignore:
    savepath = './data/fits-nosimul.csv'
else:
    savepath = './data/fits.csv'

# get `pdata` from markov-analysis.py
data = pd.read_csv('./data/timeseries.csv', index_col = 0)

# Uncomment to analyze in 1/10th s rather than ms
numeric_cols = ['begin', 'end', 'dur', 'lat']
for col in numeric_cols:
    data[col] = data[col]/100 # converts to 1/10th seconds

gIDs = pd.unique(data['gID'])#['YMY']
#Ps = {} # may not need this

pdata = []

for gID in gIDs:
    #print(gID)
    dat = data[data['gID'] == gID]
    pIDs = pd.unique(dat['pID'])
    t_max = max(dat['end'])
    T = range(int(np.ceil(t_max)))
    #N = len(pIDs)
    #ns = list(range(N))

    P = {pID: ia.get_transition_matrix(dat, pID, ignore_simultaneous = ignore) for pID in pIDs}
    #Ps[gID] = P # may not need this
    for pID in pIDs:
        row = {#(gID, pID, P[pID][0, 0], P[pID][1, 0], P[pID][0, 1], P[pID][1, 1]) #### This is wrong.
            'gID': gID, 'pID': pID,
            'AA': P[pID][0, 0], 'AB': P[pID][0, 1],
            'BA': P[pID][1, 0], 'BB': P[pID][1, 1]
        }
        pdata.append(row)

# keep in mind A is Pr('o') and B is Pr('e')
pdata = pd.DataFrame(pdata)#, columns = ['gID', 'pID', 'AA', 'AB', 'BA', 'BB'])


#cols = ['AA', 'BA', 'AB', 'BB']
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
        #print(dist.name)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore') # not sure what this does either

                params = dist.fit(dat)
                arg = params[:-2]#; print([dist.name, arg])
                loc = params[-2]
                scale = params[-1]
                k = len(params)
                
                #kldlist = []
                #AIClist = []
                #iters = 100
                #for _ in range(iters):
                #pdf = dist.pdf(x, loc = loc, scale = scale, *arg)
                #kld = rel_entr(y, pdf)
                #kldlist.append(kld)

                logLik = np.sum(dist.logpdf(dat, loc = loc, scale = scale, *arg))
                AIC = 2*k - 2*(logLik)


                
                #kld = np.mean(kldlist)
                #klds[dist.name] = kld
                AICs[dist.name] = AIC

                if arg:
                    arg1 = arg[0]
                    if len(arg) > 1:
                        arg2 = arg[1]
                    else:
                        arg2 = np.nan
                else:
                    arg1, arg2 = np.nan, np.nan

                #arg = [arg[i] for i in range(2) if arg[i] else np.nan]
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

AICresults = pd.DataFrame(AICresults)#pd.DataFrame(results)
AICresults = AICresults.replace(np.inf, np.nan)
#AICresults = AICresults.drop('vonmises')
for c in list(AICresults):
    AICresults[c] = AICresults[c].apply(lambda x: x - min(AICresults[c]))


# print()
# for col in cols:
#     ##bestdist = AICresults[AICresults[col] == min(abs(AICresults[col]))].index.values[0]
#     ##print(f'{col}: {bestdist}')
#     print(AICresults[col].sort_values().head(10))

# good candidates:
# weibull, gamma, lognormal... these are normal, pardon the pun, distributions.
# Maybe choose:

#dist_choice = {'AA': 'weibull_max', 'AB': 'lognorm', 'BA': 'weibull_min', 'BB': 'beta'}
dist_choice = {col: AICresults[col][AICresults[col] == min(AICresults[col])].index.tolist()[0] for col in cols}

solarized = {'base03': '#002b36', 'base02': '#073642', 'base01': '#586e75', 'base00': '#657b83', 'base0': '#839496', 'base1': '#93a1a1', 'base2': '#eee8d5', 'base3': '#fdf6e3', 'yellow': '#b58900', 'orange': '#cb4b16', 'red': '#dc322f', 'magenta': '#d33682', 'violet': '#6c71c4', 'blue': '#268bd2', 'cyan': '#2aa198', 'green': '#859900'}

fig, ax = plt.subplots(2, 2, tight_layout = True, figsize = (15, 10))

color_choices = ['yellow', 'orange', 'red', 'magenta', 'violet', 'blue', 'cyan', 'green', 'base03', 'base1']
colors = [solarized[choice] for choice in color_choices]
colors = colors[:len(testdists)]

with warnings.catch_warnings():
    warnings.filterwarnings('ignore') # not sure what this does either

    for a, col in zip(ax.flatten(), cols):
        dat = pdata[col].dropna()
        #a.hist(np.log(pdata[cols[i]]), bins = 20)
        #weights = np.ones_like(list(dat))/len(dat)
        y, x, patches = a.hist(dat, density = True, bins = 75, alpha = .5)
        x = (x + np.roll(x, -1))[:-1] / 2.0 # still don't understand why we do this
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
        a.set_title(f'{col}')
        a.legend()

        # dist_name = 'beta'
        # dist = getattr(stats, dist_name)
        # params = dist.fit(dat)
        # arg = params[:-2]
        # loc = params[-2]
        # scale = params[-1]
        # pdf = dist.pdf(x, loc = loc, scale = scale, *arg)
        # a.plot(x, pdf, 'k--')
if ignore:
    fig.savefig('./img/distribution-fits-nosimul.png')
else:
    fig.savefig('./img/distribution-fits.png')
plt.show()

# choosing beta distributions makes theoretical sense because we're talking about probabilities: they are only defined [0, 1]. That's just not a Weibull, even though otherwise the interpretation of the Weibull makes sense. Of course, as with any distribution, there are many interpretations---the interpretation is up to me.
# however, beta distributions are apparently weird right at 0 and 1. Scipy implementation is 0 <= x <= 1.
# and currently I'm essentially treating all the zeros as NA. What if they really are zero? They would be 1 for AA, 0 for BA, and undefined for AB and BB. 


## produce a data file that contains the parameters for the four beta functions

results.to_csv(savepath)

# parameters = {}
# for col in cols:
#     dat = pdata[col]
#     y, x, patches = a.hist(dat, density = True, bins = 75)
#     x = (x + np.roll(x, -1))[:-1] / 2.0
#     dist_name = 'beta'
#     dist = getattr(stats, dist_name)
#     params = dist.fit(dat)
#     parameters[col] = params

# parameters = pd.DataFrame.from_dict(parameters, orient = 'index',
#                                     columns = ['a', 'b', 'loc', 'scale'])

# parameters.to_csv('./data/beta-dists.csv')
