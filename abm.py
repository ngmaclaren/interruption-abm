import argparse
import os

import numpy as np
import scipy.stats as stats
import pandas as pd
import networkx as nx

import InterruptionAnalysis as ia

parser = argparse.ArgumentParser(description = """Simulate conversations for empirical or synthetic groups.""")

fit_or_random = parser.add_mutually_exclusive_group()
fit_or_random.add_argument('-e', '--exactfit',
                           help = """Draw network size and Markov transition matrices from the empirical groups. If this option is chosen, the default is to generate a randomized conversation for each group in the data set.""",
                           action = 'store_true')

fit_or_random.add_argument('-d', '--drawrandom',
                           help = """The default. Group size and Markov transition matrices will be drawn from a probability distribution fitted to the empirical data, but simulation parameters are not directly from the data.""",
                           action = 'store_true')

run_count = parser.add_mutually_exclusive_group()
run_count.add_argument('--nsims',
                       help = """Generate `nsims` (an integer) number of simulated groups and simulate their conversations. The default is 1. If used in conjunction with --exactfit, draws an empirical group to simulate with replacement `nsims` times.""",
                       #action = 'store_const',
                       nargs = '?',
                       type = int,
                       const = 1,
                       default = 1)#, nargs = '?')

run_count.add_argument('--nreps',
                       help = """Generate `nreps` (an integer) number of simulated conversations based on the observed groups. The default is 1. This option will go through each empirical group, estimate the Markov transition probability matrices for the members of that group, and run a simulated conversation with those matrices. Use only with --exactfit.""",
                       nargs = '?',
                       type = int,
                       const = 1,
                       default = 1)

# parser.add_argument('-w', '--withoutreplacement',
#                     help = """Used only with `--exactfit`. If not flagged, sample the empirical groups with replacement `nsims` times. If flagged, simulates each group exactly once and names each file for the empirical group it simulates.""",
#                     action = 'store_true')

parser.add_argument('--simtype',
                    help = """Choose which simulation to run. Options are `indep` (the default), for independent agents; `listening`, for agents that 'perceive' all other agents together as a group; and `dyadic`, for agents that assign a 'status' modifier to each other agent.""",
                    #action = 'store_const',
                    nargs = '?',
                    #const = 'indep',
                    #default = 'indep',
                    choices = ['indep', 'listening', 'dyadic'])

parser.add_argument('--minutes',
                    help = """Choose how many minutes of simulated conversation to produce. The default is 10, matching the empirical data.""",
                    #action = 'store_const',
                    nargs = '?',
                    type = int,
                    const = 10,
                    default = 10) # , nargs = '?'

parser.add_argument('--basescale',
                    help = """For testing the free parameter. Assign a value to the typical `scale` to be passed to the probability distribution that defines how agents vary in response to other agents' "speaking". Value for basescale will be converted to a float. The default is '1.0e-5'.""",
                    #action = 'store_const',
                    nargs = '?',
                    type = float,
                    const = '1.0e-5',
                    default = '1.0e-5')

parser.add_argument('--adjustscale',
                    help = """For testing the free parameter. A list of factors by which to adjust `basescale`. For example, if `--adjustscale .25 .5 1 2 4` is written, the simulation will run 5 * nsims times, with the scale parameter set to .25 * basescale for nsims runs, then .5 * basescale, and so on.""",
                    action = 'extend',
                    nargs = '+',
                    #type = float,
                    default = [])

parser.add_argument('--savepath',
                    help = """Assign a directory in which to store simulation results. The default is './data/sims/', and the program will create the directory if it does not exist. The program expects that the directory './data/' already exists: it expects to find the empirical data and the estimated beta distribution parameters for the Markov model transition probability matrices there.""",
                    #action = 'store_const',
                    nargs = '?',
                    type = str,
                    const = './data/sims/',
                    default = './data/sims/')

dist_choice = parser.add_mutually_exclusive_group()
dist_choice.add_argument('--beta_dists',
                         help = """Use fitted parameterizations of the beta distribution to generate the probability distribution from which to draw both the AB and BA transitions. Fits depend on whether or not simultaneous speech is included.""",
                         action = 'store_true')

dist_choice.add_argument('--low_AIC_dists',
                         help = """Regardless of whether simultaneous speech is ignored or considered, naively choose the distribution with the lowest AIC given the data for both the AB and BA transitions.""",
                         action = 'store_true')

dist_choice.add_argument('--theory_based_dists',
                         help = """Force the use of theoretically hypothesized probability models for the AB and BA transitions. The AB transition is modeled as a "time-to-event" process, using a Weibull distribution ('weibull_min' in SciPy); the BA transition is modeled as an event with a characteristic distribution but a heavy right tail, a lognormal (SciPy 'lognorm') distribution. Parameters are fit to data (i.e., either with or without simultaneous speech, depending on flag).""",
                         action = 'store_true')

parser.add_argument('--ignore_simul',
                    help = """If flagged, simultaneous speech will be ignored in the estimation of participant speech transition matrices.""",
                    action = 'store_true')

args = parser.parse_args()
print('Running simulation with the following options:')
print(args)
##########
# choose the sim
sim_choice = args.simtype #'indep'
if sim_choice == 'listening':
    import Listening as sim
elif sim_choice == 'dyadic':
    import Dyadic as sim
#elif sim_choice == 'indep':
else:
    import Independent as sim
##########

##########
# set the initial variable values
minutes = args.minutes #10
T = minutes*60*10

basescale = args.basescale #1.0e-5
if args.adjustscale:
    basescales = [float(adjust)*basescale for adjust in args.adjustscale] #[basescale/4, basescale/2, basescale, basescale*2, basescale*4]
else:
    basescales = [basescale]

nsims = args.nsims#1
##########

##########
savepath = args.savepath #'./data/sims/'

if not os.path.isdir(savepath):
    os.mkdir(savepath)
##########

if args.drawrandom:
    data = pd.read_csv('./data/timeseries.csv', index_col = 0)
    gIDs = pd.unique(data['gID'])
    inets = {}
    for gID in gIDs:
        inets[gID] = nx.read_gml(f'./data/networks/emp/{gID}.gml')
    sizes = [len(g) for g in inets.values()]
    possible_N = pd.unique(sizes) # give the user access to possible_N
    ps = [len([s for s in sizes if s == x])/len(sizes) for x in possible_N]

    Ns = np.random.choice(possible_N, size = nsims, p = ps, replace = True)

    if args.ignore_simul:
        dists = pd.read_csv('./data/fits-nosimul.csv', index_col = 0)
    else:
        dists = pd.read_csv('./data/fits.csv', index_col = 0)

    if args.beta_dists:
        chosen_dists = dists[dists['dist'] == 'beta']
    elif args.low_AIC_dists:
        chosen_dists = dists.loc[dists.groupby('transition')['ΔAIC'].idxmin()]
    elif args.theory_based_dists:
        d1 = dists[(dists['transition'] == 'AB') & (dists['dist'] == 'weibull_min')]
        d2 = dists[(dists['transition'] == 'BA') & (dists['dist'] == 'lognorm')]
        chosen_dists = pd.concat([d1, d2])

    chosen_dists.set_index('transition', inplace = True)
    fitrows = ['AB', 'BA']
    fits = {}
    for row in fitrows:
        dist = getattr(stats, chosen_dists.loc[row, 'dist'])
        args = [chosen_dists.loc[row, 'arg1'], chosen_dists.loc[row, 'arg2']]
        arg = [a for a in args if ~np.isnan(a)]
        loc = chosen_dists.loc[row, 'loc']
        scale = chosen_dists.loc[row, 'scale']
        if arg:
            fit = dist(loc = loc, scale = scale, *arg)
        else:
            fit = dist(loc = loc, scale = scale)
        fits[row] = fit
        
    for scale in basescales:
        for run, N in zip(range(nsims), Ns):
            ns = list(range(N))

            if run % 10 == 0:
                print(run)

            P = {}
            for n in ns: ## forgot to comment, but flipped BA -> AB etc. below. 
                AB = fits['AB'].rvs()
                AA = 1 - AB
                BA = fits['BA'].rvs()
                BB = 1 - BA
                P[n] = np.array([
                    [AA, AB],
                    [BA, BB]
                ])

            if sim_choice in ['dyadic', 'listening']:
                Y = sim.simulation(P, T, N, ns, scale)
            elif sim_choice == 'indep':
                Y = sim.simulation(P, T, N, ns)
            else:
                print('You need to specify a simulation.')
                break

            X = ia.Y_to_X(Y, ns)
            X['scale'] = scale
            X.to_csv(f'{savepath}/{sim_choice}-{scale}-{run}.csv') # basescales.index(scale)

elif args.exactfit:

    data = pd.read_csv('./data/timeseries.csv', index_col = 0)
    numericcols = ['begin', 'end', 'dur', 'lat']
    for col in numericcols:
        data[col] /= 100
    Rs = pd.unique(data['gID'])

    if args.nreps: #args.withoutreplacement:
        nreps = args.nreps
        
        for rep in range(nreps):

            if rep % 10 == 0:
                print(rep)

            #Rs = ['YMY']
            for R in Rs:
                #print(R)
                dat = data[data['gID'] == R] # leave it ~dat~ b/c col names reflect coding
                r_ns = pd.unique(dat['pID'])
                t_max = int(np.ceil(max(dat['end'])))
                T = range(t_max)
                N = len(r_ns)
                ns = list(range(N))

                P = {n: ia.get_transition_matrix(dat, r_n) for n, r_n in zip(ns, r_ns)}

                Y = sim.simulation(P, t_max, N, ns)#, basescale)
                X = ia.Y_to_X(Y, ns)
                X['scale'] = np.nan
                X['gID'] = R
                X['rep'] = rep
                #if args.save:
                X.to_csv(f'{savepath}/exactfit-{R}-{rep}.csv')
            
    else:
        nsims = args.nsims

        for run in range(nsims):
            if run % 10 == 0:
                print(run)

            R = np.random.choice(Rs)
            dat = data[data['gID'] == R] # leave it ~dat~ b/c col names reflect coding
            r_ns = pd.unique(dat['pID'])
            t_max = int(np.ceil(max(dat['end'])))
            T = range(t_max)
            N = len(r_ns)
            ns = list(range(N))

            P = {n: ia.get_transition_matrix(dat, r_n) for n, r_n in zip(ns, r_ns)}

            Y = sim.simulation(P, t_max, N, ns)#, basescale)
            X = ia.Y_to_X(Y, ns)
            X['scale'] = np.nan
            X['gID'] = R
            #if args.save:
            X.to_csv(f'{savepath}/exactfit-{run}.csv')

print(f'Done. Simulation results are saved in {savepath}.')
    # beta_dists = pd.read_csv('./data/beta-dists.csv', index_col = 1)
    # # fitrows = ['BA', 'AB']
    # # fits = {row: stats.beta(
    # #     beta_dists.loc[row, 'arg1'],
    # #     beta_dists.loc[row, 'arg2'],
    # #     beta_dists.loc[row, 'loc'],
    # #     beta_dists.loc[row, 'scale']) for row in fitrows}
    
