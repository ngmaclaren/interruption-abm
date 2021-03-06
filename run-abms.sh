#!/bin/sh

nsims=100 # $1
nreps=10
basescale='1.0e-5'

# python abm.py -e --nreps 10 --savepath './data/sims/exactfit/'

python abm.py -d --low_AIC_dists --nsims 330 --simtype indep --savepath './data/sims/indep/'

python abm.py -d --low_AIC_dists --ignore_simul --nsims 330 --simtype indep --savepath './data/sims/indep-nosimul/'

python abm.py -d --low_AIC_dists --ignore_simul --nsims 330 --simtype listening --savepath './data/sims/listening-lowAIC/'

python abm.py -d --theory_based_dists --ignore_simul --nsims 330 --simtype listening --savepath './data/sims/listening/'

python abm.py -d --low_AIC_dists --ignore_simul --nsims 330 --simtype dyadic --savepath './data/sims/dyadic-lowAIC/'

python abm.py -d --theory_based_dists --ignore_simul --nsims 330 --simtype dyadic --savepath './data/sims/dyadic/'
