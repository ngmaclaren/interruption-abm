# interruption-abm

This project builds an agent-based model of speech activity (who spoke when) in a small group. Agents are viewed as Markov chains which are either speaking or not with transition probabilities drawn from empirical or theoretical (fitted) distributions. This is a work in progress: code may not work as expected.

### Directory Structure

This code expects a project directory with `data` and `img` subdirectories. In `data` should be a `diarizations` subdirectory with the reference data and a `simulations` subdirectory to be filled by the generator scripts:

```
/interruption-abm/data/simulations: # to match /diarizations
./mimic-agents
./mimic-groups
./synthetic-groups-independent
./synthetic-groups-listening
./synthetic-groups-dyadic
```

