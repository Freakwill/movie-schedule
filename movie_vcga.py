#!/usr/bin/env python

from movie_problem import *


# define the multi-population
toolbox.register("populations", tools.initRepeat, list, lambda:toolbox.population(40), 20)

toolbox.register("mutate", mutRandom, indpb1=0.15, indpb2=0.8)
toolbox.register("select", tools.selTournament, tournsize=5)

if __name__ == "__main__":
    import multiprocessing
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    import vcga
    pops = toolbox.populations()
    ind = tools.selBest(pops[0], 1)[0]
    manager.schedule(ind)
    manager.print_fitness()
    aga = vcga.AdaptiveGA(algorithms.eaSimple, epochs=10)
    pga = vcga.RandomParallelGA(aga, epochs=2, send_best=10) 
    pga(pops, toolbox, cxpb=0.7, mutpb=0.32, ngen=5, verbose=False)
    ind = pga.selBest(pops)
    manager.schedule(ind)
    manager.check()
    manager.dumps()
    manager.plot()
    manager.print_criterion()
