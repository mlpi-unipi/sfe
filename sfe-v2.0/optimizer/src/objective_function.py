from src.workspace import WorkspacePoolManager
from src.utils import simulate
import numpy as np
import math
import scipy.stats as stats

# define the objective function to be minimized
def func(x, *args):
    scenario = args[0]              # name of the scenario
    parameters = [] + args[1]       # names of the parameters
    values = [] + args[2]           # values of the fixed parameters
    num_samples = args[3]           # number of simulations
    # convert the individual from a 1-D array into a list
    individual = x.tolist()
    #print(len(x))
    values += individual
    # get the instance of the singleton class
    manager = WorkspacePoolManager.get_manager()
    # run the simulation num_samples times and collect the results
    fitness = []
    workspace = manager.get_workspace()
    for i in range(num_samples):
        ticks = simulate(workspace, scenario, parameters, values)
        # append the sample to the list
        fitness.append(ticks)
    # add the workspace to the pool of free resources
    manager.release_workspace(workspace)
    # convert the list of the samples into a numpy array
    sample = np.array(fitness)
    # get the mean of the samples
    sample_mean = sample.mean()
    # get the standard deviation of the samples
    sample_stdev = sample.std()
    # estimate the standard deviation of the population of the mean values of the samples (of num_samples dimension)
    sigma = sample_stdev / math.sqrt(num_samples)
    # get the confidence interval for the mean value
    ci = stats.t.interval(  alpha=0.95,             # level of confidence at 95%
                            df=num_samples - 1,     # degrees of freedom
                            loc=sample_mean,        # mean value of the samples
                            scale=sigma )           # standard deviation of the mean values
    # get the upper bound (worst case)
    fitness_value = ci[1]
    print(x, ":", fitness_value)
    return fitness_value
