from src.workspace import WorkspacePoolManager
from src.utils import simulate
from src.log import LogManager
import numpy as np
import math
import scipy.stats as stats
import time

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
    #print(parameters)
    #print(values)
    # get the instance of the singleton class
    workspace_manager = WorkspacePoolManager.get_manager()
    # run the simulation num_samples times and collect the results
    fitness_list = []
    workspace = workspace_manager.get_workspace()
    start_time = time.time()
    for i in range(num_samples):
        # fitness: ticks or target average percentage
        fitness = simulate(workspace, scenario, parameters, values)
        # append the sample to the list
        fitness_list.append(fitness)
    end_time = time.time() - start_time
    # add the workspace to the pool of free resources
    workspace_manager.release_workspace(workspace)
    # convert the list of the samples into a numpy array
    fitness_list = [float(i) for i in fitness_list]
    sample = np.array(fitness_list)
    # get the mean of the samples
    sample_mean = sample.mean()
    # get the standard deviation of the samples
    sample_std = sample.std()
    # estimate the standard deviation of the population of the mean values of the samples (of num_samples dimension)
    sigma = sample_std / math.sqrt(num_samples)
    # get the confidence interval for the mean value
    ci = stats.t.interval(  alpha=0.95,             # level of confidence at 95%
                            df=num_samples - 1,     # degrees of freedom
                            loc=sample_mean,        # mean value of the samples
                            scale=sigma )           # standard deviation of the mean values
    # get the upper bound (worst case)
    fitness_value = ci[1]
    # get the instance of the (singleton) class LogManager
    log_manager = LogManager.get_manager()
    # log the evaluated individual
    parameters_to_optimize = parameters[-len(individual):]
    log_manager.log_individual(parameters_to_optimize, individual, fitness_value, end_time)
    # print the individual
    print("\n", x, ":", fitness_value, "\n")
    return fitness_value
