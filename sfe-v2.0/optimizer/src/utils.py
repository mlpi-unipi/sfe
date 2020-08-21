from scipy.optimize import LinearConstraint
import numpy as np
import time

def get_parameters(*args):
    parameters_to_set = []
    values = []
    parameters_to_optimize = []
    bounds = []
    # open the configuration file in reading mode
    configuration = open(args[0], 'r')
    # loop through the file line by line
    for line in configuration:
        # if the line is empty or contains a comment, then go to the next line
        if line.strip() == "" or line.strip()[0] == "#":
            continue
        # split the line between parameter name and value
        parameter_name = line.split("=")[0].strip()
        parameter_value = line.split("=")[1].strip()
        # append the parameter to the right list
        if parameter_value[0] == "(":
            parameters_to_optimize.append(parameter_name)
            bounds.append(eval(parameter_value.replace("(", "").replace(")", "")))
        else:
            parameters_to_set.append(parameter_name)
            values.append(float(parameter_value))
    configuration.close()
    # return the lists of the parameters
    return parameters_to_set, values, parameters_to_optimize, bounds

def get_linear_constraint(parameters_to_optimize):
    # Linear constraints on the parameters:
    # 1) mark-radius-top <= mark-radius-down
    a_1 = np.zeros(len(parameters_to_optimize))
    # 2) separate-radius <= align-radius
    a_2 = np.zeros(len(parameters_to_optimize))
    # 3) separate-radius <= cohere-radius
    a_3 = np.zeros(len(parameters_to_optimize))
    # 4) align-radius <= cohere-radius
    a_4 = np.zeros(len(parameters_to_optimize))
    # define the first constraint
    if "radius-top" in parameters_to_optimize and "radius-down" in parameters_to_optimize:
        a_1[parameters_to_optimize.index("radius-top")] = 1
        a_1[parameters_to_optimize.index("radius-down")] = -1
    # define the second constraint
    if "separate-radius" in parameters_to_optimize and "align-radius" in parameters_to_optimize:
        a_2[parameters_to_optimize.index("separate-radius")] = 1
        a_2[parameters_to_optimize.index("align-radius")] = -1
    # define the third constraint
    if "separate-radius" in parameters_to_optimize and "cohere-radius" in parameters_to_optimize:
        a_3[parameters_to_optimize.index("separate-radius")] = 1
        a_3[parameters_to_optimize.index("cohere-radius")] = -1
    # define the fourth constraint
    if "align-radius" in parameters_to_optimize and "cohere-radius" in parameters_to_optimize:
        a_4[parameters_to_optimize.index("align-radius")] = 1
        a_4[parameters_to_optimize.index("cohere-radius")] = -1
    # create the matrix defining the constraint
    A = np.array((a_1, a_2, a_3, a_4)).reshape((4,len(parameters_to_optimize)))
    # create and return the LinearConstraint object on the variables
    lc = LinearConstraint(A, -np.inf, 0)
    return lc

def simulate(workspace_, scenario, parameter_names, parameter_values):
    workspace_.command("stop")
    # select the right scenario
    workspace_.command("set scenario " + "\"" + scenario + "\"")
    # create empty lists for parameters and values
    workspace_.command("set parameters []")
    workspace_.command("set values []")
    # append to each list the parameters and the values
    for name, value in zip(parameter_names, parameter_values):
        cmd = "set parameters lput \"{0}\" parameters".format(name)
        workspace_.command(cmd)
        cmd = "set values lput {0} values".format(value)
        workspace_.command(cmd)
    # run the model until a stop criterion is met
    workspace_.command("run-simulation parameters values")
    # wait for the completion of the simulation
    ticks = -1
    while ticks == -1 or ticks == 0:
        # get the time needed to complete the mission (for static scenarios)
        ticks = workspace_.report("fitness-value")
    time.sleep(2)
    return ticks
    