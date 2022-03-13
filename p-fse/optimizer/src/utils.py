from scipy.optimize import LinearConstraint
import numpy as np
import time
import pickle

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

def simulate(workspace, scenario, parameter_names, parameter_values):
    workspace.command("stop")
    # select the right scenario
    workspace.command("set scenario " + "\"" + scenario + "\"")
    # create empty lists for parameters and values
    workspace.command("set parameters []")
    workspace.command("set values []")
    # append to each list the parameters and the values
    for name, value in zip(parameter_names, parameter_values):
        cmd = "set parameters lput \"{0}\" parameters".format(name)
        workspace.command(cmd)
        cmd = "set values lput {0} values".format(value)
        workspace.command(cmd)
    # run the model until a stop criterion is met
    workspace.command("run-simulation parameters values")
    # wait for the completion of the simulation
    ticks = -1
    while ticks == -1 or ticks == 0:
        # get the time needed to complete the mission (for static scenarios)
        ticks = workspace.report("fitness-value")
    time.sleep(2)
    return ticks

def write_on_shared_memory(shared_memory, lock, data_to_write):
    """
    Writes data on the shared memory buffer

    Parameters
    ----------
    shared_memory: multiprocessing.shared_memory.SharedMemory
        Shared memory on which we want to write data
    
    lock: multiprocessing.synchronize.Lock or multiprocessing.synchronize.RLock
        Lock used to grant mutually exclusive access to shared memory
    
    data_to_write:
        Data to be written on the shared buffer.
        It can be of any built-in type
    """
    data_binary = pickle.dumps(data_to_write, protocol=pickle.HIGHEST_PROTOCOL)
    with lock:
        shared_memory.buf[:len(data_binary)] = data_binary

def extract_from_shared_memory(shared_memory, lock):
    """
    Extracts and returns the content of the shared memory buffer

    Parameters
    ----------
    shared_memory: multiprocessing.shared_memory.SharedMemory
        Shared memory storing data to be extracted
    
    lock: multiprocessing.synchronize.Lock or multiprocessing.synchronize.RLock
        Lock used to grant mutually exclusive access to shared memory
    
    Returns
    -------
    extracted_data:
        Data stored in the shared buffer
    """
    with lock:
        extracted_data = pickle.loads(shared_memory.buf)
    return extracted_data
