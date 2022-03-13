from scipy.optimize import differential_evolution
from src.workspace import WorkspacePoolManager
from src.log import LogManager
from src.utils import get_parameters, get_linear_constraint
from src.objective_function import func
import multiprocessing as mp
import sys

### CONFIGURATION ###
PATH_TO_NETLOGO = "/Users/manilo/Documents/Dottorato/NetLogo 6.2.0"
PATH_TO_MODEL = "../sciadro.nlogo"
SCENARIO = "dump"
PATH_TO_PARAMETERS = "../scenarios/" + SCENARIO + "/bounds.txt"
NUM_NL_SAMPLES = 30
NUM_DE_SAMPLES = 10
PATH_TO_LOGS = "logs/" + SCENARIO
LOG_SHARED_MEMORY_BLOCK_SIZE = 2097152          # 2MB,  size of the shared memory block to contain the log data dictionary
GENERATION_SHARED_MEMORY_BLOCK_SIZE = 65536     # 64KB, size of the shared memory block to contain a list of all the individual of the current generation

if __name__ == "__main__":
    # select the method "fork" to start a process
    mp.set_start_method('fork')
    for i in range(1, NUM_DE_SAMPLES + 1):
        PATH_TO_LOG = PATH_TO_LOGS + "_log_" + str(i) + ".json"
        # create a NetLogo Headless Workspace pool manager
        workspace_manager = WorkspacePoolManager.create_manager(PATH_TO_NETLOGO, PATH_TO_MODEL)
        # create the shared memory blocks to be used for log management
        new_log_shared_memory = mp.shared_memory.SharedMemory(create=True, size=LOG_SHARED_MEMORY_BLOCK_SIZE)
        new_generation_shared_memory = mp.shared_memory.SharedMemory(create=True, size=GENERATION_SHARED_MEMORY_BLOCK_SIZE)
        new_generation_counter_shared_memory = mp.shared_memory.SharedMemory(create=True, size=sys.getsizeof(int()))
        new_individual_counter_shared_memory = mp.shared_memory.SharedMemory(create=True, size=sys.getsizeof(int()))
        # create a log manager
        log_manager = LogManager.create_manager(PATH_TO_LOG,
                                                new_log_shared_memory.name,
                                                new_generation_shared_memory.name,
                                                new_generation_counter_shared_memory.name,
                                                new_individual_counter_shared_memory.name)
        # get all the parameters of the model (at first the fixed parameters and then the parameters to optimize)
        parameter_names, fixed_parameter_values, parameters_to_optimize, bounds = get_parameters(PATH_TO_PARAMETERS)
        parameter_names.extend(parameters_to_optimize)
        #print(len(bounds))
        # get constraints on the solver, over and above those applied by the bounds
        lc = get_linear_constraint(parameters_to_optimize)
        # run Differential Evolution
        result = differential_evolution(func,
                                        bounds,
                                        args=(SCENARIO, parameter_names, fixed_parameter_values, NUM_NL_SAMPLES),
                                        strategy='rand1bin',
                                        maxiter=20,
                                        popsize=5,
                                        tol=0.01,
                                        mutation=0.4,
                                        recombination=0.4,
                                        seed=None,
                                        callback=log_manager.callback,
                                        disp=True,
                                        polish=False,
                                        init='latinhypercube',
                                        atol=0,
                                        updating='deferred',
                                        workers=-1,
                                        constraints=(lc))
        # close and deallocate all the shared memory blocks
        new_log_shared_memory.close()
        new_log_shared_memory.unlink()
        new_generation_shared_memory.close()
        new_generation_shared_memory.unlink()
        new_generation_counter_shared_memory.close()
        new_generation_counter_shared_memory.unlink()
        new_individual_counter_shared_memory.close()
        new_individual_counter_shared_memory.unlink()
        # close all the opened models
        workspace_manager.close_models()
        # delete all the existing workspaces and their controllers currently on the NetLogoControllerServer
        workspace_manager.delete_workspaces()
