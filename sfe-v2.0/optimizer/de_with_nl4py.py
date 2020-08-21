from scipy.optimize import differential_evolution
from src.workspace import WorkspacePoolManager
from src.utils import get_parameters, get_linear_constraint
from src.objective_function import func
import multiprocessing as mp

### CONFIGURATION ###
PATH_TO_NETLOGO = "/Users/manilo/Documents/NetLogo 6.1.1"
PATH_TO_MODEL = "../sciadro.nlogo"
SCENARIO = "dump"
PATH_TO_PARAMETERS = "../scenarios/" + SCENARIO + "/bounds.txt"
NUM_SAMPLES = 3

if __name__ == "__main__":
    # select the method "fork" to start a process
    mp.set_start_method('fork')
    # create a NetLogo Headless Workspace pool manager
    manager = WorkspacePoolManager.create_manager(PATH_TO_NETLOGO, PATH_TO_MODEL)
    # get all the parameters of the model (at first the fixed parameters and then the parameters to optimize)
    parameter_names, fixed_parameter_values, parameters_to_optimize, bounds = get_parameters(PATH_TO_PARAMETERS)
    parameter_names.extend(parameters_to_optimize)
    #print(len(bounds))
    # get constraints on the solver, over and above those applied by the bounds
    lc = get_linear_constraint(parameters_to_optimize)
    # run Differential Evolution
    result = differential_evolution(func,
                                    bounds,
                                    args=(SCENARIO, parameter_names, fixed_parameter_values, NUM_SAMPLES),
                                    strategy='rand1bin',
                                    maxiter=20,
                                    popsize=5,
                                    tol=0.01,
                                    mutation=0.4,
                                    recombination=0.4,
                                    seed=None,
                                    callback=None,
                                    disp=True,
                                    polish=False,
                                    init='latinhypercube',
                                    atol=0,
                                    updating='deferred',
                                    workers=-1,
                                    constraints=(lc))
    # close all the opened models
    manager.close_models()
    # delete all the existing workspaces and their controllers currently on the NetLogoControllerServer
    manager.delete_workspaces()
    # stop the NetLogoControllerServer
    manager.stop_server()
