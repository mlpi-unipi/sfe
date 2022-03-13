import nl4py
import threading
import sys

class WorkspacePoolManager:

    __instance = None
    __resources = []

    def __init__(self, path_to_netlogo_, path_to_model_):
        if WorkspacePoolManager.__instance != None:
            raise NotImplementedError("This is a singleton class.")
        self.path_to_netlogo = path_to_netlogo_
        self.path_to_model = path_to_model_
        self.lock = threading.Lock()
        self.start_server()

    @staticmethod
    def create_manager(path_to_netlogo, path_to_model):
        if WorkspacePoolManager.__instance == None:
            WorkspacePoolManager.__instance = WorkspacePoolManager(path_to_netlogo, path_to_model)
        return WorkspacePoolManager.__instance

    @staticmethod
    def get_manager():
        return WorkspacePoolManager.__instance
    
    def start_server(self):
        # start the NetLogoControllerServer
        nl4py.initialize(self.path_to_netlogo)
        print("NetLogoControllerServer started!")

    def get_workspace(self):
        with self.lock:
            if len(self.__resources) > 0:
                return self.__resources.pop(0)
            else:
                # NetLogoControllerServer creates a new HeadlessWorkspace and returns its controller for access from Python
                workspace = nl4py.create_headless_workspace()
                # open the NetLogo model from the specified file within the NetLogoHeadlessWorkspace
                workspace.open_model(self.path_to_model)
                return workspace

    def release_workspace(self, workspace):
        with self.lock:
            self.__resources.append(workspace)
            #print("Number of resources:", len(self.__resources))

    def close_models(self):
        # close the opened models
        for i in range(len(self.__resources)):
            self.__resources[i].close_model()

    def delete_workspaces(self):
        # delete all the existing workspaces and their controllers currently on the NetLogoControllerServer
        nl4py.delete_all_headless_workspaces()

    def stop_server(self):
        # stop the NetLogoControllerServer
        # @deprecated("Global shutdown not required anymore.")
        nl4py.stopServer()
        print("NetLogoControllerServer stopped!")
