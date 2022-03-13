from multiprocessing import shared_memory, RLock
from src.utils import write_on_shared_memory, extract_from_shared_memory
import os
import json

class LogManager:

    __instance = None
    __SUPPORTED_FILE_FORMATS = [".json"]

    def __init__(self, path_to_log_, log_shared_memory_name, generation_shared_memory_name, generation_counter_shared_memory_name, individual_counter_shared_memory_name):
        if LogManager.__instance != None:
            raise NotImplementedError("This is a singleton class.")
        self.path_to_log = path_to_log_
        self.__log_shared_memory = shared_memory.SharedMemory(log_shared_memory_name)
        self.__generation_shared_memory = shared_memory.SharedMemory(generation_shared_memory_name)
        self.__generation_counter_shared_memory = shared_memory.SharedMemory(generation_counter_shared_memory_name)
        self.__individual_counter_shared_memory = shared_memory.SharedMemory(individual_counter_shared_memory_name)
        self.lock = RLock()
        self.initialize_buffers()
    
    @staticmethod
    def create_manager(path_to_log, log_shared_memory_name, generation_shared_memory_name, generation_counter_shared_memory_name, individual_counter_shared_memory_name):
        if LogManager.__instance == None:
            LogManager.__instance = LogManager(path_to_log, log_shared_memory_name, generation_shared_memory_name, generation_counter_shared_memory_name, individual_counter_shared_memory_name)
        return LogManager.__instance
    
    @staticmethod
    def get_manager():
        return LogManager.__instance
    
    def initialize_buffers(self):
        """
        Initializes the shared memory buffers
        """
        write_on_shared_memory(self.__log_shared_memory, self.lock, { "individuals" : []})
        write_on_shared_memory(self.__generation_shared_memory, self.lock, list())
        write_on_shared_memory(self.__generation_counter_shared_memory, self.lock, 0)
        write_on_shared_memory(self.__individual_counter_shared_memory, self.lock, 1)
    
    def __read_log(self):
        """
        Reads and returns the content of the log file located at path_to_log.

        The way the file is handled depends on its format

        Returns
        -------
        log_data : dict
            Data structure containing the log data
        """
        log_data = dict()
        file_format = self.__file_format(self.path_to_log)
        with self.lock:
            if file_format == ".json":
                with open(self.path_to_log, 'r') as file:
                    log_data = json.load(file)
            else:
                print("The", file_format, "format is not supported. Formats currently supported are:", ", ".join(self.__SUPPORTED_FILE_FORMATS))
        return log_data
    
    def __write_log(self, log_data):
        """
        Writes data on the log file, overwriting any existing content

        Parameters
        ----------
        log_data : dict
            Data structure containing the data to log
        """
        file_format = self.__file_format(self.path_to_log)
        with self.lock:
            if file_format == ".json":
                with open(self.path_to_log, 'w') as file:
                    json.dump(log_data, file, indent=2)
            else:
                print("The", file_format, "format is not supported. Formats currently supported are:", ", ".join(self.__SUPPORTED_FILE_FORMATS))
    
    def __file_format(self, path_to_file):
        filename, file_format = os.path.splitext(path_to_file)
        return file_format
    
    def log_individual(self, parameter_names, parameter_values, fitness_value, execution_time):
        """
        Logs the parameters related to an individual

        Parameters
        ----------
        parameter_names : list
            Names of the parameters related to the individual
        
        parameter_values : list
            Values of the parameters related to the individual
        
        fitness_value : float
            Fitness value of the individual
        
        execution_time: float
            Time [in seconds] elapsed to evaluate the individual
        """
        # extract generation counter
        generation = extract_from_shared_memory(self.__generation_counter_shared_memory, self.lock)
        # extract and increase individual counter
        with self.lock:
            individual_ID = extract_from_shared_memory(self.__individual_counter_shared_memory, self.lock)
            write_on_shared_memory(self.__individual_counter_shared_memory, self.lock, individual_ID + 1)
        # create the individual to be logged
        individual = {}
        individual["generation"] = generation
        individual["individual-ID"] = individual_ID
        individual["execution-time"] = execution_time
        for name, value in zip(parameter_names, parameter_values):
            individual[name] = value
        individual["fitness-value"] = fitness_value
        # extract the list of individuals of the current generation and add the new individual
        with self.lock:
            current_generation_individuals = extract_from_shared_memory(self.__generation_shared_memory, self.lock)
            current_generation_individuals.append(individual)
            write_on_shared_memory(self.__generation_shared_memory, self.lock, current_generation_individuals)
    
    def callback(self, xk, convergence):
        with self.lock:
            # extract and reset the list of individuals of the last generation
            last_generation_individuals = extract_from_shared_memory(self.__generation_shared_memory, self.lock)
            write_on_shared_memory(self.__generation_shared_memory, self.lock, list())
            # update the log adding the individuals of the last generation
            log_data = extract_from_shared_memory(self.__log_shared_memory, self.lock)
            log_data["individuals"].extend(last_generation_individuals)
            write_on_shared_memory(self.__log_shared_memory, self.lock, log_data)
            self.__write_log(log_data)
            # increase generation counter
            generation = extract_from_shared_memory(self.__generation_counter_shared_memory, self.lock)
            write_on_shared_memory(self.__generation_counter_shared_memory, self.lock, generation + 1)
            # reset individual counter
            write_on_shared_memory(self.__individual_counter_shared_memory, self.lock, 1)
