# David Treadwell
# treadwell.d@northeastern.edu
# Northeastern University
# Professor Bruce Maxwell
# learning_tuning_parameters_manager.py - manages the parameters for the learning tuning


from constancy_model.parameter_manager import load_json


class LearningParametersManager:
    """
    Class to manage parameters for learning tuning
    """
    def __init__(self, path_learning_params_json):
        """
        Initialization to read the JSON and set parameters
        :param path_learning_params_json: Path to the JSON file to read parameters from
        """
        # Load the JSON file
        parameter_data = load_json(path_learning_params_json)

        # Set the set of structural variables to test
        self.schedulers = parameter_data['schedulers']
        self.learning_rates = parameter_data['learning_rates']
        self.optimizers = parameter_data['optimizers']
        self.momentums = parameter_data['momentums']
        self.batch_sizes = parameter_data['batch_sizes']
