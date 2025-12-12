# David Treadwell
# treadwell.d@northeastern.edu
# Northeastern University
# Professor Bruce Maxwell
# learning_tuning.py - Compares different ways of handling the leaning parameters when training the CNN


import os
import copy
import sys
from constancy_model.parameter_manager import ParameterManager
from constancy_model.learning_tuning_parameters_manager import LearningParametersManager
from constancy_model.train_from_start import train_from_start, check_and_set_directory
import numpy as np


def run_learning_tuning(output_dir, default_parameters, learning_tuning_parameters, valid_fold_number=1):
    # Get counter for tuning number
    trial_number = 1
    best_trial = 1

    # Track best values
    # Pre-concatenation parameters
    best_scheduler = default_parameters.scheduler
    best_learning_rate = default_parameters.learning_rate
    best_optimizer = default_parameters.optimizer
    best_momentum = default_parameters.momentum
    best_batch_size = default_parameters.batch_size

    # Set current test loss to high number, so it is beaten quickly
    best_loss = 10000

    # Make copy of parameters object
    parameters_tuning = copy.deepcopy(default_parameters)

    # Create blank histograms var for future data
    preloaded_histograms = None

    # learning parameters tuning
    for scheduler in learning_tuning_parameters.schedulers:
        for learning_rate in learning_tuning_parameters.learning_rates:
            for optimizer in learning_tuning_parameters.optimizers:

                # Don't need to run additional tests on momentum if the optimizer is adam and not sgd
                if optimizer == "adam":
                    momentum_vals = [1.0]
                else:
                    momentum_vals = learning_tuning_parameters.momentums

                for momentum in momentum_vals:
                    for batch_size in learning_tuning_parameters.batch_sizes:

                        # Starting message
                        print("~~~~~    Starting trial {}    ~~~~~".format(trial_number))
                        print("Scheduler: {}, learning rate: {}, optimizer: {}, momentum: {}, batch size: {}\n".format(
                            scheduler, learning_rate, optimizer, momentum, batch_size))

                        # Set output directory
                        child_dir = "tune_{}".format(trial_number)
                        full_output_dir = check_and_set_directory(output_dir, child_dir)

                        # Set the parameters
                        parameters_tuning.scheduler = scheduler
                        parameters_tuning.learning_rate = learning_rate
                        parameters_tuning.optimizer = optimizer
                        parameters_tuning.momentum = momentum
                        parameters_tuning.batch_size = batch_size

                        # Run the full training
                        preloaded_histograms, curr_loss_metrics = train_from_start(parameters_tuning,
                                                                                   full_output_dir,
                                                                                   valid_fold_number,
                                                                                   preloaded_histograms)

                        # Get the best loss (lowest mean loss across all epochs) and the number of parameters at that epoch
                        curr_loss = np.min(curr_loss_metrics['mean'])

                        # If current loss is less than the current minimum loss, update the best parameters
                        if curr_loss < best_loss:
                            best_loss = curr_loss
                            best_trial = trial_number
                            best_scheduler = scheduler
                            best_learning_rate = learning_rate
                            best_optimizer = optimizer
                            best_momentum = momentum
                            best_batch_size = batch_size

                        # Save a text file with the best parameters currently
                        write_params_to_file("best", full_output_dir, best_trial, best_loss,
                                             best_scheduler, best_learning_rate, best_optimizer,
                                             best_momentum, best_batch_size)

                        # Save a text file with the current epoch's parameters
                        write_params_to_file("curr", full_output_dir, trial_number, curr_loss,
                                             scheduler, learning_rate, optimizer, momentum,
                                             batch_size)

                        # Concluding message
                        print("~~~~~ Finished trial {} ~~~~~\n\n\n".format(trial_number))

                        # Increment trial number
                        trial_number += 1

                        # Reset parameters
                        parameters_tuning = copy.deepcopy(default_parameters)

    # Save a text file with the best parameters currently
    write_params_to_file("best", output_dir, best_trial, best_loss,
                         best_scheduler, best_learning_rate, best_optimizer,
                         best_momentum, best_batch_size)


def write_params_to_file(identifier, full_output_dir, trial_number, mean_loss,
                         scheduler, learning_rate, optimizer, momentum, batch_size):
    # Save a text file with the parameters passed in
    out_file = open(full_output_dir + os.sep + "{}_{}".format(identifier, trial_number) + ".txt", "w")
    out_file.write("Parameters from trial {}\n".format(trial_number))
    out_file.write("Mean loss for trial {}\n".format(round(mean_loss, 4)))
    out_file.write("{} scheduler: {}\n".format(identifier, scheduler))
    out_file.write("{} learning rate: {}\n".format(identifier, learning_rate))
    out_file.write("{} optimizer: {}\n".format(identifier, optimizer))
    out_file.write("{} momentum: {}\n".format(identifier, momentum))
    out_file.write("{} batch_size: {}\n".format(identifier, batch_size))

    out_file.close()


def main(argv):
    # Check that enough arguments were passed in
    if len(argv) < 4:
        print("ERROR - not enough arguments passed in via command line")
        print(" - Model output directory")
        print(" - Starting parameters file path")
        print(" - Learning tuning parameter file path")
        print(" - Validation fold number")
        
    output_dir = argv[1]
    parameter_file_path = argv[2]
    learning_tuning_file_path = argv[3]
    valid_fold_number = int(argv[4]) if len(argv) == 5 else 1  # Set valid fold number to 1 if no number passed in

    # Print the command line variables being used
    print("Model output directory:        {}".format(output_dir))
    print("Starting parameters file path: {}".format(parameter_file_path))
    print("Learning tuning file path:     {}".format(learning_tuning_file_path))
    print("Validation fold number:        {}".format(learning_tuning_file_path))

    # Load the parameters objects
    parameters = ParameterManager(parameter_file_path)
    learning_tuning_file_path = LearningParametersManager(learning_tuning_file_path)

    # Run full learning parameter tuning
    run_learning_tuning(output_dir, parameters, learning_tuning_file_path, valid_fold_number)


# Main function
if __name__ == "__main__":
    """
    Command line arguments needed:
        1. Output directory for models (will save a checkpoint every n epochs, as defined in the parameters file)
        2. Path to starting parameters JSON file
        3. Path to learning tuning parameters JSON file
        4. Validation fold number (use 1 for SimpleCube++)
    """
    main(sys.argv)
