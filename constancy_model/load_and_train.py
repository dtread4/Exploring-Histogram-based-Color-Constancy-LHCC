# David Treadwell
# treadwell.d@northeastern.edu
# Northeastern University
# Professor Bruce Maxwell
# load_and_train.py - Trains a CNN from an already trained model, and saves a new one
# Note that the parameters file should be the same as the one used to train the original model, except for # epochs

from constancy_model.training_utility import *
from constancy_model.parameter_manager import ParameterManager
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_and_train(parameters, model_path, output_dir, valid_fold_number, preloaded_histograms):
    """
    Trains a model from the start (epoch 0, initialize model)
    :param parameters:            The initialized parameters object to manage experiment parameters
    :param model_path:            Path to the pre-trained model
    :param output_dir:            Output directory for the model, optimizer, and loss curves
    :param valid_fold_number:     The fold number to use as the validation set
    :param preloaded_histograms:  When training > 1 cross validation fold, histograms could be preloaded if generated
        in a static way. This dictionary prevents regenerating histograms unnecessarily
    :return: None
    """
    # Set output directories
    output_dir_fold = check_and_set_directory(output_dir, "fold_{}".format(valid_fold_number))
    model_dir = check_and_set_directory(output_dir_fold, "models")
    optimizer_dir = check_and_set_directory(output_dir_fold, "optimizers")
    graph_dir = check_and_set_directory(output_dir_fold, "loss_curves")

    # Set torch determinism
    set_determinism(parameters.random_state)

    # Load the original network, optimizer, and scheduler
    network = load_trained_model(model_path, parameters, torch.cuda.is_available())
    optimizer = define_optimizer(network, parameters)
    scheduler = None  

    # Load the correct dataset
    image_dataset = select_dataset(parameters.dataset_name, valid_fold_number, parameters.image_scale_factor)

    # Get the dataloaders
    train_loader = get_dataloader(parameters=parameters, train_data=True, dataset=image_dataset,
                                  preloaded_histograms=preloaded_histograms,
                                  batch_size=parameters.batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    test_loader = get_dataloader(parameters=parameters, train_data=False, dataset=image_dataset,
                                 preloaded_histograms=preloaded_histograms,
                                 batch_size=parameters.batch_size, shuffle=True, pin_memory=torch.cuda.is_available())

    # Storage for train and test loss (for graphing)
    train_losses = []
    train_counter = []
    test_loss_metrics = {
        "mean": [],
        "median": [],
        "trimean": [],
        "best_25%": [],
        "worst_25%": [],
        "worst_10%": [],
        "worst_5%": [],
        "worst_1%": [],
        "95_percentile": [],
        "99_percentile": []
    }
    test_counter = [0]

    # Track time to complete full training
    print()
    start_time = int(round(time.time()))

    # Run a test pass on the untrained network
    test_loss_metrics = test(network, test_loader, test_loss_metrics)

    # For the number of epochs specified in parameters, train the network, with a train + test pass each epoch
    for epoch in range(1, parameters.epochs + 1):
        # Train and test pass
        train_losses, train_counter = train_epoch(network, optimizer, scheduler, train_loader, epoch, train_losses,
                                                  train_counter, parameters.batch_size, parameters.loss_function,
                                                  parameters.save_model_checkpoint, model_dir, optimizer_dir)
        test_loss_metrics = test(network, test_loader, test_loss_metrics)
        test_counter.append(train_counter[-1])

        # If on a checkpoint, also save the loss curve
        if epoch % parameters.save_model_checkpoint == 0:
            save_loss_curve(train_losses, train_counter, test_loss_metrics, test_counter, graph_dir, epoch,
                            parameters.model_title, parameters.loss_function)

    # Print time taken to train model and completion message
    end_time = int(round(time.time()))
    time_taken = end_time - start_time
    print("Finished training fold {}! Took {} seconds.\n\n".format(valid_fold_number, time_taken))

    # Add the number of parameters to the model training metrics
    param_count = sum(p.numel() for p in network.parameters() if p.requires_grad)

    # Save the minimum error and the epoch it occurred at
    save_lowest_epoch_error(test_loss_metrics, output_dir_fold, parameters.epochs, param_count)

    # Update histograms if training with static/chromaticity histograms and if more than one cross validation fold
    # Only do this if the preloaded histograms dictionary is None (not yet initialized)
    if (parameters.projection_type == "STATIC" or parameters.projection_type == "CHROMATICITY")\
            and preloaded_histograms is None:
        preloaded_histograms = set_preloaded_histograms(train_loader.dataset.preloaded_histograms,
                                                        test_loader.dataset.preloaded_histograms,
                                                        image_dataset)

    # Add parameter count to test loss metrics - MUST be done after metrics are saved for fold
    test_loss_metrics['param_count'] = param_count

    # Return preloaded histograms and test loss metrics (for tuning)
    return preloaded_histograms, test_loss_metrics


def main(argv):
    # Check that enough arguments were passed in
    if len(argv) < 4:
        print(" - Experiment JSON parameter file path")
        print(" - Pre-trained model path")
        print(" - Model output directory")

    # Get parameters
    parameter_file_path = argv[1]
    model_path = argv[2]
    output_dir = argv[3]

    # Load the parameters objects
    parameters = ParameterManager(parameter_file_path)

    # Print the command line variables being used
    print("Parameters file:        {}".format(parameter_file_path))
    print("Original model file:    {}".format(model_path))
    print("Model output directory: {}".format(output_dir))
    print()

    # Run the full training process for each fold
    preloaded_histograms = None  # Only used for three-fold cross-validation when using static histograms
    for valid_fold_number in range(parameters.num_folds):
        print("~~~~~ TRAINING ON VALIDATION FOLD {} ~~~~~".format(valid_fold_number + 1))  # + 1 so starts at 1
        preloaded_histograms, metrics = load_and_train(parameters, model_path, output_dir,
                                                       valid_fold_number + 1, preloaded_histograms)


# Main function
if __name__ == "__main__":
    """
    Command line arguments needed:
        1. Path to experiment parameters JSON file
        2. Pre-trained model path
        3. Output directory for models (will save a checkpoint every n epochs, as defined in the parameters file)
    """
    main(sys.argv)
