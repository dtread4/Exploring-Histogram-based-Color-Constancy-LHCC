# David Treadwell
# treadwell.d@northeastern.edu
# Northeastern University
# Professor Bruce Maxwell
# training_utility.py - defines functions used to train the CNN
# use 'train_from_start.py' or 'train_from_checkpoint.py' to actually train the model

import pandas as pd
import torch
from torch.utils.data import DataLoader
import time
import os
import matplotlib.pyplot as plt
import math
import numpy as np
import sys
from constancy_model.histogram_projection.histogram_loader import HistogramDataset
from constancy_model.cnn_structure import ConstancyNetwork
from constancy_model.data_classes.simplecube_image_loader import SimpleCubeLoader
from constancy_model.data_classes.nus_8_image_loader import NUS8ImageLoader
from constancy_model.data_classes.gehler_shi_reprocessed_image_loader import GehlerShiReprocessedLoader


def create_network(parameters):
    """
    Creates a network and optimizer from a set of parameters
    :param parameters: Parameters object to get experiment parameters from
    :return: The network, optimizer, and scheduler to use in training
    """
    # Initialize the network
    network = ConstancyNetwork(parameters)
    print("Network Structure\n{}\n".format(network))
    print("Network parameters: {}\n".format(sum(p.numel() for p in network.parameters() if p.requires_grad)))

    # Check if GPU available for training and send network to GPU if it is
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print("Training on GPU [{}]\n".format(device_name))
        network.cuda()
    else:
        print("Training on device [CPU]\n")

    # Define the optimizer
    optimizer = define_optimizer(network, parameters)

    # Define the scheduler
    scheduler = define_scheduler(optimizer, parameters)

    return network, optimizer, scheduler


def define_optimizer(network, parameters):
    """
    Defines the correct optimizer based on the parameters object
    :param network:    The network to optimize
    :param parameters: Parameters object to train with
    :return: The correct optimizer
    """
    # Get learning rate
    learning_rate = parameters.learning_rate

    # Create correct optimizer type based on parameters file
    if parameters.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    elif parameters.optimizer.lower() == "sgd":
        momentum = parameters.momentum
        optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    else:
        optimizer = None
    return optimizer


def define_scheduler(optimizer, parameters):
    """
    Defines a scheduler based on a parameters object
    :param optimizer:  The optimizer to schedule
    :param parameters: The parameters object
    :return: The correct scheduler
    """
    # Create correct scheduler based on parameters
    if parameters.scheduler.lower() == "none":
        scheduler = None
    elif parameters.scheduler.lower() == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=parameters.epochs)
    elif parameters.scheduler.lower() == "cosine_annealing_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
    else:
        scheduler = None
    return scheduler


def get_angular_error(output, ground_truth):
    """
    Gets the angular error between an output and a target. NOTE this does not return a scalar
    :param output:       The output (predicted) vector
    :param ground_truth: The target (actual) vector
    :return: The angular error between the two vectors
    """
    # Set both output and target to double to avoid dtype overflow issues with cosine similarity
    output = output.type(torch.double)
    ground_truth = ground_truth.type(torch.double)

    # Set up cosine similarity
    cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    # Get cosine similarity scores
    cos_sim_errors = cosine_similarity(output, ground_truth)

    # Take inverse cosine to get angular errors
    angular_errors_radians = torch.arccos(cos_sim_errors)
    angular_errors_degrees = radians_to_degrees(angular_errors_radians)
    return angular_errors_degrees


def get_euclidean_distance(output, ground_truth):
    """
    Calculates the Euclidean distance between two tensors
    :param output:       The predicted illuminations
    :param ground_truth: The ground truth illuminations
    :return: The average for the batch euclidean distance
    """
    differences = torch.sub(output, ground_truth)
    squared = torch.pow(differences, exponent=2)
    summed = torch.sum(squared, dim=1)
    rooted = torch.sqrt(summed)
    return torch.mean(rooted)


def train_epoch(network, optimizer, scheduler, train_loader, epoch, train_losses, train_counter, batch_size,
                loss_function, save_model_checkpoint, model_dir, optimizer_dir):
    """
    Trains a single epoch of a network
    :param network:               The network to train
    :param optimizer:             The optimizer to use in training
    :param scheduler:             The scheduler to use in training (could be None)
    :param train_loader:          The dataloader for the training data
    :param epoch:                 The current epoch
    :param train_losses:          The list of training losses
    :param train_counter:         The counter of training examples seen
    :param batch_size:            The number of training examples per batch
    :param loss_function:         The loss function (Euclidean distance or angular error) to use in training
    :param save_model_checkpoint: The checkpoint for number of epochs to save the model on
    :param model_dir:             The output directory for the model
    :param optimizer_dir:         The output directory for the optimizer
    :return: The updated training losses and training counter
    """
    # Print start message
    print("Starting training for epoch {}".format(epoch))

    # Set network to training mode
    network.train()

    # Track time to train epoch
    start_time = int(round(time.time()))

    # Train for each batch
    for batch_idx, (hist_set, gt_illumination) in enumerate(train_loader):
        # If training on a GPU, send data to GPU
        if torch.cuda.is_available():
            hist_set = hist_set.to(torch.cuda.current_device())
            gt_illumination = gt_illumination.to(torch.cuda.current_device())

        # Get the loss
        optimizer.zero_grad()
        output_illumination = network(hist_set)
        if loss_function == "euclidean":
            loss = get_euclidean_distance(output_illumination, gt_illumination)
        elif loss_function == "angular":
            loss = get_angular_error(output_illumination, gt_illumination).mean()
        else:
            print("ERROR: INVALID TRAINING LOSS FUNCTION! EXITING...")
            sys.exit(-1)

        # Perform backpropagation
        loss.backward()
        optimizer.step()

        # Update the training loss tracker
        train_losses.append(loss.item())
        train_counter.append((batch_idx * batch_size) + ((epoch - 1) * len(train_loader.dataset)))

    # If using a scheduler, update the scheduler (done after each epoch)
    if scheduler is not None:
        scheduler.step()

    # Save the model if the epoch is on the checkpoint interval
    if epoch % save_model_checkpoint == 0:
        torch.save(network.state_dict(), model_dir + os.sep + "model_epoch{}.pth".format(epoch))
        torch.save(optimizer.state_dict(), optimizer_dir + os.sep + "optimizer_epoch{}.pth".format(epoch))

    # Print message at the end of the epoch
    end_time = int(round(time.time()))
    time_taken = end_time - start_time
    print("Finished epoch {}, took {} seconds".format(epoch, time_taken))

    return train_losses, train_counter


def test(network, test_loader, test_loss_metrics):
    """
    Tests a network and gets the current test loss
    :param network:           The network to test with
    :param test_loader:       The dataloader for the test data
    :param test_loss_metrics: The list of average test loss after every epoch
    :return: The updated list of test losses
    """
    # Print start message
    print("Generating test evaluation... ", end='')

    # Set to evaluation mode
    network.eval()

    # Get the average loss on the test set
    test_loss_vals = None
    start_time = int(round(time.time()))
    with torch.no_grad():
        # Get total test loss over the batches
        for hist_set, gt_illumination in test_loader:
            # If training on GPU, send data to GPU
            if torch.cuda.is_available():
                hist_set = hist_set.to(torch.cuda.current_device())
                gt_illumination = gt_illumination.to(torch.cuda.current_device())

            # Get test loss for the batch
            output_illumination = network(hist_set)
            losses = get_angular_error(output_illumination, gt_illumination)
            losses = losses.cpu().numpy().flatten()

            # Add loss to overall test loss
            if test_loss_vals is None:
                test_loss_vals = losses
            else:
                test_loss_vals = np.concatenate((test_loss_vals, losses), axis=0)

        # Get the test loss metrics
        test_loss_vals = np.array(test_loss_vals)
        test_loss_metrics = append_test_loss_metrics(test_loss_metrics, test_loss_vals)

        # Get the time taken
        end_time = int(round(time.time()))
        time_taken = end_time - start_time

        # Print test metrics
        print_metric = "mean"
        print("Finished!\nTest set {} angular loss: {:.4f} | took {} seconds\n".format(
            print_metric, test_loss_metrics[print_metric][-1], time_taken))

    return test_loss_metrics


def append_test_loss_metrics(test_loss_metrics, test_loss_vals):
    """
    Appends all error metrics to the dictionary of arrays of test loss metrics
    :param test_loss_metrics: The original array to add to
    :param test_loss_vals:    The set of test loss values for this epoch
    :return: The updated dictionary of arrays of test loss metrics
    """
    # Sort test metrics before calculating
    test_loss_vals = np.sort(test_loss_vals)

    # Calculate all tests metrics
    test_loss_metrics["mean"].append(np.mean(test_loss_vals))
    test_loss_metrics["median"].append(np.median(test_loss_vals))
    test_loss_metrics["trimean"].append((0.25 * np.percentile(test_loss_vals, 25)) +
                                        (0.50 * np.percentile(test_loss_vals, 50)) +
                                        (0.25 * np.percentile(test_loss_vals, 75)))
    test_loss_metrics["best_25%"].append(np.mean(test_loss_vals[:int(0.25 * len(test_loss_vals)) + 1]))  # +1 because right index is exclusive)
    test_loss_metrics["worst_25%"].append(np.mean(test_loss_vals[int(0.75 * len(test_loss_vals)):]))
    test_loss_metrics["worst_10%"].append(np.mean(test_loss_vals[int(0.9 * len(test_loss_vals)):]))
    test_loss_metrics["worst_5%"].append(np.mean(test_loss_vals[int(0.95 * len(test_loss_vals)):]))
    test_loss_metrics["worst_1%"].append(np.mean(test_loss_vals[int(0.99 * len(test_loss_vals)):]))
    test_loss_metrics["95_percentile"].append(np.percentile(test_loss_vals, 95))
    test_loss_metrics["99_percentile"].append(np.percentile(test_loss_vals, 99))
    return test_loss_metrics


def radians_to_degrees(rad_angle):
    """
    Converts an angle in radians to an angle in degrees
    :param rad_angle: The angle to convert in radians
    :return: The converted angle in degrees
    """
    return rad_angle * 180 / math.pi


def save_loss_curve(train_losses, train_counter, test_loss_metrics, test_counter, output_dir, epoch, model_title,
                    loss_function):
    """
    Saves the loss curve for the current state of the network
    :param train_losses:      The list of training losses
    :param train_counter:     The list of training instances seen
    :param test_loss_metrics: The list of test losses
    :param test_counter:      The list of test instances seen
    :param output_dir:        The output directory for the image
    :param epoch:             The current epoch
    :param model_title:       Title of the model
    :param loss_function:     Loss function used in training
    :return: None
    """
    # Set the x-axis labels
    test_epochs = len(test_counter)
    num_ticks = 10 if test_epochs >= 10 else test_epochs
    tick_width = int(test_epochs / num_ticks)
    x_ticks = [test_counter[x] for x in range(0, test_epochs, tick_width)]
    x_labels = [x for x in range(0, test_epochs, tick_width)]

    # Set colors
    euclidean_color = 'tab:blue'
    angular_color = 'tab:orange'

    # Create plot
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Set loss function for training
    if loss_function == "euclidean":
        train_loss_label = "Euclidean distance"
    elif loss_function == "angular":
        train_loss_label = "Angular error"
    else:
        print("ERROR: INVALID TRAINING LOSS FUNCTION! Exiting...")
        sys.exit(-1)

    # Plot train loss
    ax1.plot(train_counter, train_losses, color=euclidean_color, label='Train Loss')
    ax1.set_ylabel(f'{train_loss_label} loss', color=euclidean_color)
    ax1.set_xlabel('# epochs')
    ax1.set_xticks(x_ticks, x_labels)
    ax1.set_title("{} | test loss at epoch {}: {:.4f}".format(model_title, epoch, test_loss_metrics['mean'][-1]))

    # Plot test loss
    ax2.scatter(test_counter, test_loss_metrics['mean'], color=angular_color, s=10, label='Test Loss')
    ax2.set_ylabel('Angular error in degrees', color=angular_color)

    # Create the legend
    fig.legend(['Train Loss', 'Test Loss'], loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    # Save plot
    plt.savefig(output_dir + os.sep + "loss_curve_epoch{}.png".format(epoch))
    plt.clf()
    plt.close()


def get_dataloader(parameters, dataset, preloaded_histograms, train_data, batch_size, shuffle, pin_memory):
    """
    Loads a single dataloader object for a directory
    :param parameters:           Parameters object
    :param dataset:              The dataset object to use
    :param preloaded_histograms: A set of preloaded histograms
        only used for cross-fold validation with static projections
    :param train_data:           True if training data, False if validation data
    :param batch_size:           Batch size for loading data
    :param shuffle:              Whether to shuffle
    :param pin_memory:           Whether to pin memory (CUDA)
    :return: The loaded dataloader object
    """
    histogram_loader = HistogramDataset(parameters, train_data, dataset, preloaded_histograms)
    dataloader = DataLoader(histogram_loader, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
    return dataloader


def set_determinism(random_seed):
    """
    Sets determinism
    :param: The random seed to use
    :return: None
    """
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


def load_trained_model(model_path, parameters, cuda):
    """
    Loads an already-trained network
    :param model_path: The path to the trained model
    :param parameters: The parameters used to train the model/continue training the model
    :param cuda:       Whether to send the network to GPU
    :return: The loaded model
    """
    # Load the network
    trained_model = ConstancyNetwork(parameters)
    model_state_dict = torch.load(model_path)
    trained_model.load_state_dict(model_state_dict)

    # Print information about loaded model
    print("Network Structure\n{}\n".format(trained_model))
    print("Network parameters: {}\n".format(sum(p.numel() for p in trained_model.parameters() if p.requires_grad)))

    # Send the network to GPU if desired
    if cuda:
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print("Training on GPU [{}]\n".format(device_name))
        trained_model.cuda()
    else:
        print("Training on device [CPU]\n")
    return trained_model


def load_existing_optimizer(optimizer_path, network, parameters):
    """
    Loads an existing optimizer
    :param optimizer_path: The path to the trained optimizer
    :param network:        The trained network associated with the optimizer
    :param parameters:     Parameters file to load optimizer with
    :return: The loaded optimizer
    """
    # Define the optimizer
    optimizer = define_optimizer(network, parameters)

    # Load the optimizer
    optimizer_state_dict = torch.load(optimizer_path)
    optimizer.load_state_dict(optimizer_state_dict)
    return optimizer


def save_lowest_epoch_error(test_loss_metrics, output_dir, epochs, param_count):
    """
    Used after training to identify the epoch where the lowest test error occurred, and to save last epoch's results
    :param test_loss_metrics: The list of test losses
    :param output_dir:        The output directory to save the file to
    :param epochs:            The number of epochs the model was trained for
    :param param_count:       The number of epochs the model was trained for
    :return: None
    """
    # Write header
    output_path = output_dir + os.sep + "min_epoch_error.txt"
    out_file = open(output_path, 'w')
    out_file.write("Minimum error metrics: median, mean, worst 5%\n")
    out_file.close()

    # Only want to look at minimum epochs for mean, median, and worst 5%
    select_metrics = ["median", "mean", "worst_5%"]

    # Get the minimum error and epoch for each metric and append to file
    min_epochs = []
    for metric in select_metrics:
        # Write value to the txt file
        test_metric_np = np.array(test_loss_metrics[metric])
        min_epoch = np.argmin(test_metric_np)
        min_error = test_metric_np[min_epoch]
        out_file = open(output_path, 'a')
        out_file.write("Minimum {} error: {:.4} from epoch: {}\n".format(metric, min_error, min_epoch))
        out_file.close()

        # Write all metrics at this epoch to a csv file
        save_epoch_metrics_to_csv(test_loss_metrics, min_epoch, output_dir, epoch_best_metric=f"best_{metric}")

        # Add the epoch to the set of minimum epochs (to clean up files later)
        min_epochs.append(min_epoch)

    # Also save last epoch's metrics to a CSV (put mean error in min_epoch_error.txt file)
    last_epoch = epochs - 1
    last_epoch_mean_error = test_loss_metrics['mean'][last_epoch]
    out_file = open(output_path, 'a')
    out_file.write("\nMean error: {:.4} from final epoch: {}\n".format(last_epoch_mean_error, last_epoch))
    out_file.close()
    save_epoch_metrics_to_csv(test_loss_metrics, epoch=last_epoch, output_dir=output_dir, epoch_best_metric='last_epoch')

    # Delete all model files not containing a minimum epoch
    # Subtract one from the length so the last model from the last epoch is kept
    for i in range(epochs - 1):
        epoch = i + 1
        if epoch not in min_epochs:
            model_path = output_dir + os.sep + "models" + os.sep + "model_epoch{}.pth".format(epoch)
            if os.path.exists(model_path):
                os.remove(model_path)
            optimizer_path = output_dir + os.sep + "optimizers" + os.sep + "optimizer_epoch{}.pth".format(epoch)
            if os.path.exists(optimizer_path):
                os.remove(optimizer_path)
            graph_path = output_dir + os.sep + "loss_curves" + os.sep + "loss_curve_epoch{}.png".format(epoch)
            if os.path.exists(graph_path):
                os.remove(graph_path)

    # Write number of parameters to bottom of file
    out_file = open(output_path, 'a')
    out_file.write("\nNumber of trainable model parameters: [{}]".format(param_count))
    out_file.close()


def save_epoch_metrics_to_csv(test_loss_metrics, epoch, output_dir, epoch_best_metric):
    """
    Saves the metrics at a specific epoch to a csv file
    :param test_loss_metrics: The full set of test metrics
    :param epoch:             The epoch to save the values from
    :param output_dir:        The output directory to save to
    :param epoch_best_metric: The metric being saved
    :return: None
    """
    epoch_metrics = []
    for key in test_loss_metrics.keys():
        epoch_metrics.append(round(test_loss_metrics[key][epoch], 4))
    df = pd.DataFrame([epoch_metrics], columns=test_loss_metrics.keys())
    df.to_csv(output_dir + os.sep + "{}_epoch_metrics.csv".format(epoch_best_metric), index=False)


def convert_tensor_to_numpy(tensor_value):
    """
    Converts a tensor to numpy (and accounts for CPU/GPU)
    :param tensor_value: The tensor value to convert
    :return: The value as a numpy array
    """
    if torch.cuda.is_available():
        return tensor_value.cpu().numpy()
    else:
        return tensor_value.numpy()


def check_and_set_directory(parent_dir, child_dir):
    """
    Checks if a directory exists from parent and child, and creates it if not
    :param parent_dir: The parent directory
    :param child_dir:  The child directory name (not full path) within the parent directory
    :return: The full path to the child directory through the parent directory
    """
    # Get full directory path
    full_dir = parent_dir + os.sep + child_dir

    # Check if directory exists, and if not, create it
    if not os.path.isdir(full_dir):
        os.mkdir(full_dir)

    # Return the full directory name
    return full_dir


def select_dataset(dataset_name, validation_fold_num, image_scale_factor):
    """
    Returns the correct dataset object to load images with
    Ensure that the path to data is in the same directory
    :param dataset_name:        The name of the dataset to select the dataset with
    :param validation_fold_num: The validation fold number to set train/valid data within the dataset
    :param image_scale_factor:  Factor to scale images by
    :return: A dataset object to load images with
    """
    if dataset_name == "SimpleCube++":
        return SimpleCubeLoader(validation_fold_num, image_scale_factor)
    if dataset_name[:5] == "NUS-8":
        return NUS8ImageLoader(dataset_name[6:], validation_fold_num, image_scale_factor)
    if dataset_name == "Gehler-Shi Reprocessed":
        return GehlerShiReprocessedLoader(validation_fold_num, image_scale_factor)
    else:
        print("Invalid dataset name <{}>! Please only select from the list below.".format(dataset_name))
        print("~~~ AVAILABLE DATASETS ~~~")
        print("Gehler-Shi Reprocessed")
        print("NUS-8_<camera name> (insert the actual camera name between the <>)")
        print("SimpleCube++")
        print("\nExiting")
        sys.exit(-1)


def set_preloaded_histograms(train_histograms, valid_histograms, image_dataset):
    # Update preloaded histograms type to dictionary
    preloaded_histograms = {}

    # Set histograms from train set
    for i in range(len(image_dataset.train_gt_df)):
        preloaded_histograms[image_dataset.train_gt_df.iloc[i][0]] = train_histograms[i]

    # Set histograms from valid set
    for i in range(len(image_dataset.valid_gt_df)):
        preloaded_histograms[image_dataset.valid_gt_df.iloc[i][0]] = valid_histograms[i]

    return preloaded_histograms