# David Treadwell
# treadwell.d@northeastern.edu
# Northeastern University
# Professor Bruce Maxwell
# test_set_evaluation.py - Produces a csv of the output errors and illuminations (gt and predicted) for a dataset

import sys
import os
import torch
import pandas as pd
from constancy_model.training_utility import get_dataloader, get_angular_error, load_trained_model, \
    convert_tensor_to_numpy
from constancy_model.parameter_manager import ParameterManager
from constancy_model.training_utility import select_dataset
import numpy as np


def evaluate_all_images(model_path, parameter_path, valid_fold_number, output_dir):
    """
    Runs an evaluation for all images in the input directory and saves a csv of results to the output directory
    :param model_path:           The path to the model to evaluate with
    :param parameter_path:       The path to the parameter file used to create the model
    :param valid_fold_number:    The fold used as the validation fold
    :param output_dir:           The output directory for the csv path
    :return: None
    """
    # Load the parameters object
    parameters = ParameterManager(parameter_path)

    # Load the model
    model = load_trained_model(model_path, parameters, torch.cuda.is_available())
    model.eval()

    # Load the correct dataset
    image_dataset = select_dataset(parameters.dataset_name, valid_fold_number, parameters.image_scale_factor)

    # Get list of images in directory
    image_names = image_dataset.get_valid_image_names()

    # Initialize dataloader
    parameters.batch_size = 1  # Set specifically for the dataloader so only one hist set loaded
    dataloader = get_dataloader(parameters=parameters, train_data=False, dataset=image_dataset,
                                preloaded_histograms=None,
                                batch_size=parameters.batch_size, shuffle=False,
                                pin_memory=torch.cuda.is_available())

    # Create a dataframe to store the outputs
    columns = ['image', 'gt_red', 'gt_green', 'gt_blue', 'pred_red', 'pred_green', 'pred_blue', 'angular_error']
    df_eval = pd.DataFrame(columns=columns)

    # For each image in the image set, get the actual and predicted illumination values, and the angular error
    with torch.no_grad():
        for i, (hist_set, gt_illumination) in enumerate(dataloader):
            # Get predictions and error
            predicted_illumination, angular_error = evaluate_input(hist_set, gt_illumination, model)
            predicted_illumination = predicted_illumination / predicted_illumination.sum()

            # Get the current image name for the histogram set
            image_name = image_names[i]

            # Convert gt illumination, pred illumination, and error to numpy
            gt_illumination = convert_tensor_to_numpy(gt_illumination)
            predicted_illumination = convert_tensor_to_numpy(predicted_illumination)
            angular_error = convert_tensor_to_numpy(angular_error)

            # Add all info for image to dataframe
            df_eval.loc[len(df_eval)] = [image_name,
                                         gt_illumination[0][0],
                                         gt_illumination[0][1],
                                         gt_illumination[0][2],
                                         predicted_illumination[0][0],
                                         predicted_illumination[0][1],
                                         predicted_illumination[0][2],
                                         angular_error[0]]

    # Save the dataframe to a csv in the output folder
    save_illuminations_to_csv(output_dir, df_eval)

    # Sort by error metrics
    df_eval.sort_values(by='angular_error', inplace=True)

    # Create aggregate error metrics
    save_aggregate_metrics(df_eval, output_dir)


def save_aggregate_metrics(df_eval, output_dir):
    """
    Computes the aggregated error metrics necessary to evaluate fully
    :param df_eval:    The DataFrame of angular errors
    :param output_dir: The output directory to save values to
    :return: None
    """
    # Create a dictionary for each error metric
    # For best/worst x%, always round up to include one extra number
    angular_error_values = df_eval['angular_error']
    aggregate_error_metrics = {
        'mean':          np.mean(angular_error_values),
        'median':        np.median(angular_error_values),
        'trimean':       ((0.25 * np.percentile(angular_error_values, 25)) +
                         (0.50 * np.percentile(angular_error_values, 50)) +
                         (0.25 * np.percentile(angular_error_values, 75))),
        'best_25%':      np.mean(angular_error_values[:int(0.25 * len(angular_error_values)) + 1]),  # +1 because right index is exclusive
        'worst_25%':     np.mean(angular_error_values[int(0.75 * len(angular_error_values)):]),
        'worst_10%':     np.mean(angular_error_values[int(0.9 * len(angular_error_values)):]),
        'worst_5%':      np.mean(angular_error_values[int(0.95 * len(angular_error_values)):]),
        'worst_1%':      np.mean(angular_error_values[int(0.99 * len(angular_error_values)):]),
        '95_percentile': np.percentile(angular_error_values, 95),
        '99_percentile': np.percentile(angular_error_values, 99)
    }

    # Create DataFrame with the error metrics and save to csv in the output directory
    df_aggregate_metrics = pd.DataFrame([aggregate_error_metrics])
    df_aggregate_metrics.to_csv(output_dir + os.sep + "aggregated_metrics.csv", index=False)


def evaluate_input(input_set, gt_illumination, model):
    """
    Get predicted illumination and error for a set of histograms
    :param input_set:       The input for a single image (could be histograms or the image)
    :param gt_illumination: The ground-truth to use in calculating error
    :param model:           The model to get predictions with
    :return: The predicted illumination values, the angular error
    """
    # If training on GPU, send data to GPU
    if torch.cuda.is_available():
        input_set = input_set.to(torch.cuda.current_device())
        gt_illumination = gt_illumination.to(torch.cuda.current_device())

    # Get the predictions for the histogram set
    predicted_illumination = model(input_set)

    # Get the angular error
    angular_error = get_angular_error(predicted_illumination, gt_illumination)
    return predicted_illumination, angular_error


def save_illuminations_to_csv(output_dir, csv_df):
    """
    Saves a dataframe to a csv file (specifically for illumination estimates and their error)
    :param output_dir: The directory to save csv to
    :param csv_df: The dataframe to save to csv
    :return: None
    """
    # Check if csv of output illuminations exists and delete if it does
    csv_name = 'illumination_estimations.csv'
    csv_path = output_dir + os.sep + csv_name
    if os.path.isfile(csv_path):
        os.remove(csv_path)

    # Save to csv
    csv_df.to_csv(csv_path, index=False, header=True)


def main(argv):
    # Get variables for command line arguments
    model_path = argv[1]
    model_parameters = argv[2]
    valid_fold_number = int(argv[3])
    output_dir = argv[4]

    # Run calculation process
    print("Generating evaluation for all images in input directory... ")
    evaluate_all_images(model_path, model_parameters, valid_fold_number, output_dir)
    print("Finished!")


if __name__ == "__main__":
    """
    Command line arguments needed:
        1. Path to model to evaluate with
        2. Parameter file (model parameters should be the same as the ones used to train the model)
        3. Which validation fold to evaluate with (use 1 for SimpleCube++)
        4. Output directory
    """   
    main(sys.argv)

