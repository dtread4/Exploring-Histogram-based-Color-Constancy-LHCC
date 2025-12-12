# David Treadwell
# treadwell.d@northeastern.edu
# Northeastern University
# Professor Bruce Maxwell
# evaluate_gray_world.py - gets correction estimates for images used the Gray World method

import os
import cv2
import numpy as np
import sys
import pandas as pd
import torch
from constancy_model.training_utility import get_angular_error, convert_tensor_to_numpy
from constancy_model.evaluation_code.simplecube_image_correction_from_estimate import correct_image_black_level
from constancy_model.evaluation_code.test_set_evaluation import save_illuminations_to_csv


max_value_simplecube = 2**14 - 1 - 2048
max_value_8bit = 255


def convert_image_8bit(image, org_max):
    """
    Creates a copy of the image and converts to 8bit
    :param image:   The original image
    :param org_max: The max value (possible) for the original image
    :return: The 8 bit image
    """
    new_image = image.copy()
    new_image = (new_image / org_max * max_value_8bit).astype(np.uint8)
    return new_image


def get_gray_world_scales(image):
    """
    Uses the Gray World estimation to color correct the image
    :param image: The image to correct
    :return: The Gray World corrected image
    """
    # SimpeCube++ uses (2^14 - 1) max intensity images, with 2048 black point (already subtracted).
    # Scale to [0, 255] first
    image_8bit = convert_image_8bit(image, max_value_simplecube)

    # Get the average color of the image for each channel
    average_colors = np.mean(image_8bit, axis=(0, 1))

    # From the average color
    blue_scale = average_colors[0] / average_colors[1]
    green_scale = 1
    red_scale = average_colors[2] / average_colors[1]

    # Normalize the scales
    sums = blue_scale + green_scale + red_scale
    blue_scale = blue_scale / sums
    green_scale = green_scale / sums
    red_scale = red_scale / sums

    return blue_scale, green_scale, red_scale


def save_gray_world_estimates(input_dir, gt_csv_path, output_dir):
    """
    Calculates a Gray World estimate for each image, then saves it's estimated illuminations
    :param input_dir:   The input directory of images to evaluate
    :param gt_csv_path: The ground-truth csv for the dataset
    :param output_dir:  The output directory for the estimations csv
    :return: None
    """
    # Load the ground-truth csv
    gt_df = pd.read_csv(gt_csv_path)

    # Create a dataframe to store the outputs
    columns = ['image', 'gt_red', 'gt_green', 'gt_blue', 'pred_red', 'pred_green', 'pred_blue', 'angular_error']
    df_eval = pd.DataFrame(columns=columns)

    # For each image in the input directory, calculate its gray world estimated illumination values and add to csv
    for image_name in os.listdir(input_dir):
        # Get the image path
        image_path = input_dir + os.sep + image_name

        # Load the image
        image = correct_image_black_level(cv2.imread(image_path, cv2.IMREAD_UNCHANGED))

        # Get the Gray World illumination estimation
        blue_scale, green_scale, red_scale = get_gray_world_scales(image)

        # Put illuminations into a tensor
        predicted_illumination = (red_scale, green_scale, blue_scale)
        predicted_illumination_tensor = torch.unsqueeze(torch.tensor(predicted_illumination), 0)

        # Get the ground truth illumination value and put into a tensor
        gt_row = gt_df.loc[gt_df['image'] == image_name[:-4]]
        gt_illumination = (gt_row['mean_r'].item(), gt_row['mean_g'].item(), gt_row['mean_b'].item())
        gt_illumination_tensor = torch.unsqueeze(torch.tensor(gt_illumination), 0)

        # Get the angular error in degrees
        angular_error = get_angular_error(predicted_illumination_tensor, gt_illumination_tensor)
        angular_error = convert_tensor_to_numpy(angular_error)

        # Add the values to the output dataframe
        df_eval.loc[len(df_eval)] = [image_name,
                                     gt_illumination[0],
                                     gt_illumination[1],
                                     gt_illumination[2],
                                     predicted_illumination[0],
                                     predicted_illumination[1],
                                     predicted_illumination[2],
                                     angular_error[0]]

    # Save to csv
    save_illuminations_to_csv(output_dir, df_eval)


def main(argv):
    # Get the parameters from command line arguments
    input_dir = argv[1]
    gt_csv_path = argv[2]
    output_dir = argv[3]

    # Get each image's Gray World estimate and save to csv
    print("Evaluating images with a Gray World estimate... ", end='')
    save_gray_world_estimates(input_dir, gt_csv_path, output_dir)
    print("Finished!")


if __name__ == "__main__":
    """
    Command line arguments needed:
        1. Input directory of images to evaluate
        2. Ground truth CSV path
        3. Output directory
    """   
    main(sys.argv)
