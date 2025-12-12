# David Treadwell
# treadwell.d@northeastern.edu
# Northeastern University
# Professor Bruce Maxwell
# image_correction_from_estimate.py - corrects an image from ground truth, LHCC, and gray world estimates

import sys
import cv2
import pandas as pd
import numpy as np
import os


max_value_simplecube = 2**14 - 1 - 2048
max_value_8bit = 255


def correct_image_black_level(image, black_level=2048, upper_bound=2**14 - 1):
    """
    Reduces an image's black level to zero
    :param image:       The image to reduce the black level of
    :param black_level: The black level to subtract away
    :param upper_bound: The upper bound of the image
    :return: The image with its black level normalized
    """
    org_image_type = image.dtype
    image = image.astype(np.int32)
    image = np.clip(image - black_level, 0, upper_bound)
    return image.astype(org_image_type)


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


def get_gray_world_image(image, org_image_name, output_dir):
    """
    Uses the Gray World estimation to color correct the image
    :param image:          The image to correct
    :param org_image_name: Original image's name
    :param output_dir:     Output directory to save image
    :return: The Gray World corrected image
    """
    # SimpeCube++ uses (2^14 - 1) max intensity images, with 2048 black point.
    # Scale to [0, 255] first
    image_8bit = convert_image_8bit(image, max_value_simplecube)

    # Get the average color of the image for each channel
    average_colors = np.mean(image_8bit, axis=(0, 1))

    # From the average color
    blue_scale = average_colors[1] / average_colors[0]
    green_scale = 1
    red_scale = average_colors[1] / average_colors[2]

    # Set up corrected image
    gray_world_image = image_8bit.copy().astype(np.float32)
    gray_world_image /= max_value_8bit
    gray_world_image[:, :, 0] = (gray_world_image[:, :, 0] * blue_scale).clip(0, 1)
    gray_world_image[:, :, 1] = (gray_world_image[:, :, 1] * green_scale).clip(0, 1)
    gray_world_image[:, :, 2] = (gray_world_image[:, :, 2] * red_scale).clip(0, 1)
    gray_world_image = (gray_world_image * max_value_8bit).astype(np.uint8)

    # Save the image
    output_path = output_dir + os.sep + "gray_world_" + org_image_name
    cv2.imwrite(output_path, gray_world_image)
    print("Finished saving Gray World corrected image!")


def get_gt_correction_value(org_image_name, ground_truth):
    """
    Gets a tuple of illumination values from the ground truth csv
    :param org_image_name: The original image's name
    :param ground_truth:   The dataframe of ground truth illumination values
    :return: Tuple of illumination values
    """
    # Get the image name
    image_name = org_image_name[-11:-4]

    # Get the ground-truth illumination
    illumination = ground_truth.loc[ground_truth['image'] == image_name]

    # Extract the illumination values into a tuple
    illumination_tuple = (illumination['mean_b'].item(),
                          illumination['mean_g'].item(),
                          illumination['mean_r'].item())
    return illumination_tuple


def get_lhcc_correction_value(org_image_name, simplecube_estimations):
    """
    Gets a tuple of illumination values from the LHCC csv
    :param org_image_name:         The original image's name
    :param simplecube_estimations: The dataframe of LHCC illumination values
    :return: Tuple of illumination values
    """
    # Get the image name
    image_name = org_image_name

    # Get estimated illumination
    illumination = simplecube_estimations.loc[simplecube_estimations['image'] == image_name]

    # Extract the illumination values into a tuple
    illumination_tuple = (illumination['pred_blue'].item(),
                          illumination['pred_green'].item(),
                          illumination['pred_red'].item())
    return illumination_tuple


def get_grayworld_correction_value(org_image_name, gray_world_estimations):
    """
    Gets a tuple of illumination values from the Gray World
    :param org_image_name:         The original image's name
    :param gray_world_estimations: The dataframe of GoogleNet illumination values
    :return: Tuple of illumination values
    """
    # Get the image name
    image_name = org_image_name

    # Get estimated illumination
    illumination = gray_world_estimations.loc[gray_world_estimations['image'] == image_name]

    # Extract the illumination values into a tuple
    illumination_tuple = (illumination['pred_blue'].item(),
                          illumination['pred_green'].item(),
                          illumination['pred_red'].item())
    return illumination_tuple


def linear_to_srgb(linear_rgb_image):
    """
    Converts a linear color value (0.0-1.0) to an sRGB color value (0.0-1.0).
    :param linear_rgb_image: The original image in linear RGB colors
    :return: The image converted to sRGB space
    """
    threshold = 0.0031308
    srgb_image = np.where(
        linear_rgb_image <= threshold,
        12.92 * linear_rgb_image,
        1.055 * np.power(linear_rgb_image, 1 / 2.4) - 0.055
    )
    return np.clip(srgb_image, 0.0, 1.0)


def apply_correction(image, org_image_name, correction_values, output_dir, correction_name):
    """
    Applies all required corrections to the input image
    :param image:             The image to correct
    :param org_image_name:    The original image name (not including path)
    :param correction_values: The values illumination values to correct with
    :param output_dir:        The output directory for the corrected image
    :param correction_name:   The name of the correction type (where illumination values are from)
    :return: None
    """
    # Get a copy of the original image and bring to 8 bit range
    corrected_image = convert_image_8bit(image, max_value_simplecube)

    # Set up corrected image
    corrected_image = (corrected_image / max_value_8bit).astype(np.float32)

    # Get max green value
    max_green = np.max(corrected_image[1])

    # Divide each color channel by its corresponding correction value
    corrected_image[:, :, 0] = (corrected_image[:, :, 0] / correction_values[0]).clip(0, 1)
    corrected_image[:, :, 1] = (corrected_image[:, :, 1] / correction_values[1]).clip(0, 1)
    corrected_image[:, :, 2] = (corrected_image[:, :, 2] / correction_values[2]).clip(0, 1)

    # Scale by max green
    corrected_image *= max_green / np.max(corrected_image)

    # Convert to sRGB
    corrected_image_srgb = linear_to_srgb(corrected_image)

    # Scale back to 8 bit and save
    corrected_image_srgb = (corrected_image_srgb * max_value_8bit).astype(np.uint8)
    output_path = output_dir + os.sep + correction_name + "_" + org_image_name
    cv2.imwrite(output_path, corrected_image_srgb)
    print("Finished saving {} corrected image!".format(correction_name))


def update_output_dir(output_dir, image_name):
    """
    Updates the output directory to create a subdirectory for the image being created
    :param output_dir: The original output directory
    :param image_name: The original image name
    :return:
    """
    new_output_dir = output_dir + os.sep + image_name
    if not os.path.isdir(new_output_dir):
        os.mkdir(new_output_dir)
    return new_output_dir


def save_corrected_images(image_path, ground_truth, simplecube_estimations, gray_world_estimations, output_dir):
    """
    Gets the various estimated image corrections
    :param image_path:              Path to original image
    :param ground_truth:            Path to ground truth csv
    :param simplecube_estimations:  Path to LHCC estimated illuminations
    :param gray_world_estimations:  Path to Gray World estimations
    :param output_dir:              Output directory to save corrected images to
    :return: None
    """
    # Load the image
    image = correct_image_black_level(cv2.imread(image_path, cv2.IMREAD_UNCHANGED))

    # Parse the image name
    image_name = image_path[-11:] 
    print("Getting corrections for {}\n".format(image_name))

    # Update output directory for the image
    output_dir = update_output_dir(output_dir, image_name)

    # Get the estimated illumination values
    gt_illumination = get_gt_correction_value(image_name, ground_truth)
    hist_illumination = get_lhcc_correction_value(image_name, simplecube_estimations)
    gray_world_illumination = get_grayworld_correction_value(image_name, gray_world_estimations)

    # Correct for each model type
    apply_correction(image, image_name, gt_illumination, output_dir, "ground_truth")
    apply_correction(image, image_name, hist_illumination, output_dir, "LHCC")
    apply_correction(image, image_name, gray_world_illumination, output_dir, "gray_world")

    # Also save the original image in the same format
    apply_correction(image, image_name, (1, 1, 1), output_dir, "original")


def main(argv):
    # Get the parameters from command line arguments
    image_path = argv[1]
    ground_truth_path = argv[2]
    lhcc_estimation_path = argv[3]
    gray_world_estimation_path = argv[4]
    output_dir = argv[5]

    # Load the csv files
    ground_truth_estimations = pd.read_csv(ground_truth_path)
    lhcc_estimations = pd.read_csv(lhcc_estimation_path)
    gray_world_estimations = pd.read_csv(gray_world_estimation_path)

    # Apply corrections
    save_corrected_images(image_path, ground_truth_estimations, lhcc_estimations, gray_world_estimations, output_dir)


if __name__ == "__main__":
    """
    Command line arguments needed:
        1. Path for image to correct
        2. Ground truth CSV path
        3. LHCC estimates CSV path
        4. Gray world estimates CSV path
        5. Output directory
    """   
    main(sys.argv)
