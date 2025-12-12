# David Treadwell
# treadwell.d@northeastern.edu
# Northeastern University
# Professor Bruce Maxwell
# utility.py - Utility functions and variables to help generate histogram projections

import matplotlib.pyplot as plt
import numpy as np
import sys


MAX_8BIT_INT = 2**8 - 1
MAX_16BIT_INT = 2**16 - 1
BLUE_CHANNEL = 0
GREEN_CHANNEL = 1
RED_CHANNEL = 2
ALL_CHANNELS = np.array([BLUE_CHANNEL, GREEN_CHANNEL, RED_CHANNEL])
VALID_ORIENTATIONS = ["None", "RG", "RB", "GB", "111"]

LIN_SLICE_VECTOR=np.array([-0.703437280095144, -0.003069359672173644, 0.7107507101674599])
LIN_Z_VECTOR=np.array([0.5819267888798925, 0.5716693006736312, 0.5784076616464019])
LIN_SLICE_PORTIONS = np.array([0.47,0.03,0.03,0.47])

LOG_SLICE_VECTOR=np.array([0.7253724864398704, -0.14188668529759052, -0.6735747356094219])
LOG_Z_VECTOR=np.array([0.6196670892470691, 0.5606968726870574, 0.5492100831767557])
LOG_SLICE_PORTIONS = np.array([0.45,0.025,0.025,0.5])


def convert_to_log(image):
    """
    Converts an image to log
    :param image: The image to convert
    :return: The log converted image
    """
    # Ensure image is of type float
    log_image = image.copy().astype(np.float32)

    # Clip values below 1 to 1 so that the minimum value after calculating log is 0
    log_image = np.clip(log_image, 1, np.max(log_image))

    # Take log of image where it is not 0
    log_image = np.log(log_image)
    return log_image


def is_valid_orientation(orientation):
    """
    Checks if an orientation is in the list of valid orientations
    :param orientation:
    :return:
    """
    return orientation in VALID_ORIENTATIONS


def get_1d_from_3d(hist, keep_channel):
    """
    Gets the 1D (one channel) histogram from a 3d histogram
    :param hist:         The histogram to get the 1d histogram of
    :param keep_channel: The channel to keep
    :return: The 1d histogram
    """
    remove_channels = np.delete(ALL_CHANNELS, keep_channel)
    remove_channels = (remove_channels[0], remove_channels[1])
    hist_1d = np.sum(hist, remove_channels)
    return hist_1d


def plot_histogram_no_projection(hist, buckets, rgb_lb, rgb_ub, log=False):
    """
    Plots three-channel non-projected histograms
    :param hist:    The histogram to plot
    :param buckets: Number of buckets in the histogram
    :param rgb_lb:  The lower bound (in terms of rgb, will be handled to log if necessary)
    :param rgb_ub:  The upper bound (in terms of rgb, will be handled to log if necessary)
    :param log:     Whether to modify formatting for log case
    :return: None
    """
    # Get the single-channel histograms
    hist_b = get_1d_from_3d(hist, BLUE_CHANNEL)
    hist_g = get_1d_from_3d(hist, GREEN_CHANNEL)
    hist_r = get_1d_from_3d(hist, RED_CHANNEL)

    # Set up to plot overall histogram and each histogram's dimensions
    fig, ax = plt.subplots(nrows=2, ncols=2)

    # Plot blue
    ax[0, 0].plot(hist_b, color='b')
    ax[0, 1].plot(hist_b, color='b')

    # Plot green
    ax[0, 0].plot(hist_g, color='g')
    ax[1, 0].plot(hist_g, color='g')

    # Plot green
    ax[0, 0].plot(hist_r, color='r')
    ax[1, 1].plot(hist_r, color='r')

    # Set title and x labels depending on if log charts or not
    if log:
        log_adjust = "log "
        x_labels = [round(np.log(x - 1) if x != 0 else x, 2)
                    for x in range(rgb_lb, rgb_ub + 1, int(rgb_ub / 4))]
    # If not log, working with default RGB image
    else:
        log_adjust = ''
        x_labels = [(x - 1 if x != 0 else x) for x in range(rgb_lb, rgb_ub + 1, int(rgb_ub / 4))]

    # Add x-tick labels to each subplot
    for i, axis in enumerate(ax.flat):
        axis.set_xticks([x for x in range(0, buckets + 1, int(buckets / 4))])
        axis.set_xticklabels(x_labels)

    # Overall chart formatting
    fig.legend(['{}blue'.format(log_adjust), '{}green'.format(log_adjust), '{}red'.format(log_adjust)])
    fig.suptitle("Histograms for {}RGB image ({} buckets)".format(log_adjust, buckets))
    fig.tight_layout()
    plt.show()


def plot_2d_histogram(hist_2d, axis_1_title, axis_2_title, title):
    """
    Plots a 2D histogram
    :param hist_2d:      The histogram to plot
    :param axis_1_title: Title of the x-axis
    :param axis_2_title: Title of the y-axis
    :param title:        Title of the overall plot
    :return: None
    """
    plt.imshow(hist_2d)
    plt.xlabel(axis_1_title)
    plt.ylabel(axis_2_title)
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()


def process_histogram_range(hist_range):
    """
    Processes a string containing a range of numbers to get the upper and lower bound
    :param hist_range: string containing the range
    :return: The upper and lower bounds of the range, with the proper type
    """
    # Delimit upper and lower bound
    delimiter = ','
    split_range = hist_range.split(delimiter)
    rgb_lb = convert_string_to_number(split_range[0])
    rgb_ub = convert_string_to_number(split_range[1])
    return rgb_lb, rgb_ub


def get_log_bounds(blue_lb, green_lb, red_lb, rgb_ub):
    """
    Converts an rgb lower and upper bound to log
    :param blue_lb: The lower blue channel bound to convert
    :param green_lb: The lower green channel bound to convert
    :param red_lb: The lower red channel bound to convert
    :param rgb_ub: The upper bound to convert
    :return: The pair of converted bounds
    """
    # Convert to log
    log_b_lb = np.log(blue_lb) if blue_lb != 0 else 0
    log_g_lb = np.log(green_lb) if green_lb != 0 else 0
    log_r_lb = np.log(red_lb) if red_lb != 0 else 0
    log_ub = np.log(rgb_ub) if rgb_ub != 0 else 0
    return log_b_lb, log_g_lb, log_r_lb, log_ub


def convert_string_to_number(string_num):
    """
    Converts a number to an integer or a float, depending on if it is an integer or decimal
    :param string_num: The string to convert to a number
    :return: The converted number with the proper type
    """
    try:
        converted = int(string_num)
    except:
        converted = float(string_num)
    return converted


def get_file_name_without_file_type(file):
    """
    Checks how long the file type is and returns the file name without it
    :param file: The file to get the name of
    :return: The file name without the file type
    """
    i = 0
    while file[-i] != '.':
        i += 1
    file_name = file[:-i]
    return file_name


def pad_array(arr, n):
    """
    Pads an array with the outermost values to ensure its dimensions to ensure it is divisible by a value n
    :param arr: The array to pad
    :param n  : The value the array needs to be divisible by
    :return: The padded array
    """
    h, w, c = arr.shape
    pad_h = (n - h % n) % n
    pad_w = (n - w % n) % n
    padded_arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    return padded_arr


def sample_from_squares(arr, n):
    """
    Takes a random sample from each non-overlapping nxn square within the array.
    :param arr:           The original input array
    :param n  :           The size of the input squares to sample from
    :return: An array of the samples
    """
    # Pad array to be divisible by n
    arr = pad_array(arr, n)
    h, w, c = arr.shape

    # Extract all the samples from the padded array
    samples = []
    for i in range(0, h, n):
        for j in range(0, w, n):
            block = arr[i:i + n, j:j + n, :]
            sample = block[np.random.randint(0, n), np.random.randint(0, n), :]
            samples.append(sample)

    sample_array = np.array(samples).reshape(h // n, w // n, c)
    return sample_array

def get_projection_vector_and_slice_portion(color_space):
    '''
    Takes a color_space and return the projection vectors and slicing portions accordingly
    :param color_space:           Log/Lin
    :return: z_vector, slice_vector, slice_portion
    '''
    if color_space == "LOG":
        z_vector = LOG_Z_VECTOR
        slice_vector = LOG_SLICE_VECTOR
        slice_portions = LOG_SLICE_PORTIONS
    elif color_space == "LINEAR":
        z_vector = LIN_Z_VECTOR
        slice_vector = LIN_SLICE_VECTOR
        slice_portions = LIN_SLICE_PORTIONS
    else:
        print("ERROR: invalid color space. Use only \"LINEAR\" or \"LOG\". Exiting...")
        sys.exit(-1)

    return z_vector, slice_vector, slice_portions