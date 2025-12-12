# David Treadwell
# treadwell.d@northeastern.edu
# Northeastern University
# Professor Bruce Maxwell
# histogram_projections.py - functions to create a 3D histogram for an image and project it into a target orientation

import cv2
from constancy_model.histogram_projection.histogram_projection_utility import *
import numpy as np
import sys
import time
from itertools import product

def get_hist_upper_bound(max_image_value):
    """
    Uses the size of the maximum image value to determine how big the upper bound of the histogram should be,
    implicitly determining if stretching is needed.
    8-bit maximum unsigned integer if 1 < value <= 8-bit, 16-bit maximum unsigned integer if 8-bit < value < 16-bit max
    or if the image value is 1, because images stored as floats between [0, 1] should use 16 bits
    :param max_image_value: The maximum value possible for the image
    :return: The upper bound of the histogram range
    """
    # Upper bound has 1 added because OpenCV calcHist uses exclusive upper bound
    if 1 < max_image_value <= MAX_8BIT_INT:
        return MAX_8BIT_INT + 1
    elif MAX_8BIT_INT < max_image_value <= MAX_16BIT_INT or max_image_value == 1:
        return MAX_16BIT_INT + 1
    else:
        print("ERROR: Invalid maximum image value! Should be between 1 and {}".format(MAX_16BIT_INT))
        sys.exit(-1)


def scale_to_upper_bound(image, max_image_value, hist_ub):
    """
    Normalizes image to [0, 1] via max value, then scales to the proper depth
    :param image:           The image to normalize and scale
    :param max_image_value: The maximum value to normalize the image with
    :param hist_ub:         Histogram upper bound to scale to (EXCLUSIVE)
    :return: The image scaled to the upper bound
    """
    image_upper_bound = hist_ub - 1  # upper bound will be exclusive, and therefore one bigger than actual image
    normalized_image = np.clip(image.astype(np.float32) / max_image_value, 0, 1)
    scaled_image = normalized_image * image_upper_bound
    return scaled_image


def get_projected_histogram(histogram_3d, image, blue_lb, green_lb, red_lb, hist_ub, buckets, orientation):
    """
    Function to generate the proper projection based on the orientation
    :param histogram_3d: The 3D image of the histogram
    :param image:        The image to project the histogram of
    :param blue_lb:      Histogram range lower bound for blue channel
    :param green_lb:     Histogram range lower bound for green channel
    :param red_lb:       Histogram range lower bound for red channel
    :param hist_ub:      The upper bound of the histogram value range
    :param buckets:      The number of buckets for each channel in the histogram
    :param orientation:  The orientation to project the histogram onto
    :return: The projected histogram
    """
    # Generate the projection based on the orientation passed in
    if orientation == "None":
        return histogram_3d
    elif orientation == "RG":
        return get_two_channel_histogram_projection(histogram_3d, RED_CHANNEL, GREEN_CHANNEL)
    elif orientation == "RB":
        return get_two_channel_histogram_projection(histogram_3d, RED_CHANNEL, BLUE_CHANNEL)
    elif orientation == "GB":
        return get_two_channel_histogram_projection(histogram_3d, GREEN_CHANNEL, BLUE_CHANNEL)
    elif orientation == "111":
        # Define the origin and plane
        plane = np.array([1, 1, 1])
        origin = np.array([0, 0, 0])
        return get_plane_projected_histogram(image, plane, origin, blue_lb, green_lb, red_lb, hist_ub, buckets)

    # Error handling (should never be reached)
    else:
        return None


def get_3d_histogram(image, blue_lb, green_lb, red_lb, hist_ub, buckets):
    """
    Generates a single, 3d histogram to represent an image
    :param image:    The image to get the histogram of
    :param blue_lb:  Histogram range lower bound for blue channel
    :param green_lb: Histogram range lower bound for green channel
    :param red_lb:   Histogram range lower bound for red channel
    :param hist_ub:  Histogram range upper bound
    :param buckets:  Number of buckets to bin histogram into
    :return: The 3d histogram
    """
    hist_3d = cv2.calcHist(images=[image], channels=[BLUE_CHANNEL, GREEN_CHANNEL, RED_CHANNEL], mask=None,
                           histSize=[buckets, buckets, buckets],
                           ranges=[blue_lb, hist_ub, green_lb, hist_ub, red_lb, hist_ub])
    return hist_3d


def get_two_channel_histogram_projection(hist_3d, channel_1, channel_2):
    """
    Generates a single, 3d histogram to represent an image
    :param hist_3d:   The 3D histogram to get the 2d color projection of
    :param channel_1: First channel to get histogram with
    :param channel_2: Second channel to get histogram with
    :return: The 2d histogram
    """
    # Get the channel to drop
    include = np.array([channel_1, channel_2])
    drop_channel = np.setdiff1d(ALL_CHANNELS, include).item()

    # Drop the channel (sum all counts within that channel)
    hist_2d = np.sum(hist_3d, axis=drop_channel)
    return hist_2d


def get_plane_projected_histogram(image, plane, origin, blue_lb, green_lb, red_lb, hist_ub, buckets):
    """
    Gets the histogram of an image projected onto a plane
    :param image:    The image to get the projected histogram of
    :param plane:    The plane to project the histogram onto
    :param origin:   The origin of the plane
    :param blue_lb:  Histogram range lower bound for blue channel
    :param green_lb: Histogram range lower bound for green channel
    :param red_lb:   Histogram range lower bound for red channel
    :param hist_ub:  The upper bound of the values within the image
    :param buckets:  The number of buckets in the histogram
    :return: The projected histogram
    """
    # Get the u and v axes
    u_axis, v_axis = build_planar_axes(plane, origin, blue_lb, red_lb, hist_ub)

    # Get the points that bound the histogram
    u_min_projected, u_max_projected = get_min_max_projected_boundaries(u_axis, blue_lb, green_lb, red_lb, hist_ub)
    v_min_projected, v_max_projected = get_min_max_projected_boundaries(v_axis, blue_lb, green_lb, red_lb, hist_ub)

    # Project the image into the 111 space
    u_indexes = np.dot(image, u_axis)
    v_indexes = np.dot(image, v_axis)
    projected_image = np.stack((u_indexes, v_indexes), axis=2).astype(np.float32)

    # Get the projected histogram
    projected_histogram = cv2.calcHist(images=[projected_image], channels=[0, 1], mask=None,
                                       histSize=[buckets, buckets],
                                       ranges=[u_min_projected, u_max_projected, v_min_projected, v_max_projected])

    return projected_histogram


def build_planar_axes(plane, origin, blue_lb, red_lb, upper_bound):
    """
    Builds the planar axis from a plane, an origin point, and the maximum value of the image to project
    :param plane:       The plane to project onto
    :param origin:      The origin of the plane
    :param blue_lb:     Histogram range lower bound for blue channel
    :param red_lb:      Histogram range lower bound for red channel
    :param upper_bound: The upper bound of the image to project onto the plane
    :return: The u and v axes of the projected plane
    """
    normal_plane = plane / np.linalg.norm(plane)
    green_axis_point = np.array([blue_lb, np.log(upper_bound), red_lb])

    v_axis_unnormalized = project_3d_point(green_axis_point, origin, normal_plane) - origin
    v_axis = v_axis_unnormalized / np.linalg.norm(v_axis_unnormalized)

    u_axis = np.cross(v_axis, normal_plane)

    return u_axis, v_axis


def get_cube_boundaries(blue_lb, green_lb, red_lb, upper_bound):
    """
    Gets the boundary positions for the projection onto a plane
    :param blue_lb:     Histogram range lower bound for blue channel
    :param green_lb:    Histogram range lower bound for green channel
    :param red_lb:      Histogram range lower bound for red channel
    :param upper_bound: The upper bound of values in the image to project
    :return:
    """
    # Define all possible points involving each channels' lower and upper bound, and return as numpy array
    boundaries = []
    x_values = [blue_lb, upper_bound]
    y_values = [green_lb, upper_bound]
    z_values = [red_lb, upper_bound]
    for x in x_values:
        for y in y_values:
            for z in z_values:
                boundaries.append((x, y, z))
    return np.array(boundaries)


def get_min_max_projected_boundaries(axis, blue_lb, green_lb, red_lb, upper_bound):
    """
    Calculates the minimum and maximum for the axis components on the project surface
    :param axis:        The axis to calculate the bounds for
    :param blue_lb:     Histogram range lower bound for blue channel
    :param green_lb:    Histogram range lower bound for green channel
    :param red_lb:      Histogram range lower bound for red channel
    :param upper_bound: Upper bound of image to project
    :return: The minimum and maximum values along the axis
    """
    # Get the cube boundaries
    cube_boundaries = get_cube_boundaries(blue_lb, green_lb, red_lb, upper_bound)

    # Get the dot product of all cube boundaries with the axis
    dot_products = []
    for boundary_point in cube_boundaries:
        dot_products.append(np.dot(boundary_point, axis))
    return np.min(dot_products), np.max(dot_products)


def project_3d_point(point_3d, origin, normal_plane):
    """
    Projects a 3d point onto a plane
    :param point_3d:     The 3d point to project
    :param origin:       The origin of the plane to project onto
    :param normal_plane: The normal plane of the plane to project onto
    :return:
    """
    difference = origin - point_3d
    dot_product = np.dot(difference, normal_plane)
    normal_magnitude_square = np.dot(normal_plane, normal_plane)
    projected_point = point_3d - (dot_product / normal_magnitude_square) * normal_magnitude_square
    return projected_point


def get_histograms_projection_sliced_by_vector(
    image,
    buckets,
    slice_portions,
    bgr_lower_bound,
    bgr_upper_bound,
    slice_vector,
    z_vector
):
    """
    Slice the image in RGB space along a given slice vector, and in each slice,
    project the pixels onto the plane defined by the z_vector and its orthogonal axis.

    :param image: RGB image (uint8 or float32)
    :param slice_vector: A 3D direction vector for slicing
    :param z_vector: A 3D direction vector used as part of the projection basis
    :param buckets: Number of histogram bins in each axis
    :param slice_portions: Proportions of the slices
    :param bgr_lower_bound: Lower bounds for RGB values [r_min, g_min, b_min]
    :param bgr_upper_bound: Upper bounds for RGB values [r_max, g_max, b_max]
    :return: List of 2D RGB histograms (one per slice)
    """
    start_time = time.time()
    image = image.astype(np.float32)

    # Apply RGB bounds
    bgr_lower = np.array(bgr_lower_bound, dtype=np.float32)
    bgr_upper = np.array(bgr_upper_bound, dtype=np.float32)

    # Clip image values to the specified RGB bounds
    image = np.clip(image, bgr_lower[np.newaxis, np.newaxis, :],
                    bgr_upper[np.newaxis, np.newaxis, :])

    pixels = image.reshape(-1, 3)

    # Normalize input vectors
    slice_axis = np.array(slice_vector, dtype=np.float32)
    slice_axis /= np.linalg.norm(slice_axis)

    z = np.array(z_vector, dtype=np.float32)
    z /= np.linalg.norm(z)

    # Orthogonalize slice_axis with respect to z (keep z unchanged)
    slice_axis_proj_z = np.dot(slice_axis, z) * z
    slice_axis_orthog = slice_axis - slice_axis_proj_z

    # Normalize input slice_axis
    slice_axis = slice_axis_orthog / np.linalg.norm(slice_axis_orthog)

    # Create perp_axis orthogonal to both z and slice_axis
    perp_axis = np.cross(slice_axis, z)
    perp_axis /= np.linalg.norm(perp_axis)

    # Project all pixels onto the slice direction
    # @ - doing dot product between each pixel and slice_axis
    slice_coords = pixels @ slice_axis

    # Compute min/max along slice axis using RGB cube corners with the provided bounds
    bgr_corners = np.array(list(product(
        [bgr_lower[0], bgr_upper[0]],
        [bgr_lower[1], bgr_upper[1]],
        [bgr_lower[2], bgr_upper[2]]
    )), dtype=np.float32)

    # Project RGB cube corners into the axis and get boundaries
    projections = bgr_corners @ slice_axis
    s_min, s_max = np.min(projections), np.max(projections)

    projections_z = bgr_corners @ z
    z_min, z_max = np.min(projections_z), np.max(projections_z)

    projections_y = bgr_corners @ perp_axis
    y_min, y_max = np.min(projections_y), np.max(projections_y)

    # Set up slicing bins
    slice_portions = np.array(slice_portions, dtype=np.float32)
    # eg. slice_portions = [0.25, 0.25, 0.25, 0.25]
    #  -> cumulative = [0, 0.25, 0.5, 0.75, 1]
    cumulative = np.cumsum([0.0] + slice_portions.tolist())
    bin_edges = s_min + cumulative * (s_max - s_min)

    count_hists = []

    for slice_id in range(len(slice_portions)):
        lower = bin_edges[slice_id]
        upper = bin_edges[slice_id + 1]
        mask = (slice_coords >= lower) & (slice_coords < upper)
        slice_pixels = pixels[mask]

        if len(slice_pixels) == 0:
            count_hists.append(np.ones((buckets, buckets), dtype=np.int32))
            continue

        # Project selected pixels to 2D plane defined by z and perp_axis
        z_proj = slice_pixels @ z
        y_proj = slice_pixels @ perp_axis

        # Normalized to [0, 1], times buckets, project to [0, buckets)
        bin_idx_z = np.floor((z_proj - z_min) / (z_max - z_min + 1e-6) * buckets).astype(int)
        bin_idx_y = np.floor((y_proj - y_min) / (y_max - y_min + 1e-6) * buckets).astype(int)
        bin_idx_z = np.clip(bin_idx_z, 0, buckets - 1)
        bin_idx_y = np.clip(bin_idx_y, 0, buckets - 1)

        count_hist = np.ones((buckets, buckets), dtype=np.int32)

        for i in range(len(slice_pixels)):
            y_bin = bin_idx_y[i]
            z_bin = bin_idx_z[i]
            count_hist[z_bin, y_bin] += 1

        count_hist[count_hist == 0] = 1
        count_hists.append(count_hist)
    end_time = time.time()
    elapsed_time = end_time - start_time
    # print(f"Histogram generated in {elapsed_time:.4f} seconds")
    # Return the histogram that only contains the count of the bin
    return count_hists
