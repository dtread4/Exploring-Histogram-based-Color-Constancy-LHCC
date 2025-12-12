# David Treadwell
# treadwell.d@northeastern.edu
# Northeastern University
# Professor Bruce Maxwell
# histogram_loader.py - Class to load the histograms from images, creating the projections

import sys
import numpy as np
from torch import unsqueeze, from_numpy
from torch.utils.data import Dataset
from constancy_model.histogram_projection.histogram_projections import *
import time
import multiprocessing as mp
import torch
import copy


class HistogramDataset(Dataset):
    """
    Class to load the histograms and the associated ground-truth illumination
    """
    def __init__(self, experiment_parameters, train_data, dataset, preloaded_histograms=None,
                 transform=None, target_transform=None):
        """
        Constructor for the histogram dataset
        :param experiment_parameters: The parameter manager object for the dataset
        :param train_data:            Whether to load train or validation data from data directory
        :param dataset:               The dataset to load images from
        :param preloaded_histograms:  Set of histograms that a precalculated; used with static training
        :param transform:             Transform to apply to input data (leave as None)
        :param target_transform:      Transform to apply to ground-truth illumination value (leave as None)
        """
        # Set information about data to load
        self.train_data = train_data

        # Set the image loader dataset
        self.dataset = dataset
        self.gt_df = self.dataset.train_gt_df if self.train_data else self.dataset.valid_gt_df

        # Set the threshold on the lower bounds (IN 8-BITS, MAY BE SCALED)
        self.blue_threshold_8bit = experiment_parameters.blue_threshold_8bit
        self.green_threshold_8bit = experiment_parameters.green_threshold_8bit
        self.red_threshold_8bit = experiment_parameters.red_threshold_8bit

        # Set information about histograms to generate
        self.num_buckets = experiment_parameters.num_buckets
        self.histograms_per_image = experiment_parameters.histograms_per_image
        self.color_space = experiment_parameters.color_space
        self.histogram_saturation_method = experiment_parameters.histogram_saturation_method
        self.clip_upper_bound = experiment_parameters.clip_upper_bound
        self.projection_type = experiment_parameters.projection_type

        # Set transform (should be None)
        self.transform = transform
        self.target_transform = target_transform

        # If using static projections, load and store them in a numpy array
        if self.projection_type == "STATIC" or self.projection_type == "SLICES":

            self.preloaded_histograms = []
            if preloaded_histograms is not None:
                self.set_preloaded_histograms(preloaded_histograms)

            else:
                print("Projection type: {} | ".format(self.projection_type), end='')
                if self.projection_type == "STATIC":
                    self.initialize_preloaded_histograms(self.get_static_projections)
                elif self.projection_type == "SLICES":
                    self.initialize_preloaded_histograms(self.get_histogram_slices)

    def __len__(self):
        """
        Returns the length of the dataset (number of images in the dataset)
        :return: The length of the dataset
        """
        return len(self.gt_df)

    def __getitem__(self, idx):
        """
        Loads the histograms for an image and the associated ground-truth illumination values
        :param idx: The index of the image to load
        :return: A tuple of the histograms (as a tuple), and the illumination value (as a tuple)
                 e.g. ((hist1, hist2), (red, blue, green))
        Shape of histograms object below based on projection type
        GENERAL STRUCTURE: [batch size, # histograms, bin channel, num_buckets, num_buckets]
        """
        # Get the set of histograms
        histograms = self.calculate_histogram_projections(idx)

        # Extract the ground-truth illumination value
        gt_illumination = self.get_ground_truth_illuminations(idx)

        # Apply any transforms
        if self.transform:
            histograms = self.transform(histograms)
        if self.target_transform:
            gt_illumination = self.target_transform(gt_illumination)

        # Return the histograms and illumination values
        return histograms, gt_illumination

    def set_preloaded_histograms(self, preloaded_histograms):
        """
        Sets histograms from a preloaded histogram dictionary based on image names in the train/valid set
        :param preloaded_histograms: The dictionary of preloaded histograms to set from
        :return: None
        """
        for i in range(len(self.gt_df)):
            image_name = self.gt_df.iloc[i, 0]
            self.preloaded_histograms.append(preloaded_histograms[image_name])
        self.preloaded_histograms = np.array(self.preloaded_histograms)
        print("Set histograms from preloaded {} set successfully! Loaded histograms for {} images".format(
            "train" if self.train_data else "valid", len(self.gt_df)
        ))

    def initialize_preloaded_histograms(self, histogram_function):
        """
        Initializes preloaded histograms before accessing during training/validation to speed up training time
        :param histogram_function: The function used to calculate the histogram (should be based on image indexes)
        :return: None
        """
        # Setup steps
        start_time = int(round(time.time()))
        print("Pre-generating histogram projections for {} set ({} images) in {} color space... ".format(
            "train" if self.train_data else "valid", len(self.gt_df), self.color_space), end='')

        # Calculate the histogram for each image
        for i in range(len(self.gt_df)):
            histograms = histogram_function(i)
            self.preloaded_histograms.append(histograms)

        # Make histogram array a numpy array and print conclusion message
        self.preloaded_histograms = np.array(self.preloaded_histograms)
        end_time = int(round(time.time()))
        time_taken = end_time - start_time
        print("Finished! Took {} seconds".format(time_taken))

    def calculate_histogram_projections(self, idx):
        """
        Builds the projected histograms for an image
        :param idx: The index of the image to calculate histograms for
        :return: The projected histograms
        """
        # Generate histograms based on desired type of projection
        if self.projection_type == "STATIC":
            histograms = self.preloaded_histograms[idx]
            return histograms
        if self.projection_type == "SLICES":
            histograms = self.preloaded_histograms[idx]
            return histograms
        else:
            print("ERROR: INVALID PROJECTION TYPE! Please select from the projection types below:")
            print("STATIC - generates four 2D projections (111, RG, RB, GB)")
            print("SLICES - generates four 2D histogram based on given vector")
            print("Exiting...")
            sys.exit(-1)

    def get_ground_truth_illuminations(self, idx):
        """
        Gets the triplet of ground truth RGB values associated with an index, which ties to an image
        :param idx: The index of the image to calculate histograms for
        :return: The ground truth RGB triplet
        """
        r_val = self.gt_df.iloc[idx, 1]
        g_val = self.gt_df.iloc[idx, 2]
        b_val = self.gt_df.iloc[idx, 3]
        gt_illumination = np.array((r_val, g_val, b_val))
        return gt_illumination

    def set_image_upper_bound_and_scale(self, org_image, image_index):
        """
        Sets image bounds and scales based on image properties
        :param org_image:   The original image to scale
        :param image_index: The index of the image within the dataset
        :return: The scaled image, the lower RGB bound, and the upper RGB bound
        """
        # Set upper bound in rgb space
        image_upper_bound = self.dataset.get_image_upper_bound(self.train_data, image_index)
        rgb_ub = get_hist_upper_bound(image_upper_bound)
        image_scaled = scale_to_upper_bound(org_image, image_upper_bound, rgb_ub)
        return image_scaled, rgb_ub

    def round_up_and_make_integer(self, value):
        """
        Rounds a value up and makes it in an integer
        :param value: The value to round
        :return: The value rounded up as an integer
        """
        return int(value + 0.9999)  # Adding 0.9999 so integers stay the same, but any floats round up

    def scale_lower_bounds(self, org_image):
        """
        Scales the lower bound (threshold) for generating the histogram to the correct bit depth
        :param org_image: The original image to check its bit depth
        :return: The blue, green, and red lower bounds
        """
        if org_image.dtype == np.uint8:
            return self.round_up_and_make_integer(self.blue_threshold_8bit),\
                   self.round_up_and_make_integer(self.green_threshold_8bit),\
                   self.round_up_and_make_integer(self.red_threshold_8bit)
        elif org_image.dtype == np.uint16:
            blue_threshold = self.round_up_and_make_integer(self.blue_threshold_8bit / (2**8 - 1) * (2**16 - 1))
            green_threshold = self.round_up_and_make_integer(self.green_threshold_8bit / (2**8 - 1) * (2**16 - 1))
            red_threshold = self.round_up_and_make_integer(self.red_threshold_8bit / (2**8 - 1) * (2**16 - 1))
            return blue_threshold, green_threshold, red_threshold
        else:
            print("ERROR! Invalid image type. Exiting...")
            sys.exit(-1)

    def get_static_projections(self, image_index):
        """
        Generates four standard histogram projections (111, rg, rb, gb) for an image
        :param image_index: The index of the image to get projections for
        :return: The projected histograms
        """
        # Set up values for histogram
        image_for_hist, hist_b_lb, hist_g_lb, hist_r_lb, hist_ub = self.prepare_histogram_calculation_values(image_index)

        # Get the 3D histogram
        histogram_3d = get_3d_histogram(image_for_hist, hist_b_lb, hist_g_lb, hist_r_lb, hist_ub, self.num_buckets)

        # Set up projection names
        orientations = ['RG', 'RB', 'GB', '111']

        # Generate each of the four histograms
        projected_hists = []
        for orientation in orientations:
            projected_hists.append(get_projected_histogram(histogram_3d, image_for_hist, hist_b_lb, hist_g_lb,
                                                           hist_r_lb, hist_ub, self.num_buckets, orientation))

        # Set histograms from results
        hist_rg = projected_hists[0]
        hist_rb = projected_hists[1]
        hist_gb = projected_hists[2]
        hist_111 = projected_hists[3]

        # Prepare each histogram
        hist_rg = self.prep_histogram_for_pytorch(hist_rg)
        hist_rb = self.prep_histogram_for_pytorch(hist_rb)
        hist_gb = self.prep_histogram_for_pytorch(hist_gb)
        hist_111 = self.prep_histogram_for_pytorch(hist_111)

        # Return the four histograms in single array
        histograms = np.array((hist_rg, hist_rb, hist_gb, hist_111)).astype(np.float32)
        return histograms

    def get_histogram_slices(self, image_index):
        """
        Calculates the histogram of an image based on its slices
        :param image_index: The image histogram to generate
        :return: The slices of the histogram
        """
        # Set up values for histogram
        image_scaled, hist_b_lb, hist_g_lb, hist_r_lb, hist_ub = self.prepare_histogram_calculation_values(image_index)

        # Prepare RGB bounds
        bgr_lower_bound = np.array([hist_b_lb, hist_g_lb, hist_r_lb])
        bgr_upper_bound = np.array([hist_ub, hist_ub, hist_ub])

        num_buckets = self.num_buckets

        # Decide which set of eigen vectors to use depending on linear/log image
        # Decide slicing portions depending on linear/log image
        z_vector, slice_vector, slice_portions = get_projection_vector_and_slice_portion(self.color_space)

        # Get the histogram as a list of 2D NumPy array: [[histogram] * 4]
        histograms = get_histograms_projection_sliced_by_vector(
            image_scaled,
            num_buckets,
            slice_portions,
            bgr_lower_bound,
            bgr_upper_bound,
            slice_vector,
            z_vector
        )

        # Process each histogram slice for PyTorch
        processed_histograms = []
        for histogram in histograms:
            processed_hist = self.prep_histogram_for_pytorch(histogram)
            processed_histograms.append(processed_hist)
        
        # Convert to numpy array and return
        processed_histograms = np.array(processed_histograms).astype(np.float32)
        return processed_histograms

    def prepare_histogram_calculation_values(self, image_index):
        """
        Loads the image, calculates the bounds needed to generate histograms, scales/thresholds,
        and applies a log conversion (if required)
        :param image_index: The index of the image being loaded
        :return: The processed image and the necessary bounds for the histogram
        """
        # Load the image
        org_image = self.dataset.load_image(self.train_data, image_index)

        # Set upper bound in rgb space
        image_scaled, rgb_ub = self.set_image_upper_bound_and_scale(org_image, image_index)

        # Get lower bounds
        blue_lb, green_lb, red_lb = self.scale_lower_bounds(org_image)

        # Convert image to log if necessary
        if self.color_space == "LOG":
            image_for_hist = convert_to_log(image_scaled)
            hist_b_lb, hist_g_lb, hist_r_lb, hist_ub = get_log_bounds(blue_lb, green_lb, red_lb, rgb_ub)
        elif self.color_space == "LINEAR":
            image_for_hist = copy.deepcopy(image_scaled)
            hist_b_lb, hist_g_lb, hist_r_lb, hist_ub = blue_lb, green_lb, red_lb, rgb_ub
        else:
            print("ERROR: invalid color space. Use only \"LINEAR\" or \"LOG\". Exiting...")
            sys.exit(-1)

        # Return values needed to generate histograms properly
        return image_for_hist, hist_b_lb, hist_g_lb, hist_r_lb, hist_ub

    def filter_image_by_channel_threshold(self, image, blue_lb, green_lb, red_lb):
        """
        Filters an image so only pixels that contain non-zero values remain
        :param image:    The original image to filter
        :param blue_lb:  The lower bound of the blue channel
        :param green_lb: The lower bound of the green channel
        :param red_lb:   The lower bound of the red channel
        :return: The image filtered from the lower bounds
        """
        # Create filtered image
        filter_image = image.copy()

        # Get valid indexes for each channel
        indexes_thresholds_met = (
            (filter_image[:, :, 0] >= blue_lb) &
            (filter_image[:, :, 1] >= green_lb) &
            (filter_image[:, :, 2] >= red_lb)
        )

        # Get full set of valid indexes
        filter_image = filter_image[indexes_thresholds_met][:, np.newaxis, :]
        return filter_image

    def prep_histogram_for_pytorch(self, histogram):
        """
        Prepares a histogram for use with PyTorch
        :param histogram: The histogram to prepare
        :return: The histogram ready for use with PyTorch setup
        """
        processed = self.process_histogram_saturation(histogram)
        np.set_printoptions(suppress=True)
        tensor_hist = from_numpy(processed)
        unsqueezed = unsqueeze(tensor_hist, dim=0)
        return unsqueezed

    def clip_histogram(self, histogram):
        """
        Clips a histogram to defined range to give better weight to full set of colors
        Also sets the type to uint8
        If clip_upper_bound is set to -1, no clipping or type changing will occur
        :param histogram: The histogram to clip
        :return: The clipped histogram
        """
        if self.clip_upper_bound != -1:
            # Clip the histogram
            clipped_histogram = np.clip(histogram, 0, self.clip_upper_bound)

            # Set the data type to the correct unsigned integer value
            if self.clip_upper_bound < 2**8:
                clipped_histogram = clipped_histogram.astype(np.uint8)
            elif 2**8 <= self.clip_upper_bound < 2**16:
                clipped_histogram = clipped_histogram.astype(np.int16)
            elif 2**16 <= self.clip_upper_bound < 2**32:
                clipped_histogram = clipped_histogram.astype(np.int32)
            else:
                clipped_histogram = clipped_histogram.astype(np.int64)

        # If clip bound was -1, leave histogram as is
        else:
            clipped_histogram = histogram

        return clipped_histogram

    def apply_tanh_histogram(self, histogram):
        """
        Applies the hyperbolic tangent function to the histogram
        :param histogram: The histogram to apply tanh to
        :return: The histogram after applying tanh
        """
        return np.tanh(histogram)

    def apply_log_histogram(self, histogram):
        """
        Applies a natural log to the histogram, after clipping values up to a minimum of 1 (to prevent log(0) errors)
        :param histogram: The histogram to apply the log operation to
        :return: The histogram after applying log
        """
        log_histogram = np.clip(histogram, 1, np.max(histogram))
        log_histogram = np.log(log_histogram)
        return log_histogram

    def process_histogram_saturation(self, histogram):
        """
        Applies the correct processing method for the histogram
        CLIP: Clips to a specified upper bound (-1 applies no clipping)
        TANH: Applies tanh function to the bucket counts
        LOG : Applies log function to bucket counts
        LINEAR: Applies no operation (equivalent to clip bound of -1)
        :param histogram: The histogram to process the saturation of
        :return: The processed histogram
        """
        if self.histogram_saturation_method == "CLIP":
            return self.clip_histogram(histogram)

        elif self.histogram_saturation_method == "TANH":
            return self.apply_tanh_histogram(histogram)

        elif self.histogram_saturation_method == "LOG":
            return self.apply_log_histogram(histogram)

        elif self.histogram_saturation_method == "LINEAR":
            return histogram

        else:
            print("ERROR! Invalid histogram saturation processing method used. Please only select from the following:\n"
                  "CLIP,\nTANH,\nLOG,\nLINEAR")
            sys.exit(-1)

