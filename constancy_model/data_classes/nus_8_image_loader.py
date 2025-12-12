# David Treadwell
# treadwell.d@northeastern.edu
# Northeastern University
# Professor Bruce Maxwell
# nus_8_image_loader.py - Dataset image loader for NUS-8 cameras.
# 
# Please refer to usage in class description and README

import cv2
from constancy_model.data_classes.image_loader_base_class import ImageLoader
from constancy_model.parameter_manager import load_json
from abc import ABC
import scipy.io
import numpy as np
import os


class NUS8ImageLoader(ImageLoader, ABC):
    """
    Class to load NUS-8; provides shared functionality for all NUS-8 cameras

    Data available at: https://yorkucvil.github.io/projects/public_html/illuminant/illuminant.html
    After downloading and unzipping the data, create a directory for each camera and place all images, masks,
    and ground truth MAT files into each camera's directory.
    Also need to create cross folds, which can be taken from FFCC: https://github.com/google/ffcc/tree/master
    """
    def __init__(self, camera_name, validation_fold_number, image_scale_factor):
        # Load files for camera
        dataset_name = "NUS-8"
        data_file_path = "constancy_model" + os.sep + "data_classes" + os.sep + "data_paths.json"
        data_paths = load_json(data_file_path)[dataset_name][0]
        dataset_paths = data_paths['data_path']
        cross_fold_filename = data_paths['cross_fold_file_name']

        # Get paths for specific camera
        print("Using NUS-8 camera dataset: {}".format(camera_name))
        camera_base_path = dataset_paths + os.sep + camera_name
        train_images_path = camera_base_path + os.sep + camera_name + "_PNG" + os.sep + "PNG"
        valid_images_path = train_images_path
        train_gt_path = camera_base_path + os.sep + camera_name + "_gt.mat"
        valid_gt_path = train_gt_path
        cross_fold_path = camera_base_path + os.sep + cross_fold_filename

        # Set alias paths for images and ground truth
        self.images_path = train_images_path
        self.ground_truth_path = train_gt_path

        # Load black level and saturation value for camera
        black_level, saturation_value = self.get_black_level_and_saturation_value()

        # Initialize parent class
        super().__init__(black_level, saturation_value, validation_fold_number, train_images_path, valid_images_path,
                         train_gt_path, valid_gt_path, cross_fold_path, image_scale_factor)

        # Set ground truth dataframes
        self.load_ground_truth_dataframe()
        self.coordinates_dictionary = self.create_crop_coordinates_dictionary()

    def get_black_level_and_saturation_value(self):
        """
        Reads the ground truth MAT file to obtain the black level and saturation value
        :return: Black level and saturation value
        """
        black_level = scipy.io.loadmat(self.ground_truth_path)['darkness_level']
        saturation_value = scipy.io.loadmat(self.ground_truth_path)['saturation_level']
        return black_level, saturation_value

    def create_crop_coordinates_dictionary(self):
        """
        Creates a dictionary to map images to their ColorChecker coordinates
        :return: The dictionary of image->ColorChecker coordinates
        """
        # Get the array of ColorChecker coordinates
        colorchecker_coordinates = np.array(scipy.io.loadmat(self.ground_truth_path)['CC_coords'])

        # Get the image names
        image_names = self.load_image_names(self.images_path)

        # Combine both into a dictionary to match an image to its mask
        coordinates_dictionary = dict(zip(image_names, colorchecker_coordinates))
        return coordinates_dictionary

    def load_ground_truth_dataframe(self):
        """
        Loads the ground truth data into a train and test array
        :return: None
        """
        # Read the ground truth data into an array for red, green, and blue
        ground_truth_array = np.swapaxes(np.array(scipy.io.loadmat(self.ground_truth_path)['groundtruth_illuminants']), 0, 1)
        red_gt = ground_truth_array[0]
        green_gt = ground_truth_array[1]
        blue_gt = ground_truth_array[2]

        # Normalize red, green, and blue values
        sum_illuminations = red_gt + green_gt + blue_gt
        red_normal = red_gt / sum_illuminations
        green_normal = green_gt / sum_illuminations
        blue_normal = blue_gt / sum_illuminations

        # Read all image names into an array
        image_names = self.load_image_names(self.images_path)

        # Create dataframe from the loaded information
        self.create_ground_truth_dataframes_from_folds(image_names, red_normal, green_normal, blue_normal)

    def load_image(self, train_data, image_index):
        """
        Loads an image based on its index, with preprocessing
        :param train_data:  True if loading a train image, False if not
        :param image_index: The index (within train/valid DataFrame) of the image to load
        :return: The image ready to use
        """
        # Get the original image
        original_image = self.load_original_image(train_data, image_index)

        # Crop the ColorChecker out of the image
        cropped_image = self.crop_colorchecker(original_image, train_data, image_index)

        # Resize if necessary
        resized_image = self.resize_image(cropped_image)
        return resized_image

    def crop_colorchecker(self, original_image, train_data, image_index):
        """
        Crops the ColorChecker out of an image
        Adapted from the code of Yuanming Hu et al. "FC4" https://github.com/yuanming-hu/fc4/blob/master/datasets.py
        :param original_image: The original image with the ColorChecker in it
        :param train_data:  True if a training image, False if a validation image
        :param image_index: Index of image within train/validation DataFrame
        :return: The image with its ColorChecker cropped out
        """
        # Get the coordinates of the ColorChecker
        image_name = self.get_image_name_from_index(train_data, image_index)
        coordinates = self.coordinates_dictionary[image_name]

        # Apply the crop
        y1, y2, x1, x2 = coordinates
        crop_points = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]).astype(np.int32)
        cv2.fillPoly(original_image, pts=[crop_points], color=(0, 0, 0))
        return original_image
