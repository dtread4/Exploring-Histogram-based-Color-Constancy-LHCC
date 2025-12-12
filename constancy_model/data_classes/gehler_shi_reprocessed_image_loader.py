# David Treadwell
# treadwell.d@northeastern.edu
# Northeastern University
# Professor Bruce Maxwell
# gehler_shi_reprocessed_image_loader.py - Dataset image loader for Gehler-Shi Reprocessed
# 
# Please refer to usage in class description and README

import cv2
from constancy_model.data_classes.image_loader_base_class import ImageLoader
from constancy_model.parameter_manager import load_json
from abc import ABC
import scipy.io
import numpy as np
import os


class GehlerShiReprocessedLoader(ImageLoader, ABC):
    """
    Class to load Gehler-Shi Reprocessed dataset; provides shared functionality for both cameras in single training
    Before using, put all images into a single directory, and update the paths appropriately in the data_paths.json file
    DO NOT CHANGE IMAGE NAMES as they are helpful for differentiating between cameras

    Images available at: https://www.cs.sfu.ca/~colour/data/shi_gehler/
    Coordinates available at: https://github.com/matteo-rizzo/fc4-pytorch/tree/main/dataset/coordinates
    After downloading and unzipping the data, place all images in the same directory
    Also need to create cross folds, which can be taken from FFCC:
    https://github.com/google/ffcc/blob/2fa9e1316954dbd3913630b7d597927941b4dd32/data/shi_gehler/preprocessed/GehlerShi/cvfolds.txt#L4
    """
    def __init__(self, validation_fold_number, image_scale_factor):
        # Set shared variables
        # NOTE: Canon 5D images have a native black level of 129, but they will be reprocessed during image loading
        # to subtract their black level and rescaled back to [0, 2**12 - 1] range
        black_level = 0
        saturation_value = 2**12 - 1
        self.canon_5d_black_level = 129

        # Load files for camera
        dataset_name = "Gehler-Shi Reprocessed"
        data_file_path = "constancy_model" + os.sep + "data_classes" + os.sep + "data_paths.json"
        dataset_paths = load_json(data_file_path)[dataset_name][0]

        # Get paths from dataset json
        train_images_path = dataset_paths['images_path']
        valid_images_path = dataset_paths['images_path']
        train_gt_path = dataset_paths['ground_truth_path']
        valid_gt_path = dataset_paths['ground_truth_path']
        cross_fold_path = dataset_paths['cross_fold_path']
        coordinates_path = dataset_paths['coordinates_path']

        # Initialize parent class
        super().__init__(black_level, saturation_value, validation_fold_number, train_images_path, valid_images_path,
                         train_gt_path, valid_gt_path, cross_fold_path, image_scale_factor)

        # Set alias paths for images and ground truth
        self.images_path = train_images_path
        self.ground_truth_path = train_gt_path

        # Set ground truth dataframes
        self.load_ground_truth_dataframe()
        self.coordinates_dictionary = self.create_crop_coordinates_dictionary(coordinates_path)

    def get_image_upper_bound(self, train_data, image_index):
        """
        Returns the upper bound (saturation value - black level) for an image
        Canon1D will return the saturation value directly, but Canon5D will return a different upper bound
        :param train_data:  Whether image is from training data or not
        :param image_index: The index of the image to get the upper bound for
        :return: The upper bound for the dataset
        """
        # Get the image name
        image_name = self.get_image_name_from_index(train_data, image_index)

        # Get the appropriate upper bound
        if self.is_canon_5d(image_name):
            upper_bound = self.saturation_value - self.canon_5d_black_level
        else:
            upper_bound = self.upper_bound
        return upper_bound

    def create_crop_coordinates_dictionary(self, coordinates_path):
        """
        Creates a dictionary to map images to their ColorChecker coordinates
        Adapted from FC4 implementation by Matteo Rizzo:
        https://github.com/matteo-rizzo/fc4-pytorch/blob/main/dataset/img2npy.py
        :param coordinates_path: The path to the text file containing the ColorChecker coordinates
        :return: The dictionary of image->ColorChecker coordinates
        """
        # Extract the coordinates for each ColorChecker from each file
        colorchecker_coordinates = []
        for file_name in os.listdir(coordinates_path):
            file_path = coordinates_path + os.sep + file_name
            lines = open(file_path, 'r').readlines()
            width, height = map(float, lines[0].split())
            scale_x, scale_y = 1 / width, 1 / height
            polygon = []
            for line in [lines[1], lines[2], lines[4], lines[3]]:
                line = line.strip().split()
                x, y = (scale_x * float(line[0])), (scale_y * float(line[1]))
                polygon.append((x, y))
            colorchecker_coordinates.append(polygon)
        colorchecker_coordinates = np.array(colorchecker_coordinates, dtype='float32')

        # Create a dictionary for the image names to map to their ColorChecker coordinates
        image_names = self.load_image_names(self.images_path)
        coordinates_dictionary = dict(zip(image_names, colorchecker_coordinates))
        return coordinates_dictionary

    def load_ground_truth_dataframe(self):
        """
        Loads the ground truth data into a train and test array
        :return: None
        """
        # Read the ground truth data into an array for red, green, and blue
        ground_truth_array = np.swapaxes(np.array(scipy.io.loadmat(self.ground_truth_path)['real_rgb']), 0, 1)
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

        # Black level subtraction if image is Canon5D
        image_name = self.get_image_name_from_index(train_data, image_index)
        if self.is_canon_5d(image_name):
            cropped_image = self.adjust_black_level(cropped_image, self.canon_5d_black_level,
                                                    self.saturation_value - self.canon_5d_black_level)

        # Resize if necessary
        resized_image = self.resize_image(cropped_image)
        return resized_image

    def crop_colorchecker(self, original_image, train_data, image_index):
        """
        Crops the ColorChecker out of an image
        Adapted from FC4 implementation by Matteo Rizzo:
        https://github.com/matteo-rizzo/fc4-pytorch/blob/main/dataset/img2npy.py
        :param original_image: The original image with the ColorChecker in it
        :param train_data:  True if a training image, False if a validation image
        :param image_index: Index of image within train/validation DataFrame
        :return: The image with its ColorChecker cropped out
        """
        # Get the coordinates of the ColorChecker
        image_name = self.get_image_name_from_index(train_data, image_index)
        coordinates = self.coordinates_dictionary[image_name]

        # Apply the crop
        polygon = (coordinates * np.array([original_image.shape[1], original_image.shape[0]])).astype(np.int32)
        cv2.fillPoly(original_image, pts=[polygon], color=[0, 0, 0])
        return original_image

    def is_canon_5d(self, image_name):
        """
        Checks if an image is from the Canon5D camera base don image name
        Canon5D start with "IMG_" while Canon1D start with "8D5U"
        :param image_name: The name of the image
        :return: True if the image is a Canon5D image, False if not (image is Canon1D image)
        """
        return image_name[:4] == "IMG_"
