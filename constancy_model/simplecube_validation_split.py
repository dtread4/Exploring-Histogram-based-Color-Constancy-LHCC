# David Treadwell
# treadwell.d@northeastern.edu
# Northeastern University
# Professor Bruce Maxwell
# simplecube_validation_split.py - splits SimpleCube++ training data into a separate train/validation split
# IMPORTANT USAGE NOTES
# Copy the original SimpleCube++ data to a new directory before running this script, as original files will be moved
# Create a folder called "train" that contains a subfolder called "PNG"
# Create an empty folder called "test" (in the main output directory)
# Moves 462 (size of test set) random images from the train folder into the empty test folder
# Put the original gt.csv file into the directory where validation data splits contained, and name it "gt_train_org.csv"

import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil


def move_valid_images(train_images_copy, valid_names, output_dir):
    # Create output directory
    output_image_path = output_dir + os.sep + "test"
    os.makedirs(output_image_path, exist_ok=True)
    output_image_path = output_image_path + os.sep + "PNG"
    os.makedirs(output_image_path, exist_ok=True)

    # Move images in valid_names to the test directory
    print("Moving validation images... ", end='')
    for valid_image in valid_names:
        shutil.move(train_images_copy + os.sep + valid_image, output_image_path + os.sep + valid_image)
    print("Finished!")


def create_new_gt_csvs(gt_train_org, valid_names, output_dir):
    # Build new pandas dataframes
    train_gt = pd.DataFrame(columns=['image', 'mean_r', 'mean_g', 'mean_b'])
    valid_gt = pd.DataFrame(columns=['image', 'mean_r', 'mean_g', 'mean_b'])

    # Iterate through original ground truth set and add each row to correct new dataframe
    for index, row in gt_train_org.iterrows():
        if row['image'] + ".png" in valid_names:
            valid_gt.loc[len(valid_gt)] = gt_train_org.iloc[index]
        else:
            train_gt.loc[len(train_gt)] = gt_train_org.iloc[index]

    # Save both dataframes to correct location as csv
    os.makedirs(output_dir + os.sep + "train", exist_ok=True)
    train_gt.to_csv(output_dir + os.sep + "train" + os.sep + "gt.csv", index=False)
    valid_gt.to_csv(output_dir + os.sep + "test" + os.sep + "gt.csv", index=False)
    print("Finished updating ground truth csv's!")


def create_train_valid_split(train_images_copy, gt_train_org, output_dir):
    # Split data into train and validation
    test_size = 462 / 1772
    image_names = os.listdir(train_images_copy)
    print(len(image_names))
    train_names, valid_names = train_test_split(image_names, test_size=test_size, random_state=42)

    # Move the valid images to the output directory
    move_valid_images(train_images_copy, valid_names, output_dir)

    # Create new ground truth csv's
    create_new_gt_csvs(gt_train_org, valid_names, output_dir)


def main(argv):
    # Grab arguments from command line
    train_images_copy = argv[1]
    gt_train_org_csv = argv[2] 
    output_dir = argv[3]

    # Read ground truth data
    gt_train_org = pd.read_csv(gt_train_org_csv, dtype={"image": str, "mean_r": str, "mean_g": str, "mean_b": str})

    # Split the data
    create_train_valid_split(train_images_copy, gt_train_org, output_dir)


if __name__ == "__main__":
    """
    Command line arguments needed:
        1. Path to train images (should be PNG directory within train directory, not overall train directory)
        2. Path to ground truth CSV
        3. Output directory
    """   
    main(sys.argv)
