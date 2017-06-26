# import packages
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import argparse
import os

# import modules
# import heatmap.heatmap as hmp

# GLOBAL VARIABLES
NUM_CLASSES = 4
IMG_ROW = 1200
IMG_COL = 1600
IMG_CHN = 3
BB_ROW = 32
BB_COL = 32
BB_STRIDE = 5

# parse the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--classifydir', nargs=1, type=str, required=True,
                    help='Directory containing images to classify.')
parser.add_argument('-s', '--savedir', nargs=1, type=str, required=True,
                    help='Directory where to save the image masks.')

# parse argument and convert to dictionary
args = vars(parser.parse_args())

# check that the image path exists
assert os.path.isdir(args["classifydir"][0]), \
       "Directory containing the images to classify does not exist."

# create the save directory if it doesn't exist.
if not os.path.isdir(args["savedir"][0]):
    os.mkdir(args["savedir"][0])

for fname in os.listdir(args["classifydir"][0]):
    # skip annoying hidden files
    if fname.startswith("."):
        continue

    # load the image file
    wing_img = mpimg.imread(os.path.join(args["classifydir"][0], fname))

    # construct heatmap
    heatmap = heatmap.compute_heatmap(wing_img)
    generate_image_mask(heatmap, os.path.join(args["savedir"][0], fname))
