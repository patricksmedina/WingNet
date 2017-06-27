# import external packages
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import argparse
import os

# import internal moduless
from wingnet.heatmap import heatmap as hmp

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

for fname in os.listdir(args["classifydir"][0])[:2]:
    # skip annoying hidden files
    if fname.startswith("."):
        continue

    # load the image file
    wing_img = mpimg.imread(os.path.join(args["classifydir"][0], fname))

    # construct heatmap and generate the image mask
    heatmap = hmp.compute_heatmap(wing_img)

    #TODO: Add filtering
    hmp.generate_image_mask(heatmap, os.path.join(args["savedir"][0], fname))
