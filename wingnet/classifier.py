# import packages
import tensorflow as tf
import numpy as np
import argparse
import cv2
import os

# import modules
import heatmap.heatmap as hmp

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
parser.add_argument('-c', '--classifydir', type=str,
                    help='Directory containing images to classify.')
parser.add_argument('-s', '--savedir', type=str,
                    help='Directory where to save the image masks.')
args = parser.parse_args()

print(args)
