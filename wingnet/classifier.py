# import packages
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import timeit
import cv2
import os

# import modules
import heatmap.heatmap as heatmap

# GLOBAL VARIABLES
NUM_CLASSES = 4
IMG_ROW = 1200
IMG_COL = 1600
IMG_CHN = 3
BB_ROW = 32
BB_COL = 32
BB_STRIDE = 5
