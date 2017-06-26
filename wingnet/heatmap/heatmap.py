# import packages
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import os

# import modules
from wingnet.heatmap.models import wingnet_fullyconv as fc
from wingnet.heatmap import normalize_probs as ch

# GLOBAL VARIABLES
BB_ROW = 32
BB_COL = 32
BB_STRIDE = 5
NUM_CLASSES = 4
CHECKPOINT_DIR = "/models/model_logs/"


def compute_heatmap(wing_img):
    """
    Constructs the probability heatmap for a given wing image.

    Returns numpy array of weighted probabilities of a pixel being a feature.
    """

    global NUM_CLASSES, BB_COL, BB_ROW, BB_STRIDE, CHECKPOINT_DIR

    #
    # CONSTRUCT THE PIXEL DOMAIN AND THE BOUNDING BOX DOMAIN
    #

    IMG_ROW, IMG_COL, IMG_CHN = wing_img.shape

    pixel_points_row = np.arange(IMG_ROW)
    pixel_points_col = np.arange(IMG_COL)
    pixel_points_col, pixel_points_row = np.meshgrid(pixel_points_col,
                                                     pixel_points_row)
    pixel_points_dom = np.hstack((pixel_points_row.ravel().reshape(-1, 1),
                                  pixel_points_col.ravel().reshape(-1, 1)
                                ))

    bb_points_row = np.arange(0, IMG_ROW - BB_ROW, BB_STRIDE)
    bb_points_col = np.arange(0, IMG_COL - BB_COL, BB_STRIDE)
    bb_points_row, bb_points_col = np.meshgrid(bb_points_row,
                                               bb_points_col)

    bb_points_dom = np.hstack(( bb_points_row.ravel().reshape(-1, 1),
                                bb_points_col.ravel().reshape(-1, 1)))

    # load the image and allocate space for the probabilities array
    probs = np.zeros((bb_points_dom.shape[0], NUM_CLASSES))

    #
    # TENSORFLOW GRAPH
    #

    with tf.Graph().as_default() as g:
        # load and process image for classification
        x = tf.placeholder(np.float32, shape = [BB_ROW, BB_COL, IMG_CHN])
        sm_image = tf.image.rgb_to_grayscale(x)
        sm_image = tf.image.per_image_standardization(sm_image)
        fd_image = tf.expand_dims(sm_image, 0)

        # perform inference on the image and
        # get softmax probabilities for each class
        logits = fc.inference(fd_image)
        softmax_probs = tf.nn.softmax(logits)

        saver = tf.train.Saver()
        tf.logging.set_verbosity(tf.logging.ERROR)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__) + CHECKPOINT_DIR)
            print(ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)

                for i, row in enumerate(bb_points_dom):
                    temp_prob = sess.run(softmax_probs,
                                         feed_dict = {x: wing_img[row[0]:(row[0] + BB_ROW), \
                                                                  row[1]:(row[1] + BB_COL), :]})

                    probs[i, :] = temp_prob.reshape((1, NUM_CLASSES))

    # reweight the probabilities based off of distance to a central point
    heatmap = ch.construct_heatmap(bb_points_dom,
                                   probs,
                                   IMG_ROW,
                                   IMG_COL,
                                   bb_points_dom.shape[0])

    return(np.flipud(heatmap))

def generate_image_mask(heatmap, wing_name):
    # convert to greyscale image
    heatmap = np.round(heatmap * 255).astype(np.uint8)

    # save the image mask
    mpimg.imsave(fname="{0}_mask.png".format(wing_name),
                 arr=heatmap,
                 cmap="Greys_r")
