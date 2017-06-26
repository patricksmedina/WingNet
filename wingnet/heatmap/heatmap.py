# import packages
import constructer as ch
import tensorflow as tf
import numpy as np

# import modules
from skimage import io
from models import wingnet_fullyconv as fc

# GLOBAL VARIABLES
NUM_CLASSES = 4
IMG_ROW = 1200
IMG_COL = 1600
IMG_CHN = 3
BB_ROW = 32
BB_COL = 32
BB_STRIDE = 5

def compute_heatmap(wing_name):
    """
    Constructs the probability heatmap for a given wing image.

    Returns numpy array of weighted probabilities of a pixel being a feature.
    """

    global NUM_CLASSES, IMG_ROW, IMG_COL, IMG_CHN, BB_COL, BB_ROW, BB_STRIDE

    #
    # CONSTRUCT THE PIXEL DOMAIN AND THE BOUNDING BOX DOMAIN
    #

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
    image = cv2.imread(wing_name)
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
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)

                for i, row in enumerate(bb_points_dom):
                    temp_prob = sess.run(softmax_probs, feed_dict = {x: image[row[0]:(row[0] + BB_ROW), \
                                                                          row[1]:(row[1] + BB_COL), :]})
                    probs[i, :] = temp_prob.reshape((1, NUM_CLASSES))

                # this total distance is not true near the boundaries, but should be zero there anyways
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
    cv2.imwrite("{0}_mask.jpg".format(wing_name),
                img = heatmap,
                params = [int(cv2.IMWRITE_JPEG_QUALITY), 100])
