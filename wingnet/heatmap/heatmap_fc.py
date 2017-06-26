# import packages
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import progressbar
import timeit
import cv2
import os

# import modules
import wingnet_fullyconv

# GLOBAL VARIABLES
IMG_ROW = 1200
IMG_COL = 1600
IMG_CHN = 3
BB_ROW = 32
BB_COL = 32
BB_STRIDE = 5
CL_ROW = 24
CL_COL = 24

# other variables
checkpoint_dir = "/noback/medinap/github/WingNet/tmp/train_logs_fc/"
wings_files_dir = "/noback/medinap/github/WingNet/heatmap"
# checkpoint_dir = "/Users/patrickmedina/Dropbox/WingNet/tmp/train_logs_fc/"
# wings_files_dir = "/Users/patrickmedina/Dropbox/Drosophila_Wings/Wings/"
wing_file_names = ["28131mal06","29652fem09","29660mal11","29660mal16",
                   "25174fem05", "25183fem15", "25186fem04", "25201fem14",
                   "28254mal24", "28274mal13"]

#
# CONSTRUCT THE PIXEL DOMAIN AND THE BOUNDING BOX DOMAIN
#

pixel_points_row = np.arange(IMG_ROW)
pixel_points_col = np.arange(IMG_COL)
pixel_points_col, pixel_points_row = np.meshgrid(pixel_points_col,
                                                 pixel_points_row)
pixel_points_dom = np.hstack(( \
                        pixel_points_row.ravel().reshape(-1, 1),
                        pixel_points_col.ravel().reshape(-1, 1)\
                    ))

bb_points_row = np.arange(0, IMG_ROW - BB_ROW, BB_STRIDE)
bb_points_col = np.arange(0, IMG_COL - BB_COL, BB_STRIDE)
bb_points_row, bb_points_col = np.meshgrid(bb_points_row,
                                           bb_points_col)

bb_points_dom = np.hstack(( bb_points_row.ravel().reshape(-1, 1),
                            bb_points_col.ravel().reshape(-1, 1)))

# TENSORFLOW GRAPH
with tf.Graph().as_default() as g:
    # load and process image for classification
    x = tf.placeholder(np.float32, shape = [IMG_ROW, IMG_COL, IMG_CHN])
    sm_image = tf.image.rgb_to_grayscale(x)
    sm_image = tf.image.per_image_standardization(sm_image)
    fd_image = tf.expand_dims(sm_image, 0)

    # fully convolutional logits and softmax
    logits = wingnet_fullyconv.inference(fd_image)
    softmax_probs = tf.nn.softmax(logits)

    saver = tf.train.Saver()
    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)

        for wing_name in wing_file_names:
            print("[INFO] Constructing probability heatmap for sample " + wing_name)

            # start the timer.
            start = timeit.default_timer()

            # load the image
            sample_wing = os.path.join(wings_files_dir, wing_name)
            image = cv2.imread(sample_wing, -1)
            heatmap = sess.run(softmax_probs, feed_dict = {x: image})
            heatmap = heatmap[:,(1,3)].sum(axis=1)
            heatmap = heatmap.reshape((38, 50))
            end = timeit.default_timer() - start

            plt.figure()
            plt.imshow(heatmap)
            plt.show()
            print("[INFO] Finished sample {0} in {1:.03f} seconds".format(wing_name, end))
