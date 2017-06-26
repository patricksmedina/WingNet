# import packages
import matplotlib.pyplot as plt
import construct_heatmap as ch
import tensorflow as tf
import numpy as np
import progressbar
import timeit
import cv2
import os

# import modules
import wingnet_fullyconv as fc

# GLOBAL VARIABLES
NUM_CLASSES = 4
IMG_ROW = 1200
IMG_COL = 1600
IMG_CHN = 3
BB_ROW = 32
BB_COL = 32
BB_STRIDE = 5

# other variables
checkpoint_dir = "/noback/medinap/github/WingNet/tmp/train_logs_fc/"
wings_files_dir = "/noback/medinap/github/WingNet/test_data/"
# checkpoint_dir = "/Users/patrickmedina/Dropbox/WingNet/tmp/train_logs_fc/"
# wings_files_dir = "/Users/patrickmedina/Dropbox/D_Images/Wings/"
# wing_file_names = ["29652fem09","25174fem05", "25183fem15", "25186fem04", "25201fem14"]

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

# TENSORFLOW GRAPH
with tf.Graph().as_default() as g:
    # load and process image for classification
    x = tf.placeholder(np.float32, shape = [BB_ROW, BB_COL, IMG_CHN])
    # tf_image = tf.reshape(x, shape = [BB_ROW, BB_COL, IMG_CHN])
    # sm_image = tf.image.resize_image_with_crop_or_pad(tf_image, CL_COL, CL_ROW)
    sm_image = tf.image.rgb_to_grayscale(x) #(sm_image)
    sm_image = tf.image.per_image_standardization(sm_image)
    fd_image = tf.expand_dims(sm_image, 0)

    # perform inference on the image and
    # get softmax probabilities for each class
    logits = fc.inference(fd_image) #wingnet.inference(fd_image)
    softmax_probs = tf.nn.softmax(logits)

    saver = tf.train.Saver()
    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as sess:

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)

        for wing_name in os.listdir(wings_files_dir):

            if wing_name.startswith("."):
                continue

            print("[INFO] Computing probabilities across image " + wing_name)

            # start the timer.
            start = timeit.default_timer()

            # load the image
            sample_wing = os.path.join(wings_files_dir, wing_name)
            image = cv2.imread(sample_wing)
            probs = np.zeros((bb_points_dom.shape[0], NUM_CLASSES))

            for i, row in enumerate(bb_points_dom):
                temp_prob = sess.run(softmax_probs, feed_dict = {x: image[row[0]:(row[0] + BB_ROW), \
                                                                      row[1]:(row[1] + BB_COL), :]})
                probs[i, :] = temp_prob.reshape((1, NUM_CLASSES))

            # this total distance is not true near the boundaries, but should be zero there anyways
            print("[INFO] Constructing heatmap for " + wing_name)
            heatmap = ch.construct_heatmap(bb_points_dom, probs,
                                           IMG_ROW, IMG_COL, bb_points_dom.shape[0])

            heatmap = np.asarray(heatmap)

            plt.figure()
            mycntr = plt.contourf(pixel_points_col,
                                  pixel_points_row,
                                  np.flipud(heatmap),
                                  cmap = "RdBu_r",
                                  levels = np.linspace(0,
                                                       1,
                                                       31))
            plt.colorbar(mycntr, shrink = 0.9)
            plt.savefig("heatmap_{0}.png".format(wing_name), dpi = 300)

            # convert to greyscale image
            heatmap = np.round(heatmap * 255).astype(np.uint8)
            cv2.imwrite("wingmap_{0}.jpg".format(wing_name),
                        img = heatmap,
                        params = [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            end = timeit.default_timer() - start
            print("[INFO] Finished sample {0} in {1:.03f} seconds".format(wing_name, end))
