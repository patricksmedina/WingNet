# import packages
import matplotlib.image as mpimg
import numpy as np
import argparse
import os

# import modules
from wingnet.tda import simplicial_complex as sc

# parse the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--imagedir', nargs=1, type=str, required=True,
                    help='Directory containing images to classify.')
parser.add_argument('-m', '--maskdir', nargs=1, type=str, required=True,
                    help='Directory containing the image masks.')
parser.add_argument('-s', '--savedir', nargs=1, type=str, required=True,
                    help='Directory where to save the image masks.')
# parse argument and convert to dictionary
args = vars(parser.parse_args())

for fname in os.listdir(args["imagedir"][0])[:5]:
    # skip annoying hidden files
    if fname.startswith("."):
        continue

    # load the image and the mask
    image = mpimg.imread(os.path.join(args["imagedir"][0], fname))
    mask = mpimg.imread(os.path.join(args["maskdir"][0],
                                     "{0}_mask.png".format(fname)))

    # convert to greyscale
    image = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
    mask = mask[:, :, 0]

    # compute the persistence diagram from the image and the mask.
    import timeit
    start = timeit.default_timer()
    pd = sc.compute_grid_diagram(image=image, mask=mask)
    pdfname = os.path.join(args["savedir"][0],"{0}_pd.csv".format(fname))
    np.savetxt(pdfname, pd.PD, delimiter=',')
    print("[INFO] Computed homology in {} seconds".format(start - timeit.default_timer()))
