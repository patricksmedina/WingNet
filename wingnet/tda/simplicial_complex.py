# external packages
from wavetda import persistence_diagram as prd
import numpy as np

# load dionysus tools
from dionysus import *
from dionysus import Simplex
from dionysus import Filtration
from dionysus import StaticPersistence
from dionysus import vertex_cmp
from dionysus import data_cmp
from dionysus import data_dim_cmp
from dionysus import Rips
from dionysus import PairwiseDistances
from dionysus import points_file
from dionysus import ExplicitDistances

def levelset_filtration_2d(image, mask):
    """
    Prepare the level-set filtration for Dionysus
    """

    # store filtration points as a list
    filtration = []

    # used for cycling through function and vertices
    num_rows, num_cols = image.shape

    # enumerate the (row, col) coordinates and use as vertizes
    vertices = np.arange(num_rows * num_cols).reshape(num_rows, num_cols)

    for i in range(num_rows - 1):
        for j in range(num_cols - 1):
            p0 = vertices[i, j]
            p1 = vertices[i + 1, j]
            p2 = vertices[i, j + 1]
            p3 = vertices[i + 1, j + 1]

            # add simplices associated with the
            # feature indicated by the image mask

            # add vertices
            if mask[i, j] > 0:
                filtration.append([ (p0, ), image[i, j] ])

            # add 1-simplices
            if np.all([mask[i, j], mask[i+1, j]]) > 0:
                filtration.append([ (p0, p1),
                                    max(image[i, j], image[i+1, j]) ])

            if np.all([mask[i, j], mask[i, j+1]]) > 0:
                filtration.append([ (p0, p2), max(image[i,j], image[i, j+1]) ])

            if np.all([mask[i+1, j], mask[i, j+1]]) > 0:
                filtration.append([ (p1, p2),
                                    max(image[i+1,j], image[i, j+1]) ])

            # add 2-simplices
            if np.all([mask[i,j], mask[i+1,j], mask[i, j+1]]) > 0:
                filtration.append([(p0, p1, p2),
                                   max(image[i,j], image[i+1, j], image[i, j+1])
                                  ])

            if np.all([mask[i+1,j], mask[i,j+1],mask[i+1, j+1]]) > 0:
                filtration.append([ (p1, p2, p3),
                                max(image[i+1, j], image[i, j+1], image[i+1, j+1])])

            # handle boundary simplices
            if i == num_rows - 2:
                if mask[i+1,j] > 0:
                    filtration.append([ (p1, ), image[i+1, j] ])

                    if mask[i+1,j+1] > 0:
                        filtration.append([(p1, p3), max(image[i+1, j], image[i+1, j+1])])


            if j == num_cols - 2:
                if mask[i,j+1] > 0:
                    filtration.append([ (p2, ), image[i, j+1] ])

                    if mask[i+1,j+1] > 0:
                        filtration.append([ (p2, p3),
                                            max(image[i,j+1], image[i+1,j+1])
                                         ])


            if (i == num_rows - 2) and (j == num_cols - 2):
                if mask[i+1, j+1] > 0:
                    filtration.append([(p3, ), image[i+1, j+1]])

    return(filtration)

def persistence_diagram_grid(smap, p, max_death):
    """Constructs a persistence diagram object from persistence computed in Dionysus.

    Arguments:

        1. smap: Simplex map from a persistence object (see Dionysus)
        2. p: persistence object (see Dionysus)

    Returns:

        1. Numpy array containing the persistence diagram.
            * Column 1: Homology group
            * Column 2: Birth time
            * Column 3: Death time
    """

    # list to store persistence diagram points
    new_pd = []

    # add dimension, birth, death points to persistence diagram list
    for i in p:
        if i.sign():
            b = smap[i]

            # restrict persistence diagram to max_death point instead of infinity
            if i.unpaired():
                new_pd.append([b.dimension(), b.data, max_death])
                continue

            d = smap[i.pair()]

            # eliminate cases with zero persistence
            if b.data != d.data:
                new_pd.append([b.dimension(), b.data, min(d.data, max_death)])
                continue

    new_pd.sort()
    new_pd = np.asarray(new_pd).astype(np.float32)

    # remove features with zero persistence -- sometimes they slip through...
    new_pd = np.delete(new_pd, np.where(new_pd[:,2] - new_pd[:,1] == 0)[0], axis = 0)

    # sort by persistence within each homology group
    for hom in np.unique(new_pd[:,0]):
        # get the ids of the groups associated with this homology group
        hom_idx = np.where(new_pd[:,0] == hom)[0]
        temp_pd = new_pd[hom_idx, 1:]

        # get the sorted indices and sort the array
        sorted_idx = np.argsort(temp_pd[:, 1] - temp_pd[:, 0])[::-1]
        new_pd[hom_idx, 1:] = temp_pd[sorted_idx, ]

    return(new_pd)

def compute_grid_diagram(image, mask, max_death=255):
    """Workflow to construct a Persistence Diagram object from the level sets
    of the given function.

    Arguments
        1.

    Returns
        1.

    Raises
        1.

    """

    # construct list of times the simplicies are
    # added to the simplicial complex
    filtration = levelset_filtration_2d(image=image, mask=mask)

    # construct a simplex list for Dionysus
    scomplex = [Simplex(a, b) for (a,b) in filtration]

    # construct Dionysus filtration object
    filt = Filtration(scomplex, data_cmp)

    # compute persistent homology
    p = StaticPersistence(filt)
    p.pair_simplices(True)
    smap = p.make_simplex_map(filt)

    # generate numpy persistence diagram
    pd = persistence_diagram_grid(smap, p, max_death)

    return(prd.PersistenceDiagram(PD = pd))

if __name__ == "__main__":
    ### FOR TESTING PURPOSES ONLY
    # image = np.arange(36).reshape(6,6)
    # mask = (image > 13) * (image < 16)
    # mask += (image > 19) * (image < 22)
    # mask = 1 - mask * 1
    #
    # print(image)
    # print(mask)
    # print(mask * image)
    #
    # testPD = compute_grid_diagram(image, mask)
    # print(testPD.PD)
