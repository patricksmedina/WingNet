# external packages
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
    num_rows, num_cols, num_chn = f.shape

    # index values for vertizes
    vertices = np.arange(x * y).reshape(x,y)

    for i in range(x - 1):
        for j in range(y - 1):
            p0 = vertices[i, j]
            p1 = vertices[i + 1, j]
            p2 = vertices[i, j + 1]
            p3 = vertices[i + 1, j + 1]

            # add vertices
            filtration.append([
                                (p0, ),
                                temp_f[i, j]
                             ])

            # add 1-simplices
            filtration.append([
                                (p0, p1),
                                max(temp_f[i,j], temp_f[i+1, j])
                             ])

            filtration.append([
                                (p0, p2),
                                max(temp_f[i,j], temp_f[i, j+1])
                             ])

            filtration.append([
                                (p1, p2),
                                max(temp_f[i+1,j], temp_f[i, j+1])
                             ])


            # add 2-simplices
            filtration.append([
                                (p0, p1, p2),
                                max(temp_f[i,j], temp_f[i+1, j], temp_f[i, j+1])
                             ])

            filtration.append([
                                (p1, p2, p3),
                                max(temp_f[i + 1, j], temp_f[i, j + 1], temp_f[i + 1, j+1])
                             ])

            # add boundary simplices
            if i == x - 2:
                filtration.append([
                                    (p1, ),
                                    temp_f[i + 1, j]
                                 ])
                filtration.append([
                                    (p1, p3),
                                    max(temp_f[i + 1, j], temp_f[i + 1, j+1])
                                 ])

            if j == y - 2:
                filtration.append([
                                    (p2, ),
                                    temp_f[i, j + 1]
                                 ])
                filtration.append([
                                    (p2, p3),
                                    max(temp_f[i, j + 1], temp_f[i + 1, j+1])
                                 ])

            if (i == x - 2) and (j == y - 2):
                filtration.append([
                                    (p3, ),
                                    temp_f[i + 1, j + 1]
                                 ])

    return(filtration)
