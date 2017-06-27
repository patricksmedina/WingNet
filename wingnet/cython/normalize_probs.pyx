cimport numpy as cnp
import numpy as np

cpdef construct_heatmap(long[:, :] bb_points,
                        double[:, :] probs,
                        int img_row,
                        int img_col,
                        int num_probs):

    # allocate memory for the batch output
    cdef float[:, :] heatmap = np.zeros((img_row, img_col), dtype = np.float32)
    cdef float[:, :] counts = np.ones((img_row, img_col), dtype = np.float32)

    # copy the image bounding box region to the
    # batch array
    for i in range(num_probs):
        main_row, main_col = bb_points[i, 0], bb_points[i, 1]
        p1 = probs[i, 1] + probs[i, 3] # + probs[i, 5]

        for row in range(32):
            for col in range(32):
                heatmap[main_row + row, main_col + col] += (62 - col - row) * p1
                counts[main_row + row, main_col + col] += (col + row)

    for row in range(img_row):
        for col in range(img_col):
            heatmap[row, col] = heatmap[row, col] / counts[row, col]

    return(heatmap)
