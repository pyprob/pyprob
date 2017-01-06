# Tuan Anh Le (tuananh@robots.ox.ac.uk)
# January 2017
#
# Python script which takes a filename of a numeric, 2-column, CSV file
# containing 2d vectors. It ignores points outside of [-1, 1]x[-1, 1] and
# outputs a 100x100 histogram matrix of the remaining points in
# [edn](https://github.com/edn-format/edn) format.
#
# Example:
#
#   > python csv2hst.py mydata.csv
#   > <100x100 matrix in edn format>
#
# Dependencies:
#   - NumPy (http://www.numpy.org/)

import numpy as np
import sys

def is_in_box(x, lower_bounds, upper_bounds):
    num_dims = len(lower_bounds)
    for d in range(num_dims):
        if ((x[d] < lower_bounds[d]) or (x[d] > upper_bounds[d])):
            return False
    return True

def move_to_unit_box(lst):
    num_dims = len(lst[0])
    res = []
    for x in lst:
        if is_in_box(x, np.repeat(-1, num_dims), np.repeat(1, num_dims)):
            res.append((np.array(x) + 1) / 2)
    return np.array(res)

def get_grid_index(x, dims):
    res = [None] * len(dims)
    for i in range(len(dims)):
        res[i] = int(x[i] * dims[i])
        if res[i] == dims[i]:
            res[i] = res[i] - 1
    return tuple(res)

def get_grid_counts(lst, dims):
    res = np.zeros(dims)
    for x in lst:
        res[get_grid_index(x, dims)] += 1
    return res

def gridify(lst, dims):
    grid_counts = get_grid_counts(lst, dims)
    max_count = np.max(grid_counts)
    if max_count == 0:
        f = np.vectorize(float)
    else:
        f = np.vectorize(lambda grid_item: float(grid_item) / max_count)
    return f(grid_counts)

def main(argv):
    np.set_printoptions(threshold=np.nan, linewidth=np.nan, formatter={'float': lambda x: format(x, '.3f')})
    lst = np.genfromtxt(argv[0], delimiter=",")
    histogram = gridify(move_to_unit_box(lst), [100, 100])
    print(" ".join(
        np.array_str(histogram)
            .replace('\n', '')
            .split()
    ))

if __name__ == "__main__":
    main(sys.argv[1:])
