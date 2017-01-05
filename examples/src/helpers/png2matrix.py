# Tuan Anh Le
# January 2017
#
# Python script which takes a filename of an 8-bit PNG file and outputs a matrix
# in Clojure format
#
# Example:
#
#   > python png2matrix.py myfile.png
#   > [[1 2 3]
#      [4 5 6]
#      [7 8 9]]
#
# Dependencies:
#   - Pillow (http://python-pillow.org/)
#   - SciPy (https://www.scipy.org/)
#   - NumPy (http://www.numpy.org/)

import sys, numpy
from scipy import misc

def main(argv):
    numpy.set_printoptions(threshold=numpy.nan, linewidth=numpy.nan)
    print(" ".join(
        numpy.array_str(misc.imread(argv[0], mode='L'))
            .replace('\n', '')
            .split()
    ))

if __name__ == "__main__":
    main(sys.argv[1:])
