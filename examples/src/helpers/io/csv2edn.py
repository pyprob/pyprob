# Tuan Anh Le (tuananh@robots.ox.ac.uk)
# January 2017
#
# Python script which takes a filename of a numeric CSV file and outputs a
# matrix in [edn](https://github.com/edn-format/edn) format.
#
# Example:
#
#   > python csv2edn.py myfile.csv
#   > [[1 2 3]
#      [4 5 6]
#      [7 8 9]]
#
# Dependencies:
#   - NumPy (http://www.numpy.org/)

import sys, numpy

def main(argv):
    numpy.set_printoptions(threshold=numpy.nan, linewidth=numpy.nan, formatter={'float': lambda x: format(x, '.3f')})
    print(" ".join(
        numpy.array_str(numpy.genfromtxt (argv[0], delimiter=","))
            .replace('\n', '')
            .split()
    ))

if __name__ == "__main__":
    main(sys.argv[1:])
