import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pyprob
import unittest

class PyProbTestCase(unittest.TestCase):
    def test_dummy(self):
        '''Dummy test'''
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
