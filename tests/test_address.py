import unittest

import pyprob
from pyprob import state
from pyprob import util


class TestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self._root_function_name = self.test_address.__code__.co_name
        super().__init__(*args, **kwargs)

    def _sample_address(self):
        address = state.extract_address(self._root_function_name)
        return address

    def test_address(self):
        address = self._sample_address()
        correct_address = '4/test_address.address'
        util.debug('address', 'correct_address')
        self.assertEqual(address, '4/test_address.address')
