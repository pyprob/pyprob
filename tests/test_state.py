import unittest

import pyprob
from pyprob import state
from pyprob import util


class StateTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self._root_function_name = self.test_address.__code__.co_name
        super().__init__(*args, **kwargs)

    def _sample_address(self):
        address = state.extract_address(self._root_function_name)
        return address

    def test_address(self):
        address = self._sample_address()
        address_correct = '4/test_address.address'
        util.debug('address', 'address_correct')
        self.assertEqual(address, address_correct)


if __name__ == '__main__':
    unittest.main()
