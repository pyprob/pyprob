import numpy
import pytest
import pyprob

@pytest.fixture
def random():
    pyprob.seed(123)
