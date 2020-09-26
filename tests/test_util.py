import pytest
import torch
import numpy as np

from torchmtlr.utils import encode_survival


bins = np.arange(1, 5, dtype=np.float32)
testdata = [
    (3, 1, np.array([0, 0, 0, 1, 0]), bins),
    (2, 0, np.array([0, 0, 1, 1, 1]), bins)
]
@pytest.mark.parametrize("time,event,expected,bins", testdata)
def test_encode_survival(time, event, expected, bins):
    encoded = encode_survival(time, event, bins)
    assert np.allclose(encoded, expected)
