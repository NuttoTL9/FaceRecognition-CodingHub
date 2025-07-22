import torch
import pytest

@pytest.fixture
def dummy_embedding():
    return torch.rand(1, 512)

@pytest.fixture
def dummy_image():
    import numpy as np
    return (np.ones((480, 640, 3)) * 255).astype('uint8')

@pytest.fixture
def dummy_box():
    return [50, 50, 200, 200]
