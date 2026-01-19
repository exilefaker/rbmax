import pytest
from rbmax.data_utils import load_mnist_train


@pytest.fixture
def normalized_mnist_images():
    images, _ = load_mnist_train(use_labels=False, normalize=True)
    return images

