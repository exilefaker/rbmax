import jax.numpy as jnp
import pytest
import jax.random as jr
from rbmax.data_utils import get_batches_for_ensembles


def test_load_mnist(normalized_mnist_images):
    """
    Ensure MNIST train images are loaded and normalized correctly
    """
    M = 60000
    x = 28
    y = 28
    assert normalized_mnist_images.shape == (M, x*y)
    assert jnp.all(jnp.greater_equal(normalized_mnist_images, 0.0))
    assert jnp.all(jnp.less_equal(normalized_mnist_images, 1.0))


@pytest.mark.parametrize(
    "batch_size, ensembles", 
    [(1, 1), (14, 1), (16, 1), (1, 4), (16, 4)]
)
def test_get_batches(normalized_mnist_images, batch_size, ensembles):
    dataset_size = normalized_mnist_images.shape[0]
    x = 28
    y = 28
    key = jr.PRNGKey(0)
    M, remainder = jnp.divmod(dataset_size, batch_size)
    if remainder:
        with pytest.raises(ValueError):
            _ = get_batches_for_ensembles(key, normalized_mnist_images, batch_size, ensembles)
    else: 
        batches = get_batches_for_ensembles(key, normalized_mnist_images, batch_size, ensembles)
        assert len(batches) == M
        assert batches[0].shape == (ensembles, batch_size, x*y)
