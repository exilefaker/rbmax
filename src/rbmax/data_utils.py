import struct
import numpy as np
import jax.numpy as jnp

from pathlib import Path
from array import array
from jaxtyping import Array, Key
import jax.random as jr


def load_mnist_train(use_labels: bool = True, normalize:bool=False):

    root_dir = Path(__file__).parent.parent.parent
    images_fn = root_dir / 'data/MNIST/train-images-idx3-ubyte'
    labels_fn = root_dir / 'data/MNIST/train-labels-idx1-ubyte'

    labels = None

    with open(images_fn,'rb') as f:
        _magic_num, N, rows, cols = struct.unpack(">IIII", f.read(16))
        img = array("B", f.read())

        assert N == 60000, "MNIST dataset shape doesn't match expected shape"

    if use_labels:
        with open(labels_fn, 'rb') as g:
            _magic_num, _size = struct.unpack(">II", g.read(8))
            lbl = array("b", g.read())

    ind = jnp.arange(N)

    # images = jnp.zeros((N, rows*cols), dtype=jnp.int8)

    images = jnp.array(img, dtype=jnp.uint8).reshape(N, rows*cols)

    if use_labels:
        labels = jnp.array(lbl, dtype=jnp.uint8).reshape(N)
        # labels = jnp.zeros((N), dtype=jnp.int8)

    # for i in range(len(ind)):
    #     images = images.at[i].set(jnp.array(img[ind[i]*rows*cols:(ind[i]+1)*rows*cols]))
    #     if use_labels:
    #         labels[i] = lbl[ind[i]]

    if normalize:
        images /= 256.0

    return images, labels


def denormalize_mnist_image(raw_data: Array) -> Array:
    return np.array(raw_data * 256).reshape(28,28).clip(0,255).astype(jnp.uint8)


def get_batches_for_ensembles(key: Key, data: Array, batch_size: int=16, ensemble_size:int=1):
    data_size = data.shape[0]
    keys = jr.split(key, ensemble_size)

    # Require minibatch size evenly divides the dataset, for now
    M, r = jnp.divmod(data_size, batch_size)
    if r:
        raise ValueError("Please use a minibatch size that divides the dataset size evenly")

    permutations = jnp.vstack([jr.permutation(keys[i], data_size) for i in range(ensemble_size)])
    batches = np.split(data.take(permutations, axis=0), data_size // batch_size, axis=1)

    return jnp.array(batches)
