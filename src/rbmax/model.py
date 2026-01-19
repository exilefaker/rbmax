import jax.numpy as jnp
import equinox as eqx
import jax
import jax.random as jr
import rich
from typing import Tuple
from functools import partial
from jax import lax
from jaxtyping import Array, Key
from rbmax.data_utils import get_batches_for_ensembles


def hidden_probs(v: Array, weights: Array, h_biases: Array):
    return jax.nn.sigmoid(
        jnp.einsum("ebi,eij->ebj", v, weights) + h_biases[:, None, :]
    )

def visible_probs(h: Array, weights: Array, v_biases: Array):
    return jax.nn.sigmoid(
        jnp.einsum("ebj,eij->ebi", h, weights) + v_biases[:, None, :]
    )

def gibbs_sample_iter(ph: Array, key: Key, weights: Array, bh: Array, bv: Array):
    h = jr.bernoulli(key, p=ph)
    pv = visible_probs(h, weights, bv)
    ph = hidden_probs(pv, weights, bh)
    
    return ph, pv


def cd_n(
    key: Key,
    data: Array,
    weights: Array,
    visible_biases: Array,
    hidden_biases: Array,
    n: int=1,
    lr: float=1e-2,
):
    """
    NOTE
    Data is of shape (ensemble_dim, batch_dim, ) + (data_dim,)
    Hidden layer is of shape (ensemble_dim, batch_dim, ) + (hidden_dim,)

    Assumes data has been flattened
    """
    keys = jr.split(key, n)
    M = data.shape[1]

    # ----------------- Obtain statistics -------------------

    # Positive statistics
    ph_data = hidden_probs(data, weights, hidden_biases)
    vh_data = jnp.einsum("ebi,ebj->eij", data, ph_data) / M

    # Gibbs chain
    scan_fn = partial(
        gibbs_sample_iter, 
        weights=weights, 
        bh=hidden_biases, 
        bv=visible_biases
    )
    ph_recon, pvs = lax.scan(scan_fn, ph_data, keys, length=n)

    # Negative statistics
    pv_recon = pvs[-1]
    vh_recon = jnp.einsum("ebi,ebj->eij", pv_recon, ph_recon) / M

    # ---------------- Gradient updates ---------------------

    weight_grad = (vh_data - vh_recon)
    bv_grad = (data - pv_recon).mean(axis=1)
    bh_grad = (ph_data - ph_recon).mean(axis=1)

    weights = weights + lr * weight_grad
    visible_biases = visible_biases + lr * bv_grad
    hidden_biases = hidden_biases + lr * bh_grad

    # ---------------------- Energy -------------------------

    # E = -(data.dot(visible_biases)) -(ph_data.dot(hidden_biases)) -(data @ weights @ ph_data)
    E_bv = -jnp.einsum("ebi,ei->e", data, visible_biases)
    E_bh = -jnp.einsum("ebj,ej->e", ph_data, hidden_biases)
    E_vh = -jnp.einsum("ebi,eij,ebj->e", data, weights, ph_data)
    E = E_bv + E_bh + E_vh

    return (weights, visible_biases, hidden_biases), E


def _cd_n_minibatch_scan_fn(carry: Tuple[Array], xs: Tuple[Array], n: int, lr: float):
    weights, visible_biases, hidden_biases = carry
    key, batch = xs
    return cd_n(key, batch, weights, visible_biases, hidden_biases, n, lr)


@partial(jax.jit, static_argnums=(5,6,7))
def cd_n_scan(keys, batches, weights, visible_biases, hidden_biases, n, M, lr):
    scan_fn = partial(_cd_n_minibatch_scan_fn, n=n, lr=lr)
    (weights, visible_biases, hidden_biases), E_hist = lax.scan(
        scan_fn,
        (weights, visible_biases, hidden_biases),
        (keys, batches),
        length=M
    )
    return weights, visible_biases, hidden_biases, E_hist


class RBMEnsemble(eqx.Module):
    weights: Array        # (ensemble size, visible size, hidden, size)
    visible_biases: Array # (ensemble size, visible size)
    hidden_biases: Array  # (ensemble size, hidden size)
    ensemble_size: int

    def __init__(self, ensemble_size: int=1, data_size:int=784, hidden_size:int=200, init_data:Array|None=None):
        key = jr.PRNGKey(0)
        self.weights = jr.normal(key, shape=(ensemble_size, data_size, hidden_size)) * 1e-2

        # If data is supplied, use it to initialize the visible biases
        if init_data is not None:
            data_mean = init_data.mean(axis=0).clip(1e-12, 1 - 1e-7)
            init_biases = jnp.log(data_mean / (1 - data_mean))
            self.visible_biases = jnp.broadcast_to(
                init_biases,
                (ensemble_size,) + init_biases.shape
            )
        else:
            self.visible_biases = jnp.zeros((ensemble_size, data_size))

        self.hidden_biases = jnp.zeros((ensemble_size, hidden_size))
        self.ensemble_size = ensemble_size
    
    @eqx.filter_jit
    def encode(self, data: Array) -> Array:
        ph = hidden_probs(data, self.weights, self.hidden_biases)
        _, key = jr.split(jr.PRNGKey(0), 2)
        return jr.bernoulli(key, p=ph)
    
    @eqx.filter_jit
    def generate(self, hidden_state: Array) -> Array:
        return visible_probs(hidden_state, self.weights, self.visible_biases)

    def train_cd(self, data: Array, batch_size: int=16, n: int=1, lr: float=1e-2, epochs:int=20) -> "RBMEnsemble":
        ensemble_size = self.weights.shape[0]

        base_key = jr.PRNGKey(0)

        weights, visible_biases, hidden_biases = self.weights, self.visible_biases, self.hidden_biases

        for e in jnp.arange(epochs):
            base_key, key = jr.split(base_key, 2)
            batches = get_batches_for_ensembles(key, data, batch_size, ensemble_size)
            M = len(batches)
            keys = jr.split(base_key, M)

            weights, visible_biases, hidden_biases, E_hist = cd_n_scan(
                keys, batches, weights, visible_biases, hidden_biases, n, M, lr
            )
            rich.print(f" Epoch {e}, Energy {E_hist.mean(axis=0)}")
        
        return eqx.tree_at(
            where=lambda rbm: (rbm.weights, rbm.visible_biases, rbm.hidden_biases),
            pytree=self,
            replace=(weights, visible_biases, hidden_biases)
        )
