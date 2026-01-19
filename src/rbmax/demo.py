from rbmax.model import RBMEnsemble
from src.rbmax.data_utils import load_mnist_train


data, _ = load_mnist_train(use_labels=False)

rbms = RBMEnsemble(
    ensemble_size=1,
    data_size=784, # MNIST
    hidden_size=200,   
    init_data=data,
)

rbms = rbms.train_cd(
    data,
    batch_size=16,
    n=1,
    lr=1e-2,
    epochs=20
)
