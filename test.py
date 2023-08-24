# Assuming you've implemented the PyTorch version of vime_self.py and saved it as vime_self_pytorch.py

from vime_self_pytorch import vime_self
from data_loader_pytorch import load_mnist_data

# Parameters
label_data_rate = 0.1
p_m = 0.5
alpha = 1.0
parameters = {'epochs': 5, 'batch_size': 64}

# Load MNIST data
x_label, y_label, x_unlab, x_test, y_test = load_mnist_data(label_data_rate)

# Train encoder using self-supervised learning
encoder = vime_self(x_unlab, p_m, alpha, parameters)

# Just a simple check to see if the encoder has been trained and can produce outputs
sample = x_unlab[:5]
encoded_sample = encoder(sample)
print(encoded_sample)
