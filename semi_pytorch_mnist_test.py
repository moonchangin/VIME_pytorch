import torch
from torchvision import datasets, transforms
from data_loader_pytorch import load_mnist_data
from vime_self_pytorch import vime_self
from vime_semi_pytorch import vime_semi

# Parameters
p_m = 0.3
K = 3
alpha = 2.0
beta = 1.0
label_data_rate = 0.1
self_parameters = {'epochs': 5, 'batch_size': 64}
semi_parameters = {
    'hidden_dim': 100,
    'batch_size': 128,
    'iterations': 1000,
}

# Load and preprocess MNIST data
x_label, y_label, x_unlab, x_test, y_test = load_mnist_data(label_data_rate)  # 10% labeled data

# Train encoder using the self-supervised approach
encoder = vime_self(x_unlab, p_m, alpha, parameters=self_parameters)

encoder = encoder.eval()  # Set the encoder to evaluation mode

# Train the predictor model using the semi-supervised approach
y_test_hat = vime_semi(x_label, y_label, x_unlab, x_test, semi_parameters, p_m, K, beta, encoder)

# Evaluate the results
predicted_labels = torch.argmax(y_test_hat, axis=1)
true_labels = torch.argmax(y_test, axis=1)

accuracy = (predicted_labels == true_labels).sum().item() / len(true_labels)
print(f"Accuracy on test data: {accuracy * 100:.2f}%")
