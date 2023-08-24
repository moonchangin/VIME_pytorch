"""
Reference: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar, 
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain," 
Neural Information Processing Systems (NeurIPS), 2020.
Code author: Chang In Moon
-----------------------------

data_loader_pytorch.py
- Load and preprocess MNIST data (http://yann.lecun.com/exdb/mnist/) 
"""

import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms

def load_mnist_data(label_data_rate):
    """MNIST data loading for PyTorch.
  
    Args:
    - label_data_rate: ratio of labeled data
  
    Returns:
    - x_label, y_label: labeled dataset tensors
    - x_unlab: unlabeled dataset tensor
    - x_test, y_test: test dataset tensors
    """
    # Define transformation - normalization and conversion to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalizing to [-1, 1]
    ])
    
    # Download and load MNIST train dataset
    mnist_train = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    mnist_test = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    
    x_train = mnist_train.data.numpy() / 255.0
    y_train = mnist_train.targets.numpy()
    x_test = mnist_test.data.numpy() / 255.0
    y_test = mnist_test.targets.numpy()
    
    # One hot encoding for the labels
    y_train = np.asarray(pd.get_dummies(y_train))
    y_test = np.asarray(pd.get_dummies(y_test))

    # Shape
    no, dim_x, dim_y = x_train.shape
    test_no, _, _ = x_test.shape
  
    x_train = x_train.reshape(no, dim_x * dim_y)
    x_test = x_test.reshape(test_no, dim_x * dim_y)
  
    # Divide labeled and unlabeled data
    idx = np.random.permutation(len(y_train))
  
    # Label data : Unlabeled data = label_data_rate:(1-label_data_rate)
    label_idx = idx[:int(len(idx) * label_data_rate)]
    unlab_idx = idx[int(len(idx) * label_data_rate):]
  
    # Unlabeled data
    x_unlab = torch.from_numpy(x_train[unlab_idx, :]).float()
  
    # Labeled data
    x_label = torch.from_numpy(x_train[label_idx, :]).float()
    y_label = torch.from_numpy(y_train[label_idx, :]).float()
  
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()
  
    return x_label, y_label, x_unlab, x_test, y_test
