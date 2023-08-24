"""
Reference: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar, 
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain," 
Neural Information Processing Systems (NeurIPS), 2020.
Code author: Chang In Moon
-----------------------------

vime_utils_pytorch.py
- Various utility functions for VIME framework adapted for PyTorch

(1) mask_generator: Generate mask vector for self and semi-supervised learning
(2) pretext_generator: Generate corrupted samples for self and semi-supervised learning
(3) perf_metric: prediction performances in terms of AUROC or accuracy
(4) convert_matrix_to_vector: Convert two dimensional matrix into one dimensional vector
(5) convert_vector_to_matrix: Convert one dimensional vector into one dimensional matrix
"""

# Necessary packages
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

def mask_generator(p_m, x):
    """Generate mask tensor in PyTorch.
    
    Args:
    - p_m: corruption probability
    - x: feature tensor
    
    Returns:
    - mask: binary mask tensor 
    """
    # torch.bernoulli is used to generate binary random numbers, 
    # torch.full is used to generate a tensor of the same size as x filled with p_m
    mask = torch.bernoulli(torch.full(x.shape, p_m)) 
    return mask

def pretext_generator(m, x):  
    """Generate corrupted samples in PyTorch.
  
    Args:
    m: mask tensor
    x: feature tensor
    
    Returns:
    m_new: final mask tensor after corruption
    x_tilde: corrupted feature tensor
    """
    # Parameters
    no, dim = x.shape  
    
    # Randomly (and column-wise) shuffle data
    x_bar = torch.zeros([no, dim])
    for i in range(dim):
        idx = torch.randperm(no)
        x_bar[:, i] = x[idx, i]
    
    # Corrupt samples
    x_tilde = x * (1-m) + x_bar * m  
    # Define new mask tensor
    m_new = 1 * (x != x_tilde)

    return m_new, x_tilde

def perf_metric(metric, y_test, y_test_hat):
    """Evaluate performance in PyTorch.
  
    Args:
    - metric: acc or auc
    - y_test: ground truth label tensor
    - y_test_hat: predicted values tensor
    
    Returns:
    - performance: Accuracy or AUROC performance
    """
    # Convert tensors to numpy arrays for sklearn metrics
    y_test = y_test.numpy()
    y_test_hat = y_test_hat.numpy()

    # Accuracy metric
    if metric == 'acc':
        result = accuracy_score(np.argmax(y_test, axis = 1), 
                                np.argmax(y_test_hat, axis = 1))
    # AUROC metric
    elif metric == 'auc':
        result = roc_auc_score(y_test[:, 1], y_test_hat[:, 1])      
    
    return result

def convert_matrix_to_vector(matrix):
    """Convert two dimensional tensor into one dimensional tensor in PyTorch
  
    Args:
    - matrix: two dimensional tensor
    
    Returns:
    - vector: one dimensional tensor
    """
    # Parameters
    no, dim = matrix.shape
    # Define output  
    vector = torch.zeros([no,], dtype=torch.float)
  
    # Convert matrix to vector
    for i in range(dim):
        idx = (matrix[:, i] == 1).nonzero()
        vector[idx] = i
    
    return vector

def convert_vector_to_matrix(vector):
    """Convert one dimensional tensor into two dimensional tensor in PyTorch
  
    Args:
    - vector: one dimensional tensor
    
    Returns:
    - matrix: two dimensional tensor
    """
    # Parameters
    no = len(vector)
    dim = len(torch.unique(vector))
    # Define output
    matrix = torch.zeros([no,dim])
  
    # Convert vector to matrix
    for i in range(dim):
        idx = (vector == i).nonzero()
        matrix[idx, i] = 1
    
    return matrix
