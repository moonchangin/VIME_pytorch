"""
Reference: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar, 
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain," 
Neural Information Processing Systems (NeurIPS), 2020.
Code author: Chang In Moon
-----------------------------

vime_semi_pytorch.py
- Semi-supervised learning parts of the VIME framework for PyTorch
- Using both labeled and unlabeled data to train the predictor with the help of trained encoder
"""

import torch
import torch.nn as nn
import torch.optim as optim
from vime_utils_pytorch import mask_generator, pretext_generator

class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

def vime_semi(x_train, y_train, x_unlab, x_test, parameters, 
              p_m, K, beta, encoder_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parameters
    hidden_dim = parameters['hidden_dim']
    batch_size = parameters['batch_size']
    iterations = parameters['iterations']
    data_dim = x_train.size(1)
    label_dim = y_train.size(1)

    # Convert numpy arrays to PyTorch tensors and move to device
    x_train, y_train, x_unlab, x_test = [torch.tensor(data).to(device).float() for data in [x_train, y_train, x_unlab, x_test]]

    # Model, criterion, and optimizer
    model = Predictor(data_dim, hidden_dim, label_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    for it in range(iterations):
        # Labeled data batch
        idx = torch.randint(0, len(x_train), (batch_size,))
        x_batch, y_batch = x_train[idx], y_train[idx]

        # Unlabeled data batch
        idx_unlab = torch.randint(0, len(x_unlab), (batch_size,))
        xu_batch_ori = x_unlab[idx_unlab]

        xu_batch_list = []
        for _ in range(K):
            m_batch = mask_generator(p_m, xu_batch_ori)
            _, xu_batch_temp = pretext_generator(m_batch, xu_batch_ori)
            xu_batch_temp = encoder_model(xu_batch_temp)
            xu_batch_list.append(xu_batch_temp)
        
        xu_batch = torch.stack(xu_batch_list, dim=1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        y_hat = model(x_batch)
        yu_hat = model(xu_batch)
        
        # Calculate the loss
        loss_supervised = criterion(y_hat, torch.max(y_batch, 1)[1])
        loss_unsupervised = -torch.mean(torch.var(yu_hat, 1))
        loss = loss_supervised + beta * loss_unsupervised

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if it % 100 == 0:
            print(f"Iteration {it}/{iterations}, Loss: {loss.item()}")

    # Testing
    with torch.no_grad():
        y_test_hat = model(x_test)
        
    return y_test_hat
