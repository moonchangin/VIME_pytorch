"""
Reference: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar, 
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain," 
Neural Information Processing Systems (NeurIPS), 2020.
Code author: Chang In Moon
-----------------------------

vime_self_pytorch.py
- Self-supervised learning parts of the VIME framework adapted for PyTorch
- Using unlabeled data to train the encoder
"""

import torch
import torch.nn as nn
import torch.optim as optim

from vime_utils_pytorch import mask_generator, pretext_generator

class VIMEModel(nn.Module):
    def __init__(self, dim):
        super(VIMEModel, self).__init__()
        # Encoder
        self.encoder = nn.Linear(dim, int(dim))
        # Mask estimator
        self.mask_estimator = nn.Linear(int(dim), dim)
        # Feature estimator
        self.feature_estimator = nn.Linear(int(dim), dim)
        
    def forward(self, x):
        # Encode the input
        hidden = torch.relu(self.encoder(x))
        # Estimate the mask
        mask_output = torch.sigmoid(self.mask_estimator(hidden))
        # Estimate the feature
        feature_output = torch.sigmoid(self.feature_estimator(hidden))
        # Return the mask and feature
        return mask_output, feature_output

def vime_self(x_unlab, p_m, alpha, parameters):
    # Parameters
    _, dim = x_unlab.shape # Get the number of dimensions of the data
    epochs = parameters['epochs']
    batch_size = parameters['batch_size']
    
    # Initialize model and optimizer
    model = VIMEModel(dim)
    optimizer = optim.RMSprop(model.parameters())
    
    # Losses
    criterion_mask = nn.BCELoss()
    criterion_feature = nn.MSELoss()
    
    # Generate mask m_unlab
    m_unlab = mask_generator(p_m, x_unlab)

    # Generate pretext task from the mask m_unlab and the unlabelled data x_unlab
    m_label, x_tilde = pretext_generator(m_unlab, x_unlab)
    
    # Convert numpy arrays to PyTorch tensors
    # x_tilde = torch.from_numpy(x_tilde).float()
    # m_label = torch.from_numpy(m_label).float()
    # x_unlab = torch.from_numpy(x_unlab).float()
    
    # Training loop
    for epoch in range(epochs):
        for i in range(0, len(x_unlab), batch_size):
            # Prepare minibatch
            inputs = x_tilde[i:i+batch_size]
            labels_mask = m_label[i:i+batch_size]
            labels_feature = x_unlab[i:i+batch_size]
            
            # Forward pass
            outputs_mask, outputs_feature = model(inputs)
            
            # Compute losses
            loss_mask = criterion_mask(outputs_mask, labels_mask)
            loss_feature = criterion_feature(outputs_feature, labels_feature)
            loss = loss_mask + alpha * loss_feature
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Return the encoder part of the model
    encoder = model.encoder
    return encoder
