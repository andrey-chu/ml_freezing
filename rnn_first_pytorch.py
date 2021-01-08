#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 16:51:57 2021

@author: andrey
"""
import platform
import numpy as np
import h5py


import torch
from torch import nn
"""
Let us load the data first
We  need the raw features and the output
"""
if platform.node()=='choo-desktop':
    from branch_init_choo import datadir
elif platform.node()=='andrey-cfin':
    from branch_init_cfin import datadir
elif platform.node()=='andrey-workbook':
    from branch_init_laptop import datadir





# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        #Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden


raw_united_dataset = datadir + 'united_raw_dataset_384freez31f2_w_aug.hdf5'
with h5py.File(raw_united_dataset, 'r') as f:
    d_images = f["Raw_data/images_dataset"]
    d_labels = f["Raw_data/labels_dataset"]
    d_features = f["Raw_data/features_dataset"]
    d_features2 = f["Raw_data/features2_dataset"]
    d_matlab = f["Raw_data/matlab_dataset"]
    d_exclude=f["Raw_data/exclude_dataset"]
    d_shapes=f["Raw_data/shapes_dataset"]
    
    
    

    input_seq = torch.from_numpy(d_features2[:])
    target_seq = torch.Tensor(d_labels[:])


    
    # Instantiate the model with hyperparameters
    model = Model(input_size=13, output_size=1, hidden_dim=12, n_layers=1)
    # We'll also set the model to the device that we defined earlier (default is CPU)
    model = model.to(device)
    
    # Define hyperparameters
    n_epochs = 100
    lr=0.01
    
    # Define Loss, Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    input_seq = input_seq.to(device)
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad() # Clears existing gradients from previous epoch
        #input_seq = input_seq.to(device)
        output, hidden = model(input_seq)
        output = output.to(device)
        target_seq = target_seq.to(device)
        loss = criterion(output, target_seq.view(-1).long())
        loss.backward() # Does backpropagation and calculates gradients
        optimizer.step() # Updates the weights accordingly
        
        if epoch%10 == 0:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))




