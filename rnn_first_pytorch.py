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
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
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

class FirstSimpleModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(FirstSimpleModel, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True, nonlinearity='relu')  # to make it relu one should add nonelinearity='relu' 
        # Fully connected layer # If I understand correctly this is the layer that sits on top of RNN
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

    # first let us manipulate the datasets to extract the training data
    # and the testing data and put it in two separate datasets training dataset and 
    # testing dataset
#       divide_into_training_testing(input_dataset, fields_list, output dataset)

#raw_united_dataset = datadir + 'united_raw_dataset_384freez31f2_w_aug.hdf5'
raw_united_dataset = datadir + 'united_raw_dataset_384freez31f2_aug.hdf5'
with h5py.File(raw_united_dataset, 'r') as f:
    d_images = f["Raw_data/images_dataset"]
    d_labels = f["Raw_data/labels_dataset"]
    d_features = f["Raw_data/features_dataset"]
    d_features2 = f["Raw_data/features2_dataset"]
    d_matlab = f["Raw_data/matlab_dataset"]
    d_exclude=f["Raw_data/exclude_dataset"]
    d_shapes=f["Raw_data/shapes_dataset"]
    
    

    input_seq = torch.from_numpy(np.swapaxes(np.swapaxes(d_features2[:], 1, 2), 0, 1))
    target_seq = torch.Tensor(np.swapaxes(d_labels[:], 0, 1))


    
    # Instantiate the model with hyperparameters
    model = FirstSimpleModel(input_size=13, output_size=4, hidden_dim=5, n_layers=1)
    # We'll also set the model to the device that we defined earlier (default is CPU)
    model = model.to(device)
    
    # Define hyperparameters
    n_epochs = 1000
    lr=0.001
    
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
    
        loss = criterion(output, target_seq.reshape(-1).long())
        loss.backward() # Does backpropagation and calculates gradients
        optimizer.step() # Updates the weights accordingly
        
        if epoch%10 == 0:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))




