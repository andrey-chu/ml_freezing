#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 16:51:57 2021

@author: andrey
"""
import platform
import numpy as np
import h5py
from itertools import compress


import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

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

model_path = '/home/andrey/python_programming/envs/freezing/ml_freezing/mymodel2.pt'
train_dataset = datadir + 'temp_training.hdf5'
test_dataset = datadir + 'temp_test.hdf5'

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
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True, nonlinearity='relu', bidirectional=True)  # to make it relu one should add nonelinearity='relu' 
        # Fully connected layer # If I understand correctly this is the layer that sits on top of RNN
        self.fc = nn.Linear(hidden_dim*2, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        #Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim*2)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(2*self.n_layers, batch_size, self.hidden_dim).to(device)
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden

    # first let us manipulate the datasets to extract the training data
    # and the testing data and put it in two separate datasets training dataset and 
    # testing dataset
#       divide_training_testing(input_dataset, fields_list, output dataset)

#raw_united_dataset = datadir + 'united_raw_dataset_384freez31f2_w_aug.hdf5'


    
with h5py.File(train_dataset, 'r') as f:
    #d_images = f["Raw_data/images_dataset"]
    d_labels = f["Raw_data/labels_dataset"]
    d_features2 = f["Raw_data/features2_dataset"]
    #d_features2 = f["Raw_data/features2_dataset"]
    #d_matlab = f["Raw_data/matlab_dataset"]
#    d_exclude=f["Raw_data/exclude_dataset"]
 #   d_shapes=f["Raw_data/shapes_dataset"]
    
    
    inp_size = 1
    X_train = d_features2[:,:,6:6+inp_size]
    Y_train = d_labels[:]
    
    # Let's first put 10% outside to check in the end
    
    

#    X_train = torch.from_numpy(X_train)

#    Y_train = torch.from_numpy(Y_train)

    # Instantiate the model with hyperparameters
    model = FirstSimpleModel(input_size=inp_size, output_size=4, hidden_dim=5, n_layers=1)
    # We'll also set the model to the device that we defined earlier (default is CPU)
    model = model.to(device)
    
    # Define hyperparameters
    n_epochs = 2000
    lr=0.005
    n_folds = 2
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    # Define Loss, Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    k = 0 
    
    for tr_idx, val_idx in kfold.split(X_train, Y_train):
        print('starting fold', k)
        k += 1
        print(6*'#', 'splitting and reshaping the data')
        train_input = X_train[tr_idx]
        print(train_input.shape)
        train_target = Y_train[tr_idx]
        val_input = X_train[val_idx]
        val_target = Y_train[val_idx]
        scalers = {}
        for i in range(train_input.shape[2]):
            scalers[i] = StandardScaler()
            train_input[:, :, i] = scalers[i].fit_transform(train_input[:, :, i]) 

        for i in range(val_input.shape[2]):
            val_input[:, :, i] = scalers[i].transform(val_input[:, :, i]) 
        val_input=torch.from_numpy(val_input)
        train_input=torch.from_numpy(train_input)
        val_target=torch.from_numpy(val_target)
        train_target=torch.from_numpy(train_target)
        train_input = train_input.to(device)    
        for epoch in range(1, n_epochs + 1):
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            #input_seq = input_seq.to(device)
            output, hidden = model(train_input)
            output = output.to(device)
            train_target = train_target.to(device)
        
            loss = criterion(output, train_target.reshape(-1).long())
            loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly
            
            if epoch%10 == 0:
                print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
                print("Loss: {:.4f}".format(loss.item()))
            
            
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, model_path)




# checkpoint = torch.load(model_path)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']


with h5py.File(test_dataset, 'r') as f:
    d_labels = f["Raw_data/labels_dataset"]
    d_features2 = f["Raw_data/features2_dataset"]
    test_seq = torch.from_numpy(d_features2[:,:,6:6+inp_size])
    test_seq=test_seq.to(device)
    test_target_seq = torch.Tensor(d_labels[:])
    output_test, hidden_test =model(test_seq)
    prob = nn.functional.softmax(output_test, dim=0).data
    # Taking the class with the highest probability score from the output
    test_result = torch.max(prob, dim=1)[1].reshape([test_seq.shape[0],test_seq.shape[1]])
res_np=test_result.numpy()
gt_np = test_target_seq.numpy()