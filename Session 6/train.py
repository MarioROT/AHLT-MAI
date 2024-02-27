#! /usr/bin/python3

import sys, os
import random

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torchinfo import summary

from dataset import *
from codemaps import *

from network import ddiCNN, criterion

random.seed(12345)
torch.manual_seed(121212)

#----------------------------------------------
def train(epoch):
    # set network in learn mode
   network.train()
   seen = 0
   acc_loss = 0
   # get training data in batches
   for batch_idx, X in enumerate(train_loader):
      # X is a list of input tensors plus the list of labels for each example 
      target = X.pop()  # the last one are the labels, separate them      
      # compute output of the network with current parameters
      optimizer.zero_grad()
      output = network(*X)
      # compute loss (distance betwen produced output and target)
      loss = criterion(output, target)
      # perform backpropagation (update network parameters to get closer to target)
      loss.backward()
      optimizer.step()
      # print progress statistics
      acc_loss += loss.item()
      avg_loss = acc_loss/(batch_idx+1)
      seen += len(target)
      print('Train Epoch {}: batch {}/{} [{}/{} ({:.2f}%)]   Loss: {:.6f}\r'.format(
                   epoch,
                   batch_idx+1, len(train_loader),
                   seen, len(train_loader.dataset),
                   100.*(batch_idx+1)/len(train_loader),
                   avg_loss),
            flush=True, end='')
   # newline
   print()

#----------------------------------------------
def validation():
    # set network in inference mode
    network.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
       # run each validation example and report validation loss
       for X in val_loader:
          # X is a list of input tensors plus the list of labels for each example 
          target = X.pop()  # the last one are the labels, separate them
          # compute network prediction 
          output = network(*X)
          # accumulate loss and accuracy statistics
          test_loss += criterion(output, target).item()
          pred = output.data.argmax(1)
          targ = target.data.argmax(1)
          correct += pred.eq(targ.data.view_as(pred)).sum()
          total += target.size()[0]
          
    # report validation statistics
    test_loss /= len(val_loader)
    print('Validation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
          test_loss,
          correct, total,
          100.*correct/total))


#----------------------------------------------
def encode_dataset(ds, codes) :
   X = codes.encode_words(ds) # X is a list of input tensors for each example
   y = codes.encode_labels(ds) # y are the target labels for each example
   if torch.cuda.is_available() :
      # load tensors to GPU if available
      X = [x.to(torch.device("cuda:0")) for x in X]
      y = y.to(torch.device("cuda:0"))
   # the data loader will allow access to data in batches
   return DataLoader(TensorDataset(*X, y), batch_size=16)


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  train.py train.pck devel.pck n_epochs modelname
## --

# pickle files with datasets to use
trainfile = sys.argv[1]
validationfile = sys.argv[2]
# number of training epochs
n_epochs = int(sys.argv[3])
# name under which the learned model will be saved
modelname = sys.argv[4]

# load train and validation data
traindata = Dataset(trainfile)
valdata = Dataset(validationfile)

# create indexes from training data
max_len = 150
codes = Codemaps(traindata, max_len)

# encode datasets and load them in a Data loader
train_loader = encode_dataset(traindata, codes)
val_loader = encode_dataset(valdata, codes)

# build network
network = ddiCNN(codes)
optimizer = optim.Adam(network.parameters())
if torch.cuda.is_available() :
   # load network into GPU if available
   network.to(torch.device("cuda:0"))

# print summary of network layers
summary(network)

# perform training, computing validation stats at each epoch
for epoch in range(n_epochs):
   train(epoch)
   validation()

# save model and indexs
os.makedirs(modelname,exist_ok=True)
torch.save(network, modelname+"/network.nn")
codes.save(modelname+"/codemaps")

