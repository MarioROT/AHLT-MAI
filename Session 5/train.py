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

from network import nercLSTM, criterion

random.seed(12345)
torch.manual_seed(121212)

#----------------------------------------------
def train(epoch):
   network.train()
   seen = 0
   acc_loss = 0
   for batch_idx, X in enumerate(train_loader):
      target = X.pop()
      optimizer.zero_grad()
      output = network(*X)
      output = output.flatten(0,1)
      target = target.flatten(0,1)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()
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
   print()

#----------------------------------------------
def test():
    network.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
       for X in val_loader:
          target = X.pop()
          output = network(*X)
          output = output.flatten(0,1)
          target = target.flatten(0,1)
          test_loss += criterion(output, target).item()
          pred = output.data.max(1, keepdim=True)[1]
          correct += pred.eq(target.data.view_as(pred)).sum()
          total += target.size()[0]
    test_loss /= len(val_loader)
    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
               test_loss,
               correct, total,
               100.*correct/total))

#----------------------------------------------
def encode_dataset(ds, codes) :
   X = codes.encode_words(ds)
   y = codes.encode_labels(ds)
   if torch.cuda.is_available() :
      #Xw = Xw.to(torch.device("cuda:0"))
      #Xs = Xs.to(torch.device("cuda:0"))
      X = [x.to(torch.device("cuda:0")) for x in X]
      y = y.to(torch.device("cuda:0"))
   return DataLoader(TensorDataset(*X, y), 
                     batch_size=16)



## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  train.py ../data/Train ../data/Devel  modelname
## --

# directory with files to process
traindir = sys.argv[1]
validationdir = sys.argv[2]
n_epochs = int(sys.argv[3])
modelname = sys.argv[4]

# load train and validation data
traindata = Dataset(traindir)
valdata = Dataset(validationdir)

# create indexes from training data
max_len = 150
suf_len = 5
codes  = Codemaps(traindata, max_len, suf_len)

# encode datasets
train_loader = encode_dataset(traindata, codes)
val_loader = encode_dataset(valdata, codes)

# build network
network = nercLSTM(codes)
optimizer = optim.Adam(network.parameters())
if torch.cuda.is_available() :
   network.to(torch.device("cuda:0"))

summary(network)
   
for epoch in range(n_epochs):
   train(epoch)
   test()

# save model and indexs
os.makedirs(modelname,exist_ok=True)
torch.save(network, modelname+"/network.nn")
codes.save(modelname+"/codemaps")

