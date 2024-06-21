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

import neptune
from neptune.types import File
from dotenv import dotenv_values


# random.seed(12345)
# torch.manual_seed(121212)

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
    #   if use_neptune:
    #       run['train/loss'].log(avg_loss)
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
    # if use_neptune:
    #     run['val/loss'].log(test_loss)
    #     run['val/accuracy'].log(100.*correct/total)

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

def load_glove_embeddings(file_path, word_index, embedding_dim=200):
    embeddings_index = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    embedding_matrix = np.zeros((len(word_index), embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  train.py ../data/Train ../data/Devel  modelname
## --

# directory with files to process
traindir = sys.argv[1]
validationdir = sys.argv[2]
n_epochs = int(sys.argv[3])
modelname = sys.argv[4]
# try:
#     use_neptune = sys.argv[5]
# except:
#     use_neptune = None

# if use_neptune:
#     config = dotenv_values("../.env")

#     run = neptune.init_run(
#         project="projects.mai.bcn/AHLT",
#         api_token=config['NPT_MAI_PB'],
#         tags=['NERC', use_neptune]
#     )  # your credentials

# load train and validation data
traindata = Dataset(traindir)
valdata = Dataset(validationdir)

# create indexes from training data
max_len = 150
suf_len = 5
pre_len = 5
codes  = Codemaps(traindata, max_len, suf_len, pre_len)

# encode datasets
train_loader = encode_dataset(traindata, codes)
val_loader = encode_dataset(valdata, codes)

# build network
network = nercLSTM(codes)
optimizer = optim.Adam(network.parameters())#, lr=0.003)
if torch.cuda.is_available() :
   network.to(torch.device("cuda:0"))

#summary(network)
   
for epoch in range(n_epochs):
   train(epoch)
   test()

# save model and indexs
os.makedirs(modelname,exist_ok=True)
torch.save(network, modelname+"/network.nn")
codes.save(modelname+"/codemaps")

