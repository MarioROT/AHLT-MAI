#! /usr/bin/python3

import sys, os

import torch
from torch.utils.data import TensorDataset, DataLoader

from dataset import *
from codemaps import *


## --------- Entity extractor ----------- 
## -- Extract drug entities from given text and return them as
## -- a list of dictionaries with keys "offset", "text", and "type"

def output_interactions(data, preds) :
   for exmp,tag in zip(data.sentences(),preds) :
      if tag!='null' :
         print(exmp['sid'], exmp['e1'], exmp['e2'], tag, sep="|")

#----------------------------------------------
def encode_dataset(ds, codes) :
   X = codes.encode_words(ds)
   if torch.cuda.is_available() :
      # load tensors to GPU if available
      X = [x.to(torch.device("cuda:0")) for x in X]
   # the data loader will allow access to data in batches
   return DataLoader(TensorDataset(*X), batch_size=16)

         
## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  predict.py modelname datafile.pck
## --
## -- Extracts DDI from all sentences in datafile
## --

modelname = sys.argv[1]
datafile = sys.argv[2]

# load network and its parameters from saved file
network = torch.load(modelname+"/network.nn")
if torch.cuda.is_available() :
   # load network to GPU if available
   network.to(torch.device("cuda:0"))
# set network in inference mode
network.eval()
# load code maps
codes = Codemaps(modelname+"/codemaps")

# encode datasets and load it in a Data loader
testdata = Dataset(datafile)
test_loader = encode_dataset(testdata,codes)

Y = []
# run each validation example and report validation loss
for X in test_loader:
   # X is a list of input tensors (no labels were loaded in the dataloader)
   y = network.forward(*X) # run example through the network
   # add results to result list
   Y.extend([codes.idx2label(torch.argmax(s)) for s in y])

# extract relations from result list
output_interactions(testdata, Y)



