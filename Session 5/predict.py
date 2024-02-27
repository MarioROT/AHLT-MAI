#! /usr/bin/python3

import sys
from os import system

import torch
from torch.utils.data import TensorDataset, DataLoader

from dataset import *
from codemaps import *


## --------- Entity extractor ----------- 
## -- Extract drug entities from given text and return them as
## -- a list of dictionaries with keys "offset", "text", and "type"

def output_entities(data, preds) :

   for sid,tags in zip(data.sentence_ids(),preds) :
      inside = False
      for k in range(0,min(len(data.get_sentence(sid)),codes.maxlen)) :
         y = tags[k]
         token = data.get_sentence(sid)[k]
            
         if (y[0]=="B") :
             entity_form = token['form']
             entity_start = token['start']
             entity_end = token['end']
             entity_type = y[2:]
             inside = True
         elif (y[0]=="I" and inside) :
             entity_form += " "+token['form']
             entity_end = token['end']
         elif (y[0]=="O" and inside) :
             print(sid, str(entity_start)+"-"+str(entity_end), entity_form, entity_type, sep="|")
             inside = False
        
      if inside : print(sid, str(entity_start)+"-"+str(entity_end), entity_form, entity_type, sep="|")


#----------------------------------------------
def encode_dataset(ds, codes) :
   X = codes.encode_words(ds)
   if torch.cuda.is_available() :
      X = [x.to(torch.device("cuda:0")) for x in X]
   return DataLoader(TensorDataset(*X), batch_size=32)

   
## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir
## --

fname = sys.argv[1]
datadir = sys.argv[2]

model = torch.load(fname+"/network.nn")
if torch.cuda.is_available() :
   model.to(torch.device("cuda:0"))
model.eval()
codes = Codemaps(fname+"/codemaps")

testdata = Dataset(datadir)
test_loader = encode_dataset(testdata, codes)

Y = []
for X in test_loader:
   y = model.forward(*X)
   z = [[codes.idx2label(torch.argmax(w)) for w in s] for s in y]
   Y.extend([[codes.idx2label(torch.argmax(w)) for w in s] for s in y] )

# extract & evaluate entities with basic model
output_entities(testdata, Y)

