#!/usr/bin/env python3

import os,sys
from ML_model import *

# get file where model will be written
datafile = sys.argv[1]
modelfile = sys.argv[2]
    
# make sure the model file does not exist,
# so ML_model creates an empty one
if os.path.isfile(modelfile): os.remove(modelfile)
model = ML_model(modelfile)

# Train and store the model
model.train(datafile, modelfile)
