#!/usr/bin/env python3

import sys
from ML_model import *

datafile = sys.argv[1]
modelfile = sys.argv[2]


# load data to annotate
ds = Dataset(datafile)
# load trained model to use
model = ML_model(modelfile)
    
for xseq,_,toks in ds.instances():

    # process each sentence
    # each word has a list of features (xseq) for the prediction
    # plus positional info (toks) to format the output

    # get labels for each wor in the sentence
    predictions = model.predict(xseq)

    # extract identified drugs according to BIO tags for each word.
    inside = False;
    for k in range(len(predictions)) :
        y = predictions[k]
        (sid, form, offS, offE) = toks[k]

        if (y[0]=="B") :
            entity_form = form
            entity_start = offS
            entity_end = offE
            entity_type = y[2:]
            inside = True
        elif (y[0]=="I" and inside) :
            entity_form += " "+form
            entity_end = offE
        elif (y[0]=="O" and inside) :
            print(sid, entity_start+"-"+entity_end, entity_form, entity_type, sep="|")
            inside = False

    if inside : print(sid, entity_start+"-"+entity_end, entity_form, entity_type, sep="|")

