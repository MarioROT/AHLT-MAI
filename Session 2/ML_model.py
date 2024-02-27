
import sys, os

from CRF import *
from LR import *

class ML_model:

    ## --------------------------------------------------
    ## Constructor: Load model from file
    ## --------------------------------------------------
    def __init__(self, modelfile):
        # if the model file exists, assume it is a trained model and load it
        # if it doesn't exist, create an empty model of the appropriate kind
        
        ext = modelfile[-4:]
        load = os.path.isfile(modelfile)
        
        if ext == ".crf" :
            if load : self._model = CRF(modelfile)
            else : self._model = CRF()
            
        elif ext == ".lrg" :
            if load : self._model = LR(modelfile)
            else : self._model = LR()
            
        else :
            print("Unknown model type",ext[1:])
            sys.exit(1)

        

    ## --------------------------------------------------
    ## Call trainer on a data file, save model
    ## --------------------------------------------------            
    def train(self, datafile, modelfile) :
        return self._model.train(datafile, modelfile)
            
    ## --------------------------------------------------
    ## Call predictor on a sequence
    ## --------------------------------------------------            
    def predict(self, xseq) :
        return self._model.predict(xseq)


