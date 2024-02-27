#####################################################
## Class to store an ngram ME model
#####################################################
import sys
import pickle

import scipy
import sklearn
from sklearn.linear_model import LogisticRegression

import dataset


class LR:

    ## --------------------------------------------------
    ## Constructor: Load model from file
    ## --------------------------------------------------
    def __init__(self, modelfile=None):
          
        if modelfile is not None :
            # modelfile given, assume it is an existing model and load it
            with open(modelfile, 'rb') as df :
                self.tagger = pickle.load(df)
            with open(modelfile+".idx", 'rb') as df :
                self.fidx = pickle.load(df)
        else :
            # no model file, create an empty model, waiting to be trained
            self.tagger = None

    ## --------------------------------------------------
    ## train a model on given data, store in modelfile
    ## --------------------------------------------------
    def train(self, datafile, modelfile):
        # load dataset
        ds = dataset.Dataset(datafile)
        self.fidx = ds.feature_index()

        # Read training instances 
        X,Y = ds.csr_matrix()

        # create and train classifier
        self.tagger = LogisticRegression(solver="lbfgs",
                                         verbose=1,
                                         max_iter=200,
                                         n_jobs=8)
        self.tagger.fit(X,Y)

        # save model
        pickle.dump(self.tagger, open(modelfile, 'wb'))
        pickle.dump(self.fidx, open(modelfile+".idx", 'wb'))
    

    ## --------------------------------------------------
    ## predict best class for each element in xseq
    ## --------------------------------------------------
    def predict(self, xseq):
        if len(xseq)==0 : return []
        
        # Encode xseq into a CSR sparse matrix
        rowi = [] # row (example number)
        colj = [] # column (feature number)
        data = [] # value (1 or 0 since we use binary features)
        nex = 0 # example  counter (each word is one example)
        for w in xseq :
            for f in w :
                if f in self.fidx :
                    data.append(1)
                    rowi.append(nex)
                    colj.append(self.fidx[f]) 
                    # next word           
            nex += 1
        X = scipy.sparse.csr_matrix((data, (rowi, colj)), shape=(len(xseq),len(self.fidx)))
        
        # apply model to X and return predictions
        return self.tagger.predict(X)
