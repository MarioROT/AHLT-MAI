#####################################################
## Class to store an ngram ME model
#####################################################
import sys
import dataset
import sys
import pickle

import scipy
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler

import dataset


class AB:


    ## --------------------------------------------------
    ## Constructor: Load model from file
    ## --------------------------------------------------
    def __init__(self, modelfile, params=None):

        self.modelfile = modelfile
        
        self.modelfile = modelfile
        
        if params is None :
            # modelfile given, assume it is an existing model and load it
            with open(modelfile, 'rb') as df :
                self.tagger = pickle.load(df)
            with open(modelfile+".idx", 'rb') as df :
                self.fidx = pickle.load(df)
        else :
            # no model file, create an empty model, waiting to be trained
            self.tagger = make_pipeline(AdaBoostClassifier(algorithm="SAMME"), 
                                        verbose=True)


    ## --------------------------------------------------
    ## train a model on given data, store in modelfile
    ## --------------------------------------------------
    def train(self, datafile):
        # load dataset
        ds = dataset.Dataset(datafile)
        self.fidx = ds.feature_index()

        # Read training instances 
        X,Y = ds.csr_matrix()

        # train classifier
        self.tagger.fit(X,Y)

        # save model
        pickle.dump(self.tagger, open(self.modelfile, 'wb'))
        pickle.dump(self.fidx, open(self.modelfile+".idx", 'wb'))
    

    ## --------------------------------------------------
    ## predict best class for each element in xseq
    ## --------------------------------------------------
    def predict(self, x):
        # Encode x into a CSR sparse matrix
        rowi = [] # row (example number)
        colj = [] # column (feature number)
        data = [] # value (1 or 0 since we use binary features)
        for f in x :
            if f in self.fidx :
                data.append(1)
                rowi.append(0) # we are predicting a single example
                colj.append(self.fidx[f])
                
        X = scipy.sparse.csr_matrix((data, (rowi, colj)), shape=(1,len(self.fidx)))
        
        # apply model to X and return predictions
        return self.tagger.predict(X)