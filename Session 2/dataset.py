import scipy
#from scipy.sparse import csr_matrix


#-------------------------------------------
# Class to handle a dataset made of sentences, where
# each sentence is a sequence of words, and each word
# is encoded as a list of (string) features
#-------------------------------------------
class Dataset :

    ## ------ Constructor. Load given datafile & index features.
    def __init__(self, datafile) :
        self.fidx = {}
        self.sentences = []
        with open(datafile) as df :
            for xseq, yseq, toks in self.__sequences(df):
                # load pair
                self.sentences.append((xseq,yseq,toks))
                # add features to index
                for w in xseq :
                    for f in w :
                        if f not in self.fidx :
                            self.fidx[f] = len(self.fidx)

    ## ------ auxilary for load. 
    def __sequences(self, fi):
        xseq = []
        yseq = []
        toks = []

        for line in fi:
            line = line.strip('\n')
            if not line:
                # An empty line means the end of a sentence.
                # Return accumulated sequences, and reinitialize.
                yield xseq, yseq, toks
                xseq = []
                yseq = []
                toks = []
                continue
            
            # Split the line with TAB characters.
            fields = line.split('\t')

            # Append the item features to the item sequence.
            # fields are:  0=sid, 1=form, 2=span_start, 3=span_end, 4=tag, 5...N = features
            toks.append(fields[:4]) # token info (sid, form, span)
            yseq.append(fields[4])  # label (ground truth)
            xseq.append(fields[5:]) # features

    ## ------ give access to feature index
    def feature_index(self) :
        return self.fidx

    ## ------ return dataset as a sparse matrix, plus associated gold labels
    def csr_matrix(self) :
        rowi = [] # row (example number)
        colj = [] # column (feature number)
        data = [] # value (1 or 0 since we use binary features)
        Y = [] # ground truth
        nex = 0 # example  counter (each word is one example)

        # for each sentence
        for xseq,yseq,_ in self.sentences :
            Y.extend(yseq)
            for w in xseq :
                for f in w :
                    data.append(1)
                    rowi.append(nex)
                    colj.append(self.fidx[f]) 
                # next word           
                nex += 1
            
        X = scipy.sparse.csr_matrix((data, (rowi, colj)))  
        return X,Y
                
    ## ------ iterator to access each sentence in the dataset
    def instances(self) :
        for x,y,z in self.sentences :
            yield x,y,z
