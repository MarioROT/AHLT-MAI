import os
import string
import re
import torch
import spacy

nlp = spacy.load("en_core_web_sm")

from dataset import *

class Codemaps :
    # --- constructor, create mapper either from training data, or
    # --- loading codemaps from given file
    def __init__(self, data, maxlen=None, suflen=None, prelen=None) :
        
        self.external = {}
        with open("../resources/HSDB.txt", encoding="utf-8") as h:
            for x in h.readlines():
                self.external[x.strip().lower()] = "drug"
        with open("../resources/DrugBank.txt", encoding="utf-8") as h:
            for x in h.readlines():
                (n, t) = x.strip().lower().split("|")
                self.external[n] = t
        
        # Mapping entities to unique integers
        self.entity_to_index = {"NONE": 0, "drug": 1, "drug_n": 2, "brand": 3, "group": 4}


        if isinstance(data,Dataset) and maxlen is not None and suflen is not None and prelen is not None:
            self.__create_indexs(data, maxlen, suflen, prelen)

        elif type(data) == str and maxlen is None and suflen is None and suflen is None:
            self.__load(data)

        else:
            print('codemaps: Invalid or missing parameters in constructor')
            exit()


 

            
    # --------- Create indexs from training data
    # Extract all words and labels in given sentences and 
    # create indexes to encode them as numbers when needed
    def __create_indexs(self, data, maxlen, suflen, prelen) :

        self.maxlen = maxlen
        self.suflen = suflen
        self.prelen = prelen
        words = set([])
        sufs = set([])
        prefs = set([])
        pos_tags = set([])
        lemmas = set([])
        labels = set([])
        
        for s in data.sentences() :
            sent = []
            for t in s :
                words.add(t['form'])
                sufs.add(t['lc_form'][-self.suflen:])
                prefs.add(t['lc_form'][:self.prelen])
                labels.add(t['tag'])
                sent.append(t['form'])
            sent = " ".join(sent)
            doc = nlp(sent)

            for token in doc:
                pos_tags.add(token.pos_)
                lemmas.add(token.lemma_)
                # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                #         token.shape_, token.is_alpha, token.is_stop)

        self.word_index = {w: i+2 for i,w in enumerate(list(words))}
        self.word_index['PAD'] = 0 # Padding
        self.word_index['UNK'] = 1 # Unknown words

        self.suf_index = {s: i+2 for i,s in enumerate(list(sufs))}
        self.suf_index['PAD'] = 0  # Padding
        self.suf_index['UNK'] = 1  # Unknown suffixes
        
        self.pre_index = {s: i+2 for i,s in enumerate(list(prefs))}
        self.pre_index['PAD'] = 0  # Padding
        self.pre_index['UNK'] = 1  # Unknown prefixes
        
        self.pos_index = {s: i+2 for i,s in enumerate(list(pos_tags))}
        self.pos_index['PAD'] = 0  # Padding
        self.pos_index['UNK'] = 1  # Unknown pos
        
        self.lemma_index = {s: i+2 for i,s in enumerate(list(lemmas))}
        self.lemma_index['PAD'] = 0  # Padding
        self.lemma_index['UNK'] = 1  # Unknown suffixes

        self.label_index = {t: i+1 for i,t in enumerate(list(labels))}
        self.label_index['PAD'] = 0 # Padding
        
    ## --------- load indexs ----------- 
    def __load(self, name) : 
        self.maxlen = 0
        self.suflen = 0
        self.prelen = 0
        self.word_index = {}
        self.suf_index = {}
        self.pre_index = {}
        self.pos_index = {}
        self.lemma_index = {}
        self.label_index = {}

        with open(name+".idx") as f :
            for line in f.readlines(): 
                (t,k,i) = line.split()
                if t == 'MAXLEN' : self.maxlen = int(k)
                elif t == 'SUFLEN' : self.suflen = int(k)
                elif t == 'PRELEN' : self.prelen = int(k)                
                elif t == 'WORD': self.word_index[k] = int(i)
                elif t == 'SUF': self.suf_index[k] = int(i)
                elif t == 'PRE': self.pre_index[k] = int(i)
                elif t == 'POS': self.pos_index[k] = int(i)
                elif t == 'LEMMA': self.lemma_index[k] = int(i)
                elif t == 'LABEL': self.label_index[k] = int(i)
                            
    
    ## ---------- Save model and indexs ---------------
    def save(self, name) :
        # save indexes
        with open(name+".idx","w") as f :
            print ('MAXLEN', self.maxlen, "-", file=f)
            print ('SUFLEN', self.suflen, "-", file=f)
            print ('PRELEN', self.prelen, "-", file=f)
            for key in self.label_index : print('LABEL', key, self.label_index[key], file=f)
            for key in self.word_index : print('WORD', key, self.word_index[key], file=f)
            for key in self.suf_index : print('SUF', key, self.suf_index[key], file=f)
            for key in self.pre_index : print('PRE', key, self.pre_index[key], file=f)
            for key in self.pos_index : print('POS', key, self.pos_index[key], file=f)
            for key in self.lemma_index : print('LEMMA', key, self.lemma_index[key], file=f)
##############################################################################################

    ## --------- encode X from given data -----------
    # ORIGINAL 
    # def encode_words(self, data) :

    #     # encode sentence words
    #     enc = [torch.Tensor([self.word_index[w['form']] if w['form'] in self.word_index else self.word_index['UNK'] for w in s]) for s in data.sentences()]
    #     # cut sentences longer than maxlen
    #     enc = [s[0:self.maxlen] for s in enc]
    #     enc_lower = [s[0:self.maxlen] for s in enc_lower]
    #     # create a tensor full of padding
    #     tsr = torch.Tensor([])
    #     Xw = tsr.new_full((len(enc), self.maxlen), self.word_index['PAD'], dtype=torch.int64)
    #     Xl = torch.full((len(enc_lower), self.maxlen), self.word_index['PAD'], dtype=torch.int64)

    #     # fill padding tensor with sentence data
    #     for i, s in enumerate(enc): Xw[i, 0:s.size()[0]] = s

    #     # encode sentence suffixes
    #     enc = [torch.Tensor([self.suf_index[w['lc_form'][-self.suflen:]] if w['lc_form'][-self.suflen:] in self.suf_index else self.suf_index['UNK'] for w in s]) for s in data.sentences()]
    #     # cut sentences longer than maxlen
    #     enc = [s[0:self.maxlen] for s in enc]
    #     # create a tensor full of padding
    #     enc_words
    #     Xs = tsr.new_full((len(enc), self.maxlen), self.suf_index['PAD'], dtype=torch.int64)
    #     # fill padding tensor with sentence data
    #     for i, s in enumerate(enc): Xs[i, 0:s.size()[0]] = s

    #     # cut sentences longer than maxlen
    #     enc = [s[0:self.maxlen] for s in enc]
    #     # create a tensor full of zeros
    #     Xf = torch.zeros((len(enc), self.maxlen, 11), dtype=torch.int64)
    #     # fill padding tensor with sentence data
    #     for i, s in enumerate(enc):
    #         for j, f in enumerate(enc[i]) :
    #             Xf[i, j] = f

    #     # return encoded sequences
    #     return [Xl,Xw,Xs,Xf]
    #     # return [Xw,Xs]
##############################################################
    def get_external(self, word):
        return self.entity_to_index.get(self.external.get(word.lower(), 'NONE'))

    def encode_words(self, data):
        # encode sentence words (including lowercase)
        enc_words = [
            torch.Tensor([
                self.word_index[w['form']] if w['form'] in self.word_index else self.word_index['UNK'] 
                for w in s
            ]) for s in data.sentences()
        ]
        
        enc_lower = [
            torch.Tensor([
                self.word_index[w['form'].lower()] if w['form'].lower() in self.word_index else self.word_index['UNK']
                for w in s
            ]) for s in data.sentences()
        ]

        # truncate sentences longer than maxlen
        enc_words = [s[0:self.maxlen] for s in enc_words]
        enc_lower = [s[0:self.maxlen] for s in enc_lower]

        # Create padding tensors for words and lowercased words
        Xw = torch.full((len(enc_words), self.maxlen), self.word_index['PAD'], dtype=torch.int64)
        Xl = torch.full((len(enc_lower), self.maxlen), self.word_index['PAD'], dtype=torch.int64)
        for i, s in enumerate(enc_words):
            Xw[i, 0:len(s)] = s
        for i, s in enumerate(enc_lower):
            Xl[i, 0:len(s)] = s

        # encode sentence suffixes
        enc_sufs = [
            torch.Tensor([
                self.suf_index[w['lc_form'][-self.suflen:]] if w['lc_form'][-self.suflen:] in self.suf_index else self.suf_index['UNK']
                for w in s
            ]) for s in data.sentences()
        ]
        # truncate sentences longer than maxlen
        enc_sufs = [s[0:self.maxlen] for s in enc_sufs]
        # create a tensor full of padding
        Xs = torch.full((len(enc_sufs), self.maxlen), self.suf_index['PAD'], dtype=torch.int64)
        for i, s in enumerate(enc_sufs):
            Xs[i, 0:len(s)] = s
            
        # encode sentence prefixes
        enc_prefs = [
            torch.Tensor([
                self.pre_index[w['lc_form'][:self.prelen]] if w['lc_form'][:self.prelen] in self.pre_index else self.pre_index['UNK']
                for w in s
            ]) for s in data.sentences()
        ]
        # truncate sentences longer than maxlen
        enc_prefs = [s[0:self.maxlen] for s in enc_prefs]
        # create a tensor full of padding
        Xs2 = torch.full((len(enc_prefs), self.maxlen), self.pre_index['PAD'], dtype=torch.int64)
        for i, s in enumerate(enc_prefs):
            Xs2[i, 0:len(s)] = s
        enc_pos = []
        enc_lemma = []
        # encode sentence POS tags and lemmas
        for s in data.sentences():
            sent = [w['form'] for w in s]
            sent = " ".join(sent)
            doc = nlp(sent)
            sent_pos = []
            sent_lemma = []
            for token in doc:
                sent_pos.append(self.pos_index[token.pos_] if token.pos_ in self.pos_index else self.pos_index['UNK']) 
                sent_lemma.append(self.lemma_index[token.lemma_] if token.lemma_ in self.lemma_index else self.lemma_index['UNK']) 
            enc_pos.append(torch.Tensor(sent_pos))
            enc_lemma.append(torch.Tensor(sent_lemma))
        # truncate sentences longer than maxlen
        enc_pos = [s[0:self.maxlen] for s in enc_pos]
        # create a tensor full of padding
        XPos = torch.full((len(enc_pos), self.maxlen), self.pos_index['PAD'], dtype=torch.int64)
        for i, s in enumerate(enc_pos):
            XPos[i, 0:len(s)] = s
            
        enc_lemma = [s[0:self.maxlen] for s in enc_lemma]
        # create a tensor full of padding
        XLemma = torch.full((len(enc_lemma), self.maxlen), self.lemma_index['PAD'], dtype=torch.int64)
        for i, s in enumerate(enc_lemma):
            XLemma[i, 0:len(s)] = s

        # Additional features as separate tensors
        Xcap = torch.zeros((len(enc_words), self.maxlen), dtype=torch.int64)  # Capitalization
        Xdash = torch.zeros((len(enc_words), self.maxlen), dtype=torch.int64)  # Dashes
        Xnum = torch.zeros((len(enc_words), self.maxlen), dtype=torch.int64)  # Numbers
        Xext = torch.zeros((len(enc_words), self.maxlen), dtype=torch.int64)  # External presence
        Xspecial = torch.zeros((len(enc_words), self.maxlen), dtype=torch.int64)  # Special characters
        Xlen = torch.zeros((len(enc_words), self.maxlen), dtype=torch.int64)  # Length of words
        Xpos = torch.zeros((len(enc_words), self.maxlen), dtype=torch.int64)  # Position in sentence
        
        for i, s in enumerate(data.sentences()):
            for j, w in enumerate(s):
                if j < self.maxlen:
                    Xcap[i, j] = w['form'][0].isupper()  # Capitalization
                    Xdash[i, j] = '-' in w['form']  # Dashes
                    Xnum[i, j] = any(char.isdigit() for char in w['form'])  # Numbers
                    Xext[i, j] = self.get_external(w['form'])  # External presence as integer
                    Xspecial[i, j] = any(char in '%@#&$*' for char in w['form'])  # Special characters
                    Xlen[i, j] = len(w['form'])  # Length of word
                    Xpos[i, j] = j  # Position in sentence

        # Return the encoded sequences
        return [Xw, Xs, Xs2, Xl, XPos, XLemma, Xcap, Xdash, Xnum, Xext, Xspecial, Xlen, Xpos]


    
    ## --------- encode Y from given data ----------- 
    def encode_labels(self, data) :
        # encode and pad sentence labels
        enc = [torch.Tensor([self.label_index[w['tag']] for w in s]) for s in data.sentences()]
        # cut sentences longer than maxlen
        enc = [s[0:self.maxlen] for s in enc]
        # create a tensor full of padding
        tsr = torch.Tensor([])
        Y = tsr.new_full((len(enc), self.maxlen), self.label_index['PAD'], dtype=torch.int64)
        # fill padding tensor with sentence data
        for i, s in enumerate(enc): Y[i, 0:s.size()[0]] = s
               
        return Y

    ## -------- get word index size ---------
    def get_n_words(self) :
        return len(self.word_index)
    ## -------- get suf index size ---------
    def get_n_sufs(self) :
        return len(self.suf_index)
    ## -------- get pref index size ---------
    def get_n_prefs(self) :
        return len(self.pre_index)
    ## -------- get pos index size ---------
    def get_n_pos(self) :
        return len(self.pos_index)
    ## -------- get elmma index size ---------
    def get_n_lemmas(self) :
        return len(self.lemma_index)
    ## -------- get label index size ---------
    def get_n_labels(self) :
        return len(self.label_index)
    ## -------- get index for given word ---------
    def word2idx(self, w) :
        return self.word_index[w]
    ## -------- get index for given suffix --------
    def suff2idx(self, s) :
        return self.suff_index[s]
    ## -------- get index for given label --------
    def label2idx(self, l) :
        return self.label_index[l]
    ## -------- get label name for given index --------
    def idx2label(self, i) :
        for l in self.label_index :
            if self.label_index[l] == i:
                return l
        raise KeyError
#################################################################

# #####################################################
# # FOR bert
# import os
# import string
# import re
# import torch
# from transformers import BertTokenizer


# from dataset import *

# class Codemaps :
#     # --- constructor, create mapper either from training data, or
#     # --- loading codemaps from given file
#     def __init__(self, data, maxlen=None, suflen=None):
#         self.external = {}
#         with open("../resources/HSDB.txt", encoding="utf-8") as h:
#             for x in h.readlines():
#                 self.external[x.strip().lower()] = "drug"
#         with open("../resources/DrugBank.txt", encoding="utf-8") as h:
#             for x in h.readlines():
#                 (n, t) = x.strip().lower().split("|")
#                 self.external[n] = t

#         self.entity_to_index = {"NONE": 0, "drug": 1, "drug_n": 2, "brand": 3, "group": 4}

#         if isinstance(data, Dataset) and maxlen is not None and suflen is not None:
#             self.__create_indexs(data, maxlen, suflen)
#         elif type(data) == str and maxlen is None and suflen is None:
#             self.__load(data)
#         else:
#             print('codemaps: Invalid or missing parameters in constructor')
#             exit()

#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

 

            
#     # --------- Create indexs from training data
#     # Extract all words and labels in given sentences and 
#     # create indexes to encode them as numbers when needed
#     def __create_indexs(self, data, maxlen, suflen) :

#         self.maxlen = maxlen
#         self.suflen = suflen
#         words = set([])
#         sufs = set([])
#         labels = set([])
        
#         for s in data.sentences() :
#             for t in s :
#                 words.add(t['form'])
#                 sufs.add(t['lc_form'][-self.suflen:])
#                 labels.add(t['tag'])

#         self.word_index = {w: i+2 for i,w in enumerate(list(words))}
#         self.word_index['PAD'] = 0 # Padding
#         self.word_index['UNK'] = 1 # Unknown words

#         self.suf_index = {s: i+2 for i,s in enumerate(list(sufs))}
#         self.suf_index['PAD'] = 0  # Padding
#         self.suf_index['UNK'] = 1  # Unknown suffixes

#         self.label_index = {t: i+1 for i,t in enumerate(list(labels))}
#         self.label_index['PAD'] = 0 # Padding
        
#     ## --------- load indexs ----------- 
#     def __load(self, name) : 
#         self.maxlen = 0
#         self.suflen = 0
#         self.word_index = {}
#         self.suf_index = {}
#         self.label_index = {}

#         with open(name+".idx") as f :
#             for line in f.readlines(): 
#                 (t,k,i) = line.split()
#                 if t == 'MAXLEN' : self.maxlen = int(k)
#                 elif t == 'SUFLEN' : self.suflen = int(k)                
#                 elif t == 'WORD': self.word_index[k] = int(i)
#                 elif t == 'SUF': self.suf_index[k] = int(i)
#                 elif t == 'LABEL': self.label_index[k] = int(i)
                            
    
#     ## ---------- Save model and indexs ---------------
#     def save(self, name) :
#         # save indexes
#         with open(name+".idx","w") as f :
#             print ('MAXLEN', self.maxlen, "-", file=f)
#             print ('SUFLEN', self.suflen, "-", file=f)
#             for key in self.label_index : print('LABEL', key, self.label_index[key], file=f)
#             for key in self.word_index : print('WORD', key, self.word_index[key], file=f)
#             for key in self.suf_index : print('SUF', key, self.suf_index[key], file=f)


#     def get_external(self, word):
#         return self.entity_to_index.get(self.external.get(word.lower(), 'NONE'))


#     def encode_words(self, data):
#         sentences = [sentence for sentence in data.sentences()]
#         tokens = [self.tokenizer.tokenize(' '.join([w['form'] for w in s])) for s in sentences]
#         token_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in tokens]

#         # Truncate sentences longer than maxlen
#         token_ids = [t[:self.maxlen] for t in token_ids]

#         # Additional features as separate tensors
#         Xcap = torch.zeros((len(token_ids), self.maxlen), dtype=torch.int64)
#         Xdash = torch.zeros((len(token_ids), self.maxlen), dtype=torch.int64)
#         Xnum = torch.zeros((len(token_ids), self.maxlen), dtype=torch.int64)
#         Xext = torch.zeros((len(token_ids), self.maxlen), dtype=torch.int64)
#         Xspecial = torch.zeros((len(token_ids), self.maxlen), dtype=torch.int64)
#         Xlen = torch.zeros((len(token_ids), self.maxlen), dtype=torch.int64)
#         Xpos = torch.zeros((len(token_ids), self.maxlen), dtype=torch.int64)

#         for i, s in enumerate(sentences):
#             for j, w in enumerate(s):
#                 if j < self.maxlen:
#                     Xcap[i, j] = w['form'][0].isupper()  # Capitalization
#                     Xdash[i, j] = '-' in w['form']  # Dashes
#                     Xnum[i, j] = any(char.isdigit() for char in w['form'])  # Numbers
#                     Xext[i, j] = self.get_external(w['form'])  # External presence as integer
#                     Xspecial[i, j] = any(char in '%@#&$*' for char in w['form'])  # Special characters
#                     Xlen[i, j] = len(w['form'])  # Length of word
#                     Xpos[i, j] = j  # Position in sentence

#         # Convert to tensors and pad
#         token_ids = [torch.tensor(t) for t in token_ids]
#         token_ids = torch.nn.utils.rnn.pad_sequence(token_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

#         return [token_ids, Xcap, Xdash, Xnum, Xext, Xspecial, Xlen, Xpos]
#     ## --------- encode Y from given data ----------- 
#     def encode_labels(self, data) :
#         # encode and pad sentence labels
#         enc = [torch.Tensor([self.label_index[w['tag']] for w in s]) for s in data.sentences()]
#         # cut sentences longer than maxlen
#         enc = [s[0:self.maxlen] for s in enc]
#         # create a tensor full of padding
#         tsr = torch.Tensor([])
#         Y = tsr.new_full((len(enc), self.maxlen), self.label_index['PAD'], dtype=torch.int64)
#         # fill padding tensor with sentence data
#         for i, s in enumerate(enc): Y[i, 0:s.size()[0]] = s
               
#         return Y

#     ## -------- get word index size ---------
#     def get_n_words(self) :
#         return len(self.word_index)
#     ## -------- get suf index size ---------
#     def get_n_sufs(self) :
#         return len(self.suf_index)
#     ## -------- get label index size ---------
#     def get_n_labels(self) :
#         return len(self.label_index)

#     ## -------- get index for given word ---------
#     def word2idx(self, w) :
#         return self.word_index[w]
#     ## -------- get index for given suffix --------
#     def suff2idx(self, s) :
#         return self.suff_index[s]
#     ## -------- get index for given label --------
#     def label2idx(self, l) :
#         return self.label_index[l]
#     ## -------- get label name for given index --------
#     def idx2label(self, i) :
#         for l in self.label_index :
#             if self.label_index[l] == i:
#                 return l
#         raise KeyError


# import os
# import string
# import re
# import torch

# from dataset import *

# class Codemaps :
#     # --- constructor, create mapper either from training data, or
#     # --- loading codemaps from given file
#     def __init__(self, data, maxlen=None, suflen=None) :

#         if isinstance(data,Dataset) and maxlen is not None and suflen is not None:
#             self.__create_indexs(data, maxlen, suflen)

#         elif type(data) == str and maxlen is None and suflen is None:
#             self.__load(data)

#         else:
#             print('codemaps: Invalid or missing parameters in constructor')
#             exit()

            
#     # --------- Create indexs from training data
#     # Extract all words and labels in given sentences and 
#     # create indexes to encode them as numbers when needed
#     def __create_indexs(self, data, maxlen, suflen) :

#         self.maxlen = maxlen
#         self.suflen = suflen
#         words = set([])
#         sufs = set([])
#         labels = set([])
        
#         for s in data.sentences() :
#             for t in s :
#                 words.add(t['form'])
#                 sufs.add(t['lc_form'][-self.suflen:])
#                 labels.add(t['tag'])

#         self.word_index = {w: i+2 for i,w in enumerate(list(words))}
#         self.word_index['PAD'] = 0 # Padding
#         self.word_index['UNK'] = 1 # Unknown words

#         self.suf_index = {s: i+2 for i,s in enumerate(list(sufs))}
#         self.suf_index['PAD'] = 0  # Padding
#         self.suf_index['UNK'] = 1  # Unknown suffixes

#         self.label_index = {t: i+1 for i,t in enumerate(list(labels))}
#         self.label_index['PAD'] = 0 # Padding
        
#     ## --------- load indexs ----------- 
#     def __load(self, name) : 
#         self.maxlen = 0
#         self.suflen = 0
#         self.word_index = {}
#         self.suf_index = {}
#         self.label_index = {}

#         with open(name+".idx") as f :
#             for line in f.readlines(): 
#                 (t,k,i) = line.split()
#                 if t == 'MAXLEN' : self.maxlen = int(k)
#                 elif t == 'SUFLEN' : self.suflen = int(k)                
#                 elif t == 'WORD': self.word_index[k] = int(i)
#                 elif t == 'SUF': self.suf_index[k] = int(i)
#                 elif t == 'LABEL': self.label_index[k] = int(i)
                            
    
#     ## ---------- Save model and indexs ---------------
#     def save(self, name) :
#         # save indexes
#         with open(name+".idx","w") as f :
#             print ('MAXLEN', self.maxlen, "-", file=f)
#             print ('SUFLEN', self.suflen, "-", file=f)
#             for key in self.label_index : print('LABEL', key, self.label_index[key], file=f)
#             for key in self.word_index : print('WORD', key, self.word_index[key], file=f)
#             for key in self.suf_index : print('SUF', key, self.suf_index[key], file=f)


#     ## --------- encode X from given data ----------- 
#     def encode_words(self, data) :

#         # encode sentence words
#         enc = [torch.Tensor([self.word_index[w['form']] if w['form'] in self.word_index else self.word_index['UNK'] for w in s]) for s in data.sentences()]
#         # cut sentences longer than maxlen
#         enc = [s[0:self.maxlen] for s in enc]
#         # create a tensor full of padding
#         tsr = torch.Tensor([])
#         Xw = tsr.new_full((len(enc), self.maxlen), self.word_index['PAD'], dtype=torch.int64)
#         # fill padding tensor with sentence data
#         for i, s in enumerate(enc): Xw[i, 0:s.size()[0]] = s

#         # encode sentence suffixes
#         enc = [torch.Tensor([self.suf_index[w['lc_form'][-self.suflen:]] if w['lc_form'][-self.suflen:] in self.suf_index else self.suf_index['UNK'] for w in s]) for s in data.sentences()]
#         # cut sentences longer than maxlen
#         enc = [s[0:self.maxlen] for s in enc]
#         # create a tensor full of padding
#         tsr = torch.Tensor([])
#         Xs = tsr.new_full((len(enc), self.maxlen), self.suf_index['PAD'], dtype=torch.int64)
#         # fill padding tensor with sentence data
#         for i, s in enumerate(enc): Xs[i, 0:s.size()[0]] = s

#         # cut sentences longer than maxlen
#         enc = [s[0:self.maxlen] for s in enc]
#         # create a tensor full of zeros
#         Xf = torch.zeros((len(enc), self.maxlen, 11), dtype=torch.int64)
#         # fill padding tensor with sentence data
#         for i, s in enumerate(enc):
#             for j, f in enumerate(enc[i]) :
#                 Xf[i, j] = f

#         # return encoded sequences
#         #return [Xlw,Xw,Xs,Xf]
#         return [Xw,Xs]

    
#     ## --------- encode Y from given data ----------- 
#     def encode_labels(self, data) :
#         # encode and pad sentence labels
#         enc = [torch.Tensor([self.label_index[w['tag']] for w in s]) for s in data.sentences()]
#         # cut sentences longer than maxlen
#         enc = [s[0:self.maxlen] for s in enc]
#         # create a tensor full of padding
#         tsr = torch.Tensor([])
#         Y = tsr.new_full((len(enc), self.maxlen), self.label_index['PAD'], dtype=torch.int64)
#         # fill padding tensor with sentence data
#         for i, s in enumerate(enc): Y[i, 0:s.size()[0]] = s
               
#         return Y

#     ## -------- get word index size ---------
#     def get_n_words(self) :
#         return len(self.word_index)
#     ## -------- get suf index size ---------
#     def get_n_sufs(self) :
#         return len(self.suf_index)
#     ## -------- get label index size ---------
#     def get_n_labels(self) :
#         return len(self.label_index)

#     ## -------- get index for given word ---------
#     def word2idx(self, w) :
#         return self.word_index[w]
#     ## -------- get index for given suffix --------
#     def suff2idx(self, s) :
#         return self.suff_index[s]
#     ## -------- get index for given label --------
#     def label2idx(self, l) :
#         return self.label_index[l]
#     ## -------- get label name for given index --------
#     def idx2label(self, i) :
#         for l in self.label_index :
#             if self.label_index[l] == i:
#                 return l
#         raise KeyError
