
import os
import string
import re
import torch

from dataset import *

class Codemaps :
    # --- constructor, create mapper either from training data, or
    # --- loading codemaps from given file
    def __init__(self, data, maxlen=None, suflen=None) :
        
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


        if isinstance(data,Dataset) and maxlen is not None and suflen is not None:
            self.__create_indexs(data, maxlen, suflen)

        elif type(data) == str and maxlen is None and suflen is None:
            self.__load(data)

        else:
            print('codemaps: Invalid or missing parameters in constructor')
            exit()


 

            
    # --------- Create indexs from training data
    # Extract all words and labels in given sentences and 
    # create indexes to encode them as numbers when needed
    def __create_indexs(self, data, maxlen, suflen) :

        self.maxlen = maxlen
        self.suflen = suflen
        words = set([])
        sufs = set([])
        labels = set([])
        
        for s in data.sentences() :
            for t in s :
                words.add(t['form'])
                sufs.add(t['lc_form'][-self.suflen:])
                labels.add(t['tag'])

        self.word_index = {w: i+2 for i,w in enumerate(list(words))}
        self.word_index['PAD'] = 0 # Padding
        self.word_index['UNK'] = 1 # Unknown words

        self.suf_index = {s: i+2 for i,s in enumerate(list(sufs))}
        self.suf_index['PAD'] = 0  # Padding
        self.suf_index['UNK'] = 1  # Unknown suffixes

        self.label_index = {t: i+1 for i,t in enumerate(list(labels))}
        self.label_index['PAD'] = 0 # Padding
        
    ## --------- load indexs ----------- 
    def __load(self, name) : 
        self.maxlen = 0
        self.suflen = 0
        self.word_index = {}
        self.suf_index = {}
        self.label_index = {}

        with open(name+".idx") as f :
            for line in f.readlines(): 
                (t,k,i) = line.split()
                if t == 'MAXLEN' : self.maxlen = int(k)
                elif t == 'SUFLEN' : self.suflen = int(k)                
                elif t == 'WORD': self.word_index[k] = int(i)
                elif t == 'SUF': self.suf_index[k] = int(i)
                elif t == 'LABEL': self.label_index[k] = int(i)
                            
    
    ## ---------- Save model and indexs ---------------
    def save(self, name) :
        # save indexes
        with open(name+".idx","w") as f :
            print ('MAXLEN', self.maxlen, "-", file=f)
            print ('SUFLEN', self.suflen, "-", file=f)
            for key in self.label_index : print('LABEL', key, self.label_index[key], file=f)
            for key in self.word_index : print('WORD', key, self.word_index[key], file=f)
            for key in self.suf_index : print('SUF', key, self.suf_index[key], file=f)


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
        return [Xw, Xs, Xl, Xcap, Xdash, Xnum, Xext, Xspecial, Xlen, Xpos]


    
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


