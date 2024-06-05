
import torch
import torch.nn as nn
import torch.nn.functional as func


criterion = nn.CrossEntropyLoss()

# class nercLSTM(nn.Module):
#    def __init__(self, codes) :
#       super(nercLSTM, self).__init__()

#       n_words = codes.get_n_words()
#       n_sufs = codes.get_n_sufs()
#       n_labels = codes.get_n_labels()
      
#       self.embW = nn.Embedding(n_words, 100)
#       self.embS = nn.Embedding(n_sufs, 50)      
#       self.dropW = nn.Dropout(0.1)
#       self.dropS = nn.Dropout(0.1)
#       self.lstm = nn.LSTM(150, 200, bidirectional=True, batch_first=True)
#       self.out = nn.Linear(400, n_labels)


#    def forward(self, w, s):
#       x = self.embW(w)
#       y = self.embS(s)
#       x = self.dropW(x)
#       y = self.dropS(y)

#       x = torch.cat((x, y), dim=2)
#       x = self.lstm(x)[0]              
#       x = self.out(x)
#       return x
   


# class nercLSTM(nn.Module):
#    def __init__(self, codes):
#       super(nercLSTM, self).__init__()

#       n_words = codes.get_n_words()
#       n_sufs = codes.get_n_sufs()
#       n_labels = codes.get_n_labels()
      
#       self.embW = nn.Embedding(n_words, 100)
#       self.embS = nn.Embedding(n_sufs, 50)
#       self.dropW = nn.Dropout(0.1)
#       self.dropS = nn.Dropout(0.1)

#       self.lstm1 = nn.LSTM(150, 200, bidirectional=True, batch_first=True)
#       self.lstm2 = nn.LSTM(400, 200, bidirectional=True, batch_first=True)
      
#       self.out = nn.Linear(400, n_labels)

#    def forward(self, w, s):
#       x = self.embW(w)
#       y = self.embS(s)
#       x = self.dropW(x)
#       y = self.dropS(y)

#       x = torch.cat((x, y), dim=2)
#       x, _ = self.lstm1(x)  # First LSTM layer
#       x, _ = self.lstm2(x)  # Second LSTM layer
#       x = self.out(x)
#       return x


class nercLSTM(nn.Module):
    def __init__(self, codes):
        super(nercLSTM, self).__init__()

        n_words = codes.get_n_words()
        n_sufs = codes.get_n_sufs()
        n_labels = codes.get_n_labels()
        
        # Embeddings
        self.embW = nn.Embedding(n_words, 100)  # Main words
        self.embS = nn.Embedding(n_sufs, 50)    # Suffixes
        self.embL = nn.Embedding(n_words, 100)  # Lowercase words
        
        # Dropout
        self.dropW = nn.Dropout(0.1)
        self.dropS = nn.Dropout(0.1)
        self.dropL = nn.Dropout(0.1)

        # LSTM Layers
        self.lstm1 = nn.LSTM(404, 200, bidirectional=True, batch_first=True)  # Updated input size
        self.lstm2 = nn.LSTM(400, 200, bidirectional=True, batch_first=True)
        
        # Output layer
        self.out = nn.Linear(400, n_labels)

    def forward(self, Xw, Xs, Xl, Xcap, Xdash, Xnum):
        xw = self.dropW(self.embW(Xw))    # Embed and apply dropout to main words
        xs = self.dropS(self.embS(Xs))    # Embed and apply dropout to suffixes
        xl = self.dropL(self.embL(Xl))    # Embed and apply dropout to lowercase words

        # Stack all features along the feature dimension
        x = torch.cat((xw, xs, xl, Xcap.unsqueeze(2), Xdash.unsqueeze(2), Xnum.unsqueeze(2)), dim=2)
        
        x, _ = self.lstm1(x)  # First LSTM layer
        x, _ = self.lstm2(x)  # Second LSTM layer
        x = self.out(x)       # Output layer
        
        return x


###########################################################################

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class nercLSTM(nn.Module):
    def __init__(self, codes):
        super(nercLSTM, self).__init__()

        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze BERT to not train it

        n_sufs = codes.get_n_sufs()
        n_labels = codes.get_n_labels()
        
        # Embeddings for suffixes and lowercase (unchanged)
        self.embS = nn.Embedding(n_sufs, 50)    # Suffixes
        self.embL = nn.Embedding(codes.get_n_words(), 100)  # Lowercase words
        
        # Dropout
        self.dropW = nn.Dropout(0.1)
        self.dropS = nn.Dropout(0.1)
        self.dropL = nn.Dropout(0.1)

        # LSTM Layers
        # Update input size considering BERT hidden size (768) + other features
        self.lstm1 = nn.LSTM(818, 200, bidirectional=True, batch_first=True) 
        self.lstm2 = nn.LSTM(400, 200, bidirectional=True, batch_first=True)
        
        # Output layer
        self.out = nn.Linear(400, n_labels)

    def forward(self, Xw, Xs, Xl, Xcap, Xdash, Xnum):
        # Getting BERT embeddings
        with torch.no_grad():  # Ensure BERT is in inference mode
            xw = self.bert(Xw).last_hidden_state
        
        xs = self.dropS(self.embS(Xs))  # Embed and apply dropout to suffixes
        xl = self.dropL(self.embL(Xl))  # Embed and apply dropout to lowercase words

        # Stack all features along the feature dimension
        x = torch.cat((xw, xs, xl, Xcap.unsqueeze(2), Xdash.unsqueeze(2), Xnum.unsqueeze(2)), dim=2)
        
        x, _ = self.lstm1(x)  # First LSTM layer
        x, _ = self.lstm2(x)  # Second LSTM layer
        x = self.out(x)       # Output layer
        
        return x
