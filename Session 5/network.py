
import torch
import torch.nn as nn
import torch.nn.functional as func


criterion = nn.CrossEntropyLoss()

class nercLSTM(nn.Module):
   def __init__(self, codes) :
      super(nercLSTM, self).__init__()

      n_words = codes.get_n_words()
      n_sufs = codes.get_n_sufs()
      n_labels = codes.get_n_labels()
      
      self.embW = nn.Embedding(n_words, 100)
      self.embS = nn.Embedding(n_sufs, 50)      
      self.dropW = nn.Dropout(0.1)
      self.dropS = nn.Dropout(0.1)
      self.lstm = nn.LSTM(150, 200, bidirectional=True, batch_first=True)
      self.out = nn.Linear(400, n_labels)


   def forward(self, w, s):
      x = self.embW(w)
      y = self.embS(s)
      x = self.dropW(x)
      y = self.dropS(y)

      x = torch.cat((x, y), dim=2)
      x = self.lstm(x)[0]              
      x = self.out(x)
      return x
   
   


