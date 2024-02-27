
import torch
import torch.nn as nn
import torch.nn.functional as func

criterion = nn.CrossEntropyLoss()

class ddiCNN(nn.Module):
   
   def __init__(self, codes) :
      super(ddiCNN, self).__init__()
      # get sizes
      n_words = codes.get_n_words()
      n_labels = codes.get_n_labels()
      max_len = codes.maxlen
      # create layers
      self.embW = nn.Embedding(n_words, 100, padding_idx=0)
      self.cnn = nn.Conv1d(100, 32, kernel_size=3, stride=1, padding='same')
      self.out = nn.Linear(32*max_len, n_labels)

   def forward(self, w):
      # run layers on given data
      x = self.embW(w)
      x = x.permute(0,2,1)
      x = self.cnn(x)
      x = func.relu(x)
      x = x.flatten(start_dim=1)      
      x = self.out(x)
      return x
   


