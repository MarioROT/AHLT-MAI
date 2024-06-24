import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

criterion = nn.CrossEntropyLoss()

## Atention Layers
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_dim]
        scores = torch.tanh(self.attention(lstm_output))  # [batch_size, seq_len, hidden_dim]
        scores = self.context_vector(scores).squeeze(-1)  # [batch_size, seq_len]
        weights = F.softmax(scores, dim=1)  # [batch_size, seq_len]
        weighted_output = lstm_output * weights.unsqueeze(-1)  # [batch_size, seq_len, hidden_dim]
        return weighted_output

class nercLSTM(nn.Module):
    def __init__(self, codes):
        super(nercLSTM, self).__init__()

        n_words = codes.get_n_words()
        n_sufs = codes.get_n_sufs()
        n_prefs = codes.get_n_prefs()
        n_pos = codes.get_n_pos()
        n_lemma = codes.get_n_lemma()
        n_labels = codes.get_n_labels()

        self.embW = nn.Embedding(n_words, 200)
        self.embS = nn.Embedding(n_sufs, 100)
        self.embP = nn.Embedding(n_prefs, 100)
        self.embPos = nn.Embedding(n_pos, 100)
        self.embLemma = nn.Embedding(n_lemma, 100)
        self.embL = nn.Embedding(n_words, 200)  # Embedding for lowercased words
        self.dropW = nn.Dropout(0.3)
        self.dropS = nn.Dropout(0.3)
        self.dropP = nn.Dropout(0.3)
        self.dropPos = nn.Dropout(0.3)
        self.dropLemma = nn.Dropout(0.3)
        self.dropL = nn.Dropout(0.3)  # Dropout for lowercased words

        additional_feature_dim = 7  # Xcap, Xdash, Xnum, Xext, Xspecial, Xlen, Xpos

        self.lstm_word = nn.LSTM(200, 200, bidirectional=True, batch_first=True)
        self.lstm_suf = nn.LSTM(100, 100, bidirectional=True, batch_first=True)
        self.lstm_pre = nn.LSTM(100, 100, bidirectional=True, batch_first=True)
        self.lstm_pos = nn.LSTM(100, 100, bidirectional=True, batch_first=True)
        self.lstm_lemma = nn.LSTM(100, 100, bidirectional=True, batch_first=True)
        self.lstm_lower = nn.LSTM(200, 200, bidirectional=True, batch_first=True)  # LSTM for lowercased words

        combined_input_size = 400 + 200 + 200 + 200 + 200 + 400 + additional_feature_dim  # Corrected combined input size
        #combined_input_size = 400 + 200 + 400 + additional_feature_dim  # Corrected combined input size

        self.lstm_combined = nn.LSTM(combined_input_size, 600, bidirectional=True, batch_first=True)

        self.attention = Attention(1200)
        
        self.fc1 = nn.Linear(1200, 600)
        self.fc2 = nn.Linear(600, 300)
        self.out = nn.Linear(300, n_labels)
        
        
    def forward(self, w, s, p, l, pos, lemma, Xcap, Xdash, Xnum, Xext, Xspecial, Xlen, Xpos):
        x = self.embW(w)
        x = self.dropW(x)
        x, _ = self.lstm_word(x)

        y = self.embS(s)
        y = self.dropS(y)
        y, _ = self.lstm_suf(y)
        
        y2 = self.embP(p)
        y2 = self.dropP(y2)
        y2, _ = self.lstm_pre(y2)

        z = self.embL(l)
        z = self.dropL(z)
        z, _ = self.lstm_lower(z)
        
        z1 = self.embL(pos)
        z1 = self.dropL(z1)
        z1, _ = self.lstm_lower(z1)
        
        z2 = self.embL(lemma)
        z2 = self.dropL(z2)
        z2, _ = self.lstm_lower(z2)

        combined_embeddings = torch.cat((x, y, y2, z, z1, z2), dim=2)
        #combined_embeddings = torch.cat((x, y, z), dim=2)

        additional_features = torch.stack((Xcap, Xdash, Xnum, Xext, Xspecial, Xlen, Xpos), dim=2).float()
        combined_features = torch.cat((combined_embeddings, additional_features), dim=2)

        combined_features, _ = self.lstm_combined(combined_features)

        # attention_output = self.attention(combined_features)  # Apply attention mechanism
        attention_output = combined_features
        
        fc_output = torch.relu(self.fc1(attention_output))
        fc_output = torch.relu(self.fc2(fc_output))

        output = self.out(fc_output)
        
        return output
