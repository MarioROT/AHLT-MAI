import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

criterion = nn.CrossEntropyLoss()


## Original network
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

################################################################
## Bigger network with more inputs
class nercLSTM(nn.Module):
    def __init__(self, codes):
        super(nercLSTM, self).__init__()

        n_words = codes.get_n_words()
        n_sufs = codes.get_n_sufs()
        n_labels = codes.get_n_labels()

        self.embW = nn.Embedding(n_words, 200)
        self.embS = nn.Embedding(n_sufs, 100)
        self.embL = nn.Embedding(n_words, 200)  # Embedding for lowercased words
        self.dropW = nn.Dropout(0.3)
        self.dropS = nn.Dropout(0.3)
        self.dropL = nn.Dropout(0.3)  # Dropout for lowercased words

        additional_feature_dim = 7  # Xcap, Xdash, Xnum, Xext, Xspecial, Xlen, Xpos

        self.lstm_word = nn.LSTM(200, 200, bidirectional=True, batch_first=True)
        self.lstm_suf = nn.LSTM(100, 100, bidirectional=True, batch_first=True)
        self.lstm_lower = nn.LSTM(200, 200, bidirectional=True, batch_first=True)  # LSTM for lowercased words

        combined_input_size = 400 + 200 + 400 + additional_feature_dim  # Corrected combined input size
        self.lstm_combined = nn.LSTM(combined_input_size, 400, bidirectional=True, batch_first=True)

        self.fc1 = nn.Linear(800, 400)
        self.fc2 = nn.Linear(400, 200)
        self.out = nn.Linear(200, n_labels)

    def forward(self, w, s, l, Xcap, Xdash, Xnum, Xext, Xspecial, Xlen, Xpos):
        x = self.embW(w)
        x = self.dropW(x)
        x, _ = self.lstm_word(x)

        y = self.embS(s)
        y = self.dropS(y)
        y, _ = self.lstm_suf(y)

        z = self.embL(l)
        z = self.dropL(z)
        z, _ = self.lstm_lower(z)

        combined_embeddings = torch.cat((x, y, z), dim=2)

        additional_features = torch.stack((Xcap, Xdash, Xnum, Xext, Xspecial, Xlen, Xpos), dim=2).float()
        combined_features = torch.cat((combined_embeddings, additional_features), dim=2)

        combined_features, _ = self.lstm_combined(combined_features)

        fc_output = torch.relu(self.fc1(combined_features))
        fc_output = torch.relu(self.fc2(fc_output))

        output = self.out(fc_output)
        
        return output


############################################################
## Network with Bert embeddings

# class nercLSTM(nn.Module):
#     def __init__(self, codes, freeze_bert=False):
#         super(nercLSTM, self).__init__()

#         self.bert = BertModel.from_pretrained('bert-base-cased')
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        
#         # if freeze_bert:
#         #     for param in self.bert.parameters():
#         #         param.requires_grad = False
        
#         n_labels = codes.get_n_labels()
#         additional_feature_dim = 7  # Xcap, Xdash, Xnum, Xext, Xspecial, Xlen, Xpos

#         # LSTM for combined features (updated input size)
#         combined_input_size = 768 + additional_feature_dim  # 768 is the BERT hidden size
#         self.lstm_combined = nn.LSTM(combined_input_size, 200, bidirectional=True, batch_first=True)
#         self.dropout = nn.Dropout(0.5)  # Adding dropout for regularization

#         # Fully connected layers with increased size
#         self.fc1 = nn.Linear(400, 300)  # Increased from 200 to 300
#         self.fc2 = nn.Linear(300, 200)  # Increased from 100 to 200
#         self.fc3 = nn.Linear(200, 100)  # Added a third fully connected layer
#         self.out = nn.Linear(100, n_labels)  # Adjusted to match the previous layer size

#     def forward(self, token_ids, Xcap, Xdash, Xnum, Xext, Xspecial, Xlen, Xpos):
#         # Get BERT embeddings
#         bert_outputs = self.bert(token_ids)
#         x = bert_outputs.last_hidden_state

#         # Concatenate the additional features
#         additional_features = torch.stack((Xcap, Xdash, Xnum, Xext, Xspecial, Xlen, Xpos), dim=2).float()
#         combined_features = torch.cat((x, additional_features), dim=2)

#         # Combined LSTM
#         combined_features, _ = self.lstm_combined(combined_features)
#         combined_features = self.dropout(combined_features)  # Apply dropout
#         fc_output = torch.relu(self.fc1(combined_features))
#         fc_output = torch.relu(self.fc2(fc_output))
#         fc_output = torch.relu(self.fc3(fc_output))  # Added third fully connected layer
#         output = self.out(fc_output)
#         return output
