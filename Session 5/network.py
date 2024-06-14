import torch
import torch.nn as nn
import torch.nn.functional as F


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


class nercLSTM(nn.Module):
    def __init__(self, codes):
        super(nercLSTM, self).__init__()

        n_words = codes.get_n_words()
        n_sufs = codes.get_n_sufs()
        n_labels = codes.get_n_labels()

        self.embW = nn.Embedding(n_words, 100)
        self.embS = nn.Embedding(n_sufs, 50)
        self.embL = nn.Embedding(n_words, 100)  # Embedding for lowercased words
        self.dropW = nn.Dropout(0.1)
        self.dropS = nn.Dropout(0.1)
        self.dropL = nn.Dropout(0.1)  # Dropout for lowercased words

        # Additional feature dimensions
        additional_feature_dim = 7  # Xcap, Xdash, Xnum, Xext, Xspecial, Xlen, Xpos

        # LSTM layers for word and suffix embeddings
        self.lstm_word = nn.LSTM(100, 100, bidirectional=True, batch_first=True)
        self.lstm_suf = nn.LSTM(50, 50, bidirectional=True, batch_first=True)
        self.lstm_lower = nn.LSTM(100, 100, bidirectional=True, batch_first=True)  # LSTM for lowercased words

        # LSTM for combined features (updated input size)
        combined_input_size = 200 + 100 + 200 + additional_feature_dim  # Corrected combined input size
        self.lstm_combined = nn.LSTM(combined_input_size, 200, bidirectional=True, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(400, 200)
        self.fc2 = nn.Linear(200, 100)
        
        # Output layer
        self.out = nn.Linear(100, n_labels)

    def forward(self, w, s, l, Xcap, Xdash, Xnum, Xext, Xspecial, Xlen, Xpos):
        # Word embeddings
        x = self.embW(w)
        x = self.dropW(x)
        x, _ = self.lstm_word(x)

        # Suffix embeddings
        y = self.embS(s)
        y = self.dropS(y)
        y, _ = self.lstm_suf(y)

        # Lowercased word embeddings
        z = self.embL(l)
        z = self.dropL(z)
        z, _ = self.lstm_lower(z)

        # Concatenate the word, suffix, and lowercased word LSTM outputs
        combined_embeddings = torch.cat((x, y, z), dim=2)

        # Concatenate the additional features
        additional_features = torch.stack((Xcap, Xdash, Xnum, Xext, Xspecial, Xlen, Xpos), dim=2).float()
        combined_features = torch.cat((combined_embeddings, additional_features), dim=2)

        # Combined LSTM
        combined_features, _ = self.lstm_combined(combined_features)
        
        # Fully connected layers
        fc_output = torch.relu(self.fc1(combined_features))
        fc_output = torch.relu(self.fc2(fc_output))

        # Output layer
        output = self.out(fc_output)
        
        return output



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


# class nercLSTM(nn.Module):
#     def __init__(self, codes):
#         super(nercLSTM, self).__init__()

#         n_words = codes.get_n_words()
#         n_sufs = codes.get_n_sufs()
#         n_labels = codes.get_n_labels()
        
#         # Embeddings
#         self.embW = nn.Embedding(n_words, 100)  # Main words
#         self.embS = nn.Embedding(n_sufs, 50)    # Suffixes
#         self.embL = nn.Embedding(n_words, 100)  # Lowercase words
        
#         # Dropout
#         self.dropW = nn.Dropout(0.1)
#         self.dropS = nn.Dropout(0.1)
#         self.dropL = nn.Dropout(0.1)

#         # LSTM Layers
#         self.lstm1 = nn.LSTM(404, 200, bidirectional=True, batch_first=True)  # Updated input size
#         self.lstm2 = nn.LSTM(400, 200, bidirectional=True, batch_first=True)
        
#         # Output layer
#         self.out = nn.Linear(400, n_labels)

#     def forward(self, Xw, Xs, Xl, Xcap, Xdash, Xnum):
#         xw = self.dropW(self.embW(Xw))    # Embed and apply dropout to main words
#         xs = self.dropS(self.embS(Xs))    # Embed and apply dropout to suffixes
#         xl = self.dropL(self.embL(Xl))    # Embed and apply dropout to lowercase words

#         # Stack all features along the feature dimension
#         x = torch.cat((xw, xs, xl, Xcap.unsqueeze(2), Xdash.unsqueeze(2), Xnum.unsqueeze(2)), dim=2)
        
#         x, _ = self.lstm1(x)  # First LSTM layer
#         x, _ = self.lstm2(x)  # Second LSTM layer
#         x = self.out(x)       # Output layer
        
#         return x


###########################################################################

# class nercLSTM(nn.Module):
#     def __init__(self, codes):
#         super(nercLSTM, self).__init__()

#         n_words = codes.get_n_words()
#         n_sufs = codes.get_n_sufs()
#         n_labels = codes.get_n_labels()
        
#         # Embeddings
#         self.embW = nn.Embedding(n_words, 100)  # Main words
#         self.embS = nn.Embedding(n_sufs, 50)    # Suffixes
#         self.embL = nn.Embedding(n_words, 100)  # Lowercase words
        
#         # Dropout
#         self.dropW = nn.Dropout(0.1)
#         self.dropS = nn.Dropout(0.1)
#         self.dropL = nn.Dropout(0.1)

#         # CNN Layer
#         self.conv = nn.Conv1d(in_channels=253, out_channels=100, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
#         # Transposed CNN Layer to restore sequence length
#         self.deconv = nn.ConvTranspose1d(in_channels=100, out_channels=100, kernel_size=2, stride=2)

#         # LSTM Layers
#         self.lstm1 = nn.LSTM(100, 200, bidirectional=True, batch_first=True)
#         self.lstm2 = nn.LSTM(400, 200, bidirectional=True, batch_first=True)
        
#         # Output layer
#         self.out = nn.Linear(400, n_labels)

#     def forward(self, Xw, Xs, Xl, Xcap, Xdash, Xnum):
#         xw = self.dropW(self.embW(Xw))    # Embed and apply dropout to main words
#         xs = self.dropS(self.embS(Xs))    # Embed and apply dropout to suffixes
#         xl = self.dropL(self.embL(Xl))    # Embed and apply dropout to lowercase words

#         # Stack all features along the feature dimension
#         x = torch.cat((xw, xs, xl, Xcap.unsqueeze(2), Xdash.unsqueeze(2), Xnum.unsqueeze(2)), dim=2)  # (batch, seq_length, feature)
        
#         # Permute to have channels as the second dimension for CNN layer
#         x = x.permute(0, 2, 1)  # (batch, feature, seq_length)
#         x = self.pool(F.relu(self.conv(x)))  # Apply CNN and pooling
        
#         # Restore the sequence length using transposed convolution
#         x = self.deconv(x)  # Apply transposed CNN
        
#         # Permute back for LSTM layer (batch, seq_length, feature)
#         x = x.permute(0, 2, 1)
        
#         x, _ = self.lstm1(x)  # First LSTM layer
#         x, _ = self.lstm2(x)  # Second LSTM layer
#         x = self.out(x)       # Output layer
        
#         return x

# # 52 on devel
# class nercLSTM(nn.Module):
#     def __init__(self, codes, pretrained_embeddings=None):
#         super(nercLSTM, self).__init__()

#         n_words = codes.get_n_words()
#         n_sufs = codes.get_n_sufs()
#         n_labels = codes.get_n_labels()
        
#         # Embeddings
#         self.embW = nn.Embedding(n_words, 200)  # Main words
#         self.embS = nn.Embedding(n_sufs, 100)   # Suffixes
#         self.embL = nn.Embedding(n_words, 200)  # Lowercase words
        
#         # Load pretrained embeddings if provided
#         if pretrained_embeddings is not None:
#             self.embW.weight.data.copy_(pretrained_embeddings)
#             self.embW.weight.requires_grad = False  # Freeze the pretrained embeddings

#         # Dropout
#         self.dropW = nn.Dropout(0.2)
#         self.dropS = nn.Dropout(0.2)
#         self.dropL = nn.Dropout(0.2)

#         # CNN Layers
#         self.conv1 = nn.Conv1d(in_channels=507, out_channels=256, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv1d(in_channels=128, out_channels=100, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
#         # Transposed CNN Layer to restore sequence length
#         self.deconv = nn.ConvTranspose1d(in_channels=100, out_channels=200, kernel_size=2, stride=2)

#         # LSTM Layers
#         self.lstm1 = nn.LSTM(200, 400, bidirectional=True, batch_first=True)
#         self.lstm2 = nn.LSTM(800, 400, bidirectional=True, batch_first=True)
        
#         # Additional fully connected layer
#         self.fc1 = nn.Linear(800, 800)

#         # Output layer
#         self.out = nn.Linear(800, n_labels)

#     def forward(self, Xw, Xs, Xl, Xcap, Xdash, Xnum, Xext, Xspecial, Xlen, Xpos):
#         xw = self.dropW(self.embW(Xw))    # Embed and apply dropout to main words
#         xs = self.dropS(self.embS(Xs))    # Embed and apply dropout to suffixes
#         xl = self.dropL(self.embL(Xl))    # Embed and apply dropout to lowercase words

#         # Stack all features along the feature dimension
#         x = torch.cat((xw, xs, xl, Xcap.unsqueeze(2), Xdash.unsqueeze(2), Xnum.unsqueeze(2), Xext.unsqueeze(2), Xspecial.unsqueeze(2), Xlen.unsqueeze(2), Xpos.unsqueeze(2)), dim=2)  # (batch, seq_length, feature)
        
#         # Permute to have channels as the second dimension for CNN layer
#         x = x.permute(0, 2, 1)  # (batch, feature, seq_length)

#         # Apply stacked CNN layers
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = self.pool(x)  # Apply pooling
        
#         # Restore the sequence length using transposed convolution
#         x = self.deconv(x)  # Apply transposed CNN
        
#         # Permute back for LSTM layer (batch, seq_length, feature)
#         x = x.permute(0, 2, 1)
        
#         x, _ = self.lstm1(x)  # First LSTM layer
#         x, _ = self.lstm2(x)  # Second LSTM layer

#         # Pass through additional fully connected layer
#         x = F.relu(self.fc1(x))

#         # Output layer
#         x = self.out(x)       # Output layer
        
#         return x




# import torch
# import torch.nn as nn
# from transformers import BertModel, BertTokenizer


# class nercLSTM(nn.Module):
#     def __init__(self, codes):
#         super(nercLSTM, self).__init__()

#         # Setting up the device
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # Load pre-trained BERT and move to the appropriate device
#         self.bert = BertModel.from_pretrained('bert-base-uncased').to(self.device)
#         for param in self.bert.parameters():
#             param.requires_grad = False  # Freeze BERT to not train it

#         n_sufs = codes.get_n_sufs()
#         n_labels = codes.get_n_labels()
        
#         # Embeddings for suffixes and lowercase words
#         self.embS = nn.Embedding(n_sufs, 50).to(self.device)    # Suffixes
#         self.embL = nn.Embedding(codes.get_n_words(), 100).to(self.device)  # Lowercase words
        
#         # Dropout
#         self.dropW = nn.Dropout(0.1)
#         self.dropS = nn.Dropout(0.1)
#         self.dropL = nn.Dropout(0.1)

#         # LSTM Layers
#         self.lstm1 = nn.LSTM(921, 200, bidirectional=True, batch_first=True).to(self.device)
#         self.lstm2 = nn.LSTM(400, 200, bidirectional=True, batch_first=True).to(self.device)
        
#         # Output layer
#         self.out = nn.Linear(400, n_labels).to(self.device)

#     def forward(self, Xw, Xs, Xl, Xcap, Xdash, Xnum):
#         # Move inputs to the same device as the model
#         Xw, Xs, Xl, Xcap, Xdash, Xnum = Xw.to(self.device), Xs.to(self.device), Xl.to(self.device), Xcap.to(self.device), Xdash.to(self.device), Xnum.to(self.device)

#         # Getting BERT embeddings
#         with torch.no_grad():  # Ensure BERT is in inference mode
#             xw = self.bert(Xw).last_hidden_state
        
#         xs = self.dropS(self.embS(Xs))  # Embed and apply dropout to suffixes
#         xl = self.dropL(self.embL(Xl))  # Embed and apply dropout to lowercase words

#         # Stack all features along the feature dimension
#         x = torch.cat((xw, xs, xl, Xcap.unsqueeze(2), Xdash.unsqueeze(2), Xnum.unsqueeze(2)), dim=2)
        
#         x, _ = self.lstm1(x)  # First LSTM layer
#         x, _ = self.lstm2(x)  # Second LSTM layer
#         x = self.out(x)       # Output layer
        
#         return x
