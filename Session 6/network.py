import torch
import torch.nn as nn
import torch.nn.functional as F

class ddiCNN(nn.Module):
    def __init__(self, codes):
        super(ddiCNN, self).__init__()
        # get sizes
        n_words = codes.get_n_words()
        n_labels = codes.get_n_labels()
        max_len = codes.maxlen
        embedding_dim = 100

        # create embedding layers
        self.embW = nn.Embedding(n_words, embedding_dim, padding_idx=0)
        self.embXLW = nn.Embedding(n_words, embedding_dim, padding_idx=0)
        self.embXL = nn.Embedding(n_words, embedding_dim, padding_idx=0)
        self.embXP = nn.Embedding(n_words, embedding_dim, padding_idx=0)


        # Convolutional layers
        self.conv1 = nn.Conv1d(embedding_dim * 4, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)

        # Batch normalization layers
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(64)

        # Dropout layers
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

        # Max pooling layer
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(64, 64, num_layers=1, bidirectional=True, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(64 * 2, n_labels)

    def forward(self, w, xlw, xl, xp):
        # embed the inputs
        emb_w = self.embW(w)
        emb_xlw = self.embXLW(xlw)
        emb_xl = self.embXL(xl)
        emb_xp = self.embXP(xp)

        # concatenate embeddings
        x = torch.cat((emb_w, emb_xlw, emb_xl, emb_xp), dim=2)

        # permute dimensions to fit Conv1D input requirements
        x = x.permute(0, 2, 1)

        # apply Conv1D + BatchNorm + ReLU + Dropout + MaxPooling
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.dropout3(x)

        # permute back for LSTM layer
        x = x.permute(0, 2, 1)

        # LSTM layer
        x, _ = self.lstm(x)

        # use the last hidden state for classification
        x = x[:, -1, :]

        # apply fully connected layer
        x = self.fc(x)

        return x

# Example of how to create the model
# codes = your_codes_object  # This should be defined as per your requirements
# pretrained_embeddings = load_pretrained_embeddings()  # Load pre-trained embeddings if available
# model = ImprovedDDICNNLSTM(codes, pretrained_embeddings)

# Example of defining the criterion
criterion = nn.CrossEntropyLoss()

# import torch
# import torch.nn as nn
# import torch.nn.functional as func

# criterion = nn.CrossEntropyLoss()

# class ddiCNN(nn.Module):

#     def __init__(self, codes):
#         super(ddiCNN, self).__init__()
#         # get sizes
#         n_words = codes.get_n_words()
#         n_labels = codes.get_n_labels()
#         max_len = codes.maxlen
#         # create layers
#         self.embW = nn.Embedding(n_words, 75, padding_idx=0)
#         self.lstm = nn.LSTM(75, 64, num_layers=2, bidirectional=True, batch_first=True)
#         self.maxpool = nn.MaxPool1d(150)
#         self.dropout = nn.Dropout(0.1)
#         self.dense = nn.Linear(128, 64)
#         self.out = nn.Linear(64, n_labels)

#     def forward(self, w):
#         # run layers on given data
#         x = self.embW(w)
#         x, _ = self.lstm(x)
#         x = x.permute(0, 2, 1)
#         x = self.maxpool(x)
#         x = x.squeeze(dim=2)
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = func.leaky_relu(x)
#         x = self.out(x)
#         return x

