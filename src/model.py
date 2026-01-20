import torch
import torch.nn as nn

class LSTMBaseline(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        embedding_dim, 
        hidden_dim, 
        output_dim, 
        n_layers, 
        bidirectional, 
        dropout, 
        pad_idx
    ):
        super().__init__()
        
        # Embedding Layer: Converts integers to dense vectors
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=pad_idx
        )
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=n_layers, 
            bidirectional=bidirectional, 
            dropout=dropout,
            batch_first=True
        )
        
        # Fully Connected Layer
        # If bidirectional, the hidden state doubles in size
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.fc = nn.Linear(fc_input_dim, output_dim)
        
        # Dropout (for regularization)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text = [batch size, sent len]
        
        embedded = self.dropout(self.embedding(text))
        # embedded = [batch size, sent len, emb dim]
        
        # LSTM output
        # output = [batch size, sent len, hid dim * num directions]
        # hidden = [num layers * num directions, batch size, hid dim]
        output, (hidden, cell) = self.lstm(embedded)
        
        # Take the final hidden state to make the prediction
        # If bidirectional, concatenate the final forward and backward hidden states
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
            
        # hidden = [batch size, hid dim * num directions]
            
        return self.fc(hidden)