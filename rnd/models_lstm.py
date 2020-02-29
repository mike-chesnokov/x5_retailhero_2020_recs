import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence


class TransactionLSTM(nn.Module):
    
    def __init__(self, num_products, embedding_size=128, lstm_size=128, lstm_layers=1):
        super().__init__()
        self.lstm_size = lstm_size
        #self.embedding = nn.Embedding(num_products, embedding_size)
        self.lstm = nn.LSTM(input_size=num_products,#embedding_size
                            hidden_size=lstm_size,
                            num_layers=lstm_layers,
                            batch_first=True)
        self.dense = nn.Linear(lstm_size, num_products)
        
    def forward(self, batch):
        #embed = self.embedding(x)
        x, x_lengths, _ = batch
        # pack for lstm input
        x = pack_padded_sequence(x, x_lengths, batch_first=True)
        # ignore passing hidden state to lstm input (by default h0 set to zeros)
        _, hidden = self.lstm(x)
        #output_padded, output_lengths = pad_packed_sequence(output, batch_first=True)
        
        linear_output = self.dense(hidden[0])

        return linear_output
 