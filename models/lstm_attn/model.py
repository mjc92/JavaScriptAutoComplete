import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

# RNN Based Language Model
class RNNLM_attn(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
        
    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        
    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x) 
        
        # Forward propagate RNN  
        out, h = self.lstm(x, h)
        
        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.contiguous().view(out.size(0)*out.size(1), out.size(2))
        
        # Decode hidden states of all time step
        out = self.linear(out)
        out = out.view(x.size(0),x.size(1),-1)
        return out, h