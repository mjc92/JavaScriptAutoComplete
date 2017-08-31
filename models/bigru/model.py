import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

# RNN Based Language Model
class biGRU(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, sos):
        super(biGRU, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.GRU(embed_size, hidden_size, num_layers, 
                                batch_first=True, bidirectional=True)
        self.decoder = nn.GRU(embed_size, hidden_size, num_layers, 
                                batch_first=True)
        self.W = nn.Linear(hidden_size*2, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
        self.sos = sos
        
    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        
    def forward(self, enc_in, dec_in, teacher_forcing=True):
        
        # Encoder
        enc_embedded = self.embed(enc_in)
        encoded, _ = self.encoder(enc_embedded)
        state = self.W(encoded[:,-1])
        state = state.unsqueeze(0)
        # Decoder
        sos = Variable(torch.LongTensor(np.ones([dec_in.size(0),1],dtype=int))).cuda()
        if teacher_forcing==True:
            dec_in = torch.cat([sos,dec_in[:,:-1]],dim=1)
            dec_embedded = self.embed(dec_in)
            out, _ = self.decoder(dec_embedded, state)
        else:
            out = []
            next_input = self.embed(sos)
            for i in range(dec_in.size(1)):
                tmp_out, state = self.decoder(next_input,state)
                out.append(tmp_out) # tmp_out : [b x 1 x hidden]
                next_words = self.linear(tmp_out.squeeze()) # [b x vocab]
                next_idx = next_words.max(1)[1] # [b]
                next_input = self.embed(next_idx).unsqueeze(1)   
            out = torch.cat(out,dim=1)
        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.contiguous().view(out.size(0)*out.size(1), out.size(2))
        # Decode hidden states of all time step
        out = self.linear(out)
        out = out.view(dec_in.size(0),dec_in.size(1),-1)
        return out