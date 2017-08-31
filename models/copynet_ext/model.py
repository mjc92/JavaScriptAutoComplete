import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import time
from collections import Counter

class CopyEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(CopyEncoder, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)

        self.gru = nn.GRU(input_size=embed_size,
            hidden_size=hidden_size, batch_first=True,
            bidirectional=True)

    def forward(self, x):
        # input: [b x seq]
        embedded = self.embed(x)
        out, h = self.gru(embedded) # out: [b x seq x hid*2] (biRNN)
        return out, h

class CopyDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, max_oovs=12):
        super(CopyDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.time = time.time()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(input_size=embed_size+hidden_size*2,
            hidden_size=hidden_size, batch_first=True)
        self.max_oovs = max_oovs # largest number of OOVs available per sample

        # weights
        self.Ws = nn.Linear(hidden_size*2, hidden_size) # only used at initial stage
        self.Wo = nn.Linear(hidden_size, vocab_size) # generate mode
        self.Wc = nn.Linear(hidden_size*2, hidden_size) # copy mode
        self.nonlinear = nn.Tanh()

    def forward(self, input_idx, encoded, encoded_idx, prev_state, weighted, order):
        # input_idx(y_(t-1)): [b]			<- idx of next input to the decoder (Variable)
        # encoded: [b x seq x hidden*2]		<- hidden states created at encoder (Variable)
        # encoded_idx: [b x seq]			<- idx of inputs used at encoder (numpy)
        # prev_state(s_(t-1)): [1 x b x hidden]		<- hidden states to be used at decoder (Variable)
        # weighted: [b x 1 x hidden*2]		<- weighted attention of previous state, init with all zeros (Variable)

        # hyperparameters
        start = time.time()
        time_check = False
        b = encoded.size(0) # batch size
        seq = encoded.size(1) # input sequence length
        vocab_size = self.vocab_size
        hidden_size = self.hidden_size

        # 0. set initial state s0 and initial attention (blank)
        if order==0:
            prev_state = self.Ws(encoded[:,-1])
            weighted = torch.Tensor(b,1,hidden_size*2).zero_()
            weighted = self.to_cuda(weighted)
            weighted = Variable(weighted)

        prev_state = prev_state.unsqueeze(0) # [1 x b x hidden]
        if time_check:
            self.elapsed_time('state 0')

        # 1. update states
        gru_input = torch.cat([self.embed(input_idx).unsqueeze(1), weighted],2) # [b x 1 x (h*2+emb)]
        _, state = self.gru(gru_input, prev_state)
        state = state.view(b,-1) # [b x h]

        if time_check:
            self.elapsed_time('state 1')

        # 2. predict next word y_t
        # 2-1) get scores score_g for generation- mode
        score_g = self.Wo(state) # [b x vocab_size]

        if time_check:
            self.elapsed_time('state 2-1')

        # 2-2) get scores score_c for copy mode, remove possibility of giving attention to padded values
        score_c = F.tanh(self.Wc(encoded.contiguous().view(-1,hidden_size*2))) # [b*seq x hidden_size]
        score_c = score_c.view(b,-1,hidden_size) # [b x seq x hidden_size]
        score_c = torch.bmm(score_c, state.unsqueeze(2)).view(b,-1) # [b x seq]

        score_c = F.tanh(score_c) # purely optional....

        encoded_mask = torch.Tensor(np.array(encoded_idx==0, dtype=float)*(-1000)) # [b x seq]
        encoded_mask = self.to_cuda(encoded_mask)
        encoded_mask = Variable(encoded_mask)
        score_c = score_c + encoded_mask # padded parts will get close to 0 when applying softmax

        if time_check:
            self.elapsed_time('state 2-2')

        # 2-3) get softmax-ed probabilities
        score = torch.cat([score_g,score_c],1) # [b x (vocab+seq)]
        probs = F.softmax(score)
        prob_g = probs[:,:vocab_size] # [b x vocab]
        prob_c = probs[:,vocab_size:] # [b x seq]

        if time_check:
            self.elapsed_time('state 2-3')

        # 2-4) add empty sizes to prob_g which correspond to the probability of obtaining OOV words
        oovs = Variable(torch.Tensor(b,self.max_oovs).zero_())+1e-4
        oovs = self.to_cuda(oovs)
        prob_g = torch.cat([prob_g,oovs],1)

        if time_check:
            self.elapsed_time('state 2-4')

        # 2-5) add prob_c to prob_g

        """A. try 2 for loops"""
        # prob_c_to_g = self.to_cuda(torch.Tensor(prob_g.size()).zero_())
        # prob_c_to_g = Variable(prob_c_to_g)
        # for b_idx in range(b): # for each sequence in batch
        #     for s_idx in range(seq):
        #         prob_c_to_g[b_idx,encoded_idx[b_idx,s_idx]]=\
        #         prob_c_to_g[b_idx,encoded_idx[b_idx,s_idx]]+prob_c[b_idx,s_idx]
        
        """B. try this complicated method
         encoded_idx : np, [b x in_seq]
         prob_c : Variable.cuda(), [b x 1 x seq]"""
        numbers = encoded_idx.reshape(-1).tolist()
        set_numbers = list(set(numbers))
        c = Counter(numbers)
        dup_list = [k for k in set_numbers if (c[k]>1)]
        if time_check:
            self.elapsed_time('state 2-5-1')
        dup_attn_sum = Variable(torch.FloatTensor(np.zeros([b,seq],dtype=float)))
        dup_attn_sum = self.to_cuda(dup_attn_sum)
        if time_check:
            self.elapsed_time('state 2-5-2')
            
        """
        for dup in dup_list, get mask from encoded_idx (Variable or Tensor)
        encoded_idx should be a Variable or Tensor version then
        
        """
        # masked_idx_sum = np.zeros([b,seq])
        masked_idx_sum = Variable(torch.Tensor(b,seq).zero_()).cuda()
        encoded_idx_var = Variable(torch.Tensor(np.array(encoded_idx,dtype=float))).cuda()
        
        for dup in dup_list: # for duplicating elements
            mask = (encoded_idx_var==dup).float()
            masked_idx_sum += mask
            attn_mask = torch.mul(mask,prob_c)
            attn_sum = attn_mask.sum(1).unsqueeze(1)
            dup_attn_sum += torch.mul(attn_mask,attn_sum)
            
            
            
            # mask = np.array(encoded_idx==dup,dtype=float) # mask: sparse matrix, 1 if ==dup 
            # masked_idx_sum += mask # add to matrix 'mask'
            # mask = torch.Tensor(mask)
            # mask = Variable(mask).cuda()
            # attn_mask = torch.mul(mask,prob_c.squeeze())
            # attn_mask = torch.mul(Variable(torch.Tensor(mask)).cuda(),prob_c.squeeze())
            # attn_sum = attn_mask.sum(1).unsqueeze(1) # squeeze attn_sum to a vector, each row
            # dup_attn_sum += torch.mul(attn_mask,attn_sum)
        if time_check:
            self.elapsed_time('state 2-5-3')
        # masked_idx_sum = Variable(torch.Tensor(masked_idx_sum).cuda())

        # dup_attn_sum: duplicate positions each have the same attention
        # mask_idx_sum: indicating the positions of each duplicate position

        attn = torch.mul(prob_c,(1-masked_idx_sum))+dup_attn_sum
        batch_indices = torch.arange(start=0, end=b).long()
        batch_indices = batch_indices.expand(seq,b).transpose(1,0).contiguous().view(-1)
        idx_repeat = torch.arange(start=0, end=seq).repeat(b).long()
        prob_c_to_g = torch.zeros(b,self.vocab_size+self.max_oovs)
        prob_c_to_g = self.to_cuda(Variable(prob_c_to_g))
        word_indices = encoded_idx.reshape(-1)
        prob_c_to_g[batch_indices,word_indices] += attn[batch_indices,idx_repeat]

        """C. try a huge one-hot tensor"""
        # prob_c_to_g = Variable
        # en = torch.LongTensor(encoded_idx) # [b x in_seq]
        # en.unsqueeze_(2) # [b x in_seq x 1]
        # one_hot = torch.FloatTensor(en.size(0),en.size(1),prob_g.size(1)).zero_() # [b x in_seq x vocab]
        # one_hot.scatter_(2,en,1) # one hot tensor: [b x seq x vocab]
        # one_hot = self.to_cuda(one_hot)
        # prob_c_to_g = torch.bmm(prob_c.unsqueeze(1),Variable(one_hot)) # [b x 1 x vocab]
        # prob_c_to_g = prob_c_to_g.squeeze() # [b x vocab]

        out = prob_g + prob_c_to_g
        out = out.unsqueeze(1) # [b x 1 x vocab]

        if time_check:
            self.elapsed_time('state 2-5')

        # 3. get weighted attention to use for predicting next word
        # 3-1) get tensor that shows whether each decoder input has previously appeared in the encoder
        idx_from_input = []
        input_idx_np = input_idx.data.cpu().numpy()
        idx_from_input = \
        (encoded_idx == input_idx_np.repeat(encoded_idx.shape[1]).reshape(encoded_idx.shape))
        # for i,line in enumerate(encoded_idx):
        #     idx_from_input.append([int(k==input_idx[i].data[0]) for k in line])
        if time_check:
            self.elapsed_time('state 3-1-1')
        idx_from_input = torch.Tensor(np.array(idx_from_input, dtype=float))
        # idx_from_input : np.array of [b x seq]
        idx_from_input = self.to_cuda(idx_from_input)
        idx_from_input = Variable(idx_from_input)
        for i in range(b):
            if idx_from_input[i].sum().data[0]>1:
                idx_from_input[i] = idx_from_input[i]/idx_from_input[i].sum().data[0]

        if time_check:
            self.elapsed_time('state 3-1')

        # 3-2) multiply with prob_c to get final weighted representation
        attn = prob_c * idx_from_input
        # for i in range(b):
        # 	tmp_sum = attn[i].sum()
        # 	if (tmp_sum.data[0]>1e-6):
        # 		attn[i] = attn[i] / tmp_sum.data[0]
        attn = attn.unsqueeze(1) # [b x 1 x seq]
        weighted = torch.bmm(attn, encoded) # weighted: [b x 1 x hidden*2]

        if time_check:
            self.elapsed_time('state 3-2')

        return out, state, weighted

    def to_cuda(self, tensor):
        # turns to cuda
        if torch.cuda.is_available():
            return tensor.cuda()
        else:
            return tensor

    def elapsed_time(self, state):
        elapsed = time.time()
        print("Time difference from %s: %1.4f"%(state,elapsed-self.time))
        self.time = elapsed
        return
