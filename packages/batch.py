import os
import random
import torch
from torch.autograd import Variable
import numpy as np
from functions import lex

class Batch(object):
    def __init__(self, file_dir, file_list, batch_size, in_seq, out_seq, 
                 max_oovs=30,seq2seq='m2m',ext=False):
        self.file_dir = file_dir # where files are stored
        self.full_list = file_list # list of the training files
        self.batch_size = batch_size
        self.in_seq = in_seq
        self.out_seq = out_seq
        
        self.ext = ext # whether external information is used

        self.max_oovs = max_oovs
        self.oov2idx_list = []
        self.idx2oov_list = []
        
        self.eof = list(np.zeros(self.batch_size)) # whether each file is EOF
        self.batch_files = [] # which files are now in batch
        self.batch_data = [] # stores the tokens for each file in each batch
        self.batch_in = [] # stores the inputs for each batch
        self.batch_out = [] # stores the outputs for each batch
        
        self.seq2seq = seq2seq # available from 'm2m', 'm2o', 'ed'
        
        self.epoch_end = 0
        
 # where to store states from previous minibatch
        
    def load_file(self,file,max_len=30000):
        with open(os.path.join(self.file_dir,file)) as f:
            text = f.read()
        if self.ext==True:
            text, types = lex(text,'both')
            return text[:max_len], types[:max_len]
        else:
            text = lex(text,'value')
            return text[:max_len]
    
#     def start_epoch(self): # for start of each epoch, initialize all

    def initialize_states(self, num_layers, hidden_size):
        self.states = (Variable(torch.zeros(num_layers, self.batch_size, hidden_size)).cuda(),
                  Variable(torch.zeros(num_layers, self.batch_size, hidden_size)).cuda())
    
    
    def next_epoch(self, batch_size):
        # initialize for next epoch
#         random.shuffle(self.file_list)
        self.epoch_end = 0
        self.batch_size = batch_size
        self.batch_files = self.full_list[:self.batch_size]
        self.file_list = self.full_list[self.batch_size:] # file list for one epoch
        self.eof = list(np.zeros(self.batch_size))
        self.batch_data = [self.load_file(file) for file in self.batch_files]
        self.batch_in = list(np.zeros(self.batch_size)) # just trying to make empty lists
        self.batch_out = list(np.zeros(self.batch_size))
        
    def get_minibatch(self,state_list):
        """
        stores in batch_in / batch_out tokens
        """
        self.oov2idx_list = []
        self.idx2oov_list = []
        
        # get the next batch inputs and outputs from batch_data
        for i,item in enumerate(self.batch_data):
            self.item = item
            
            if self.seq2seq == 'm2m':
                if self.in_seq>=len(item):
                    self.eof[i]=1
                    item = item + ['<EOS>']
                    for j in range((self.in_seq+1)-len(item)):
                        item = item + ['<PAD>']
                self.batch_in[i] = item[:self.in_seq]
                self.batch_out[i] = item[1:self.in_seq+1]
                self.batch_data[i] = item[self.in_seq:]
            
            elif self.seq2seq == 'm2o':
                if self.in_seq>=len(item):
                    self.eof[i]=1
                    item = item + ['<EOS>']
                    for j in range((self.in_seq+1)-len(item)):
                        item = item + ['<PAD>']
                self.batch_in[i] = item[:self.in_seq]
                self.batch_out[i] = item[self.in_seq]
                self.batch_data[i] = item[self.in_seq:]
            
            elif self.seq2seq == 'ed':
                if (self.in_seq+self.out_seq)>=len(item):
                    self.eof[i] = 1
                    item = item + ['<EOS>']
                    for j in range((self.in_seq+self.out_seq+1)-len(item)):
                        item = item + ['<PAD>']
                self.batch_in[i] = item[:self.in_seq]
                self.batch_out[i] = item[self.in_seq:self.in_seq+self.out_seq]
                self.batch_data[i] = item[self.in_seq+self.out_seq:]
        
            elif self.seq2seq == 'stmt':
                # STMT: similar to encoder-decoder, but all up the way to a statement (;)
                # 1. is the remaining list longer than the input size?
                if self.in_seq>=len(item):
                    self.eof[i] = 1
                    # 그냥 버리는 걸로
                    self.batch_in[i] = ['<PAD>' for i in range(self.in_seq)]
                    self.batch_out[i] = ['<PAD>' for i in range(self.out_seq)]
                else:
                    self.batch_in[i] = item[:self.in_seq]
                    item = item[self.in_seq:]
                    # 2. is there a semicolon?
                    try:
                        stmt_idx = item.index(';')+1 # 2. YES
                        # 2.1. is the stmt_idx before the max output length?
                        if stmt_idx<self.out_seq:
                            self.batch_out[i] = item[:stmt_idx] + ['<EOS>']
                            while len(self.batch_out[i])<self.out_seq:
                                self.batch_out[i].append('<PAD>')
                            item = item[stmt_idx+1:]
                        else:
                            self.batch_out[i] = item[:self.out_seq]
                    except ValueError: # 2. NO
                        # 2.2. is the remaining length longer than the max output
                        if self.out_seq>len(item):
                            self.eof[i] = 1
                            self.batch_out[i] = item
                            while len(self.batch_out[i])<self.out_seq:
                                self.batch_out[i].append('<PAD>')
                        else:
                            self.batch_out[i] = item[:self.out_seq]
                            item = item[self.out_seq:]
                self.batch_data[i] = item
                    
                # until we reach a specific statement such as semicolon
                
            
            # if len(item)<=self.in_seq: # if reaching end of a file
            #     self.eof[i]=1
            #     item = item + ['<EOS>']
            #     for j in range((self.in_seq+1)-len(item)):
            #         item = item + ['<PAD>']
            # # if not
            # self.batch_in[i] = item[:self.in_seq]
            # if self.seq2seq=='m2m': # if many2many mode
            #     self.batch_out[i] = item[1:self.in_seq+1]
            #     self.batch_data[i] = item[self.in_seq:]
            # elif self.seq2seq=='m2o': # 1 word at a time if many2one mode
            #     self.batch_out[i] = item[self.in_seq:self.in_seq+1]
            #     self.batch_data[i] = item[1:]
            # elif self.seq2seq=='ed': # N words in, M words out
            #     # original sequence: 0 ~ 20
            #     # input sequence: 0~10
            #     # decoder input: <SOS> + 11~19
            #     # decoder output: 11~20
            #     self.batch_out[i] = item[self.in_seq:self.in_seq+self.out_seq]
            #     self.batch_data[i] = item[self.in_seq+self.out_seq:]
            

    def next_minibatch(self):
        """
        Adds new files to the minibatch if we exhausted one or more of them
        """ 
        state_list_0 = torch.chunk(self.states[0],self.batch_size,dim=1)
        state_list_1 = torch.chunk(self.states[1],self.batch_size,dim=1)
        state_list_0 = list(state_list_0)
        state_list_1 = list(state_list_1)
#         print('===================')
        for i,val in enumerate(self.eof):
#             print(i,self.eof)
            if val==1: # if EOF for any file,
                if len(self.file_list)>0: # if there are any available files lest
                    self.batch_files[i] = self.file_list.pop()
                    self.batch_data[i] = self.load_file(self.batch_files[i])
                    self.eof[i]=0
                else:
                    idx = self.last_of_list(self.eof,0)
                    if i>=idx:
                        break
                    self.batch_data[i], self.batch_data[idx] = self.batch_data[idx], self.batch_data[i]
                    self.batch_in[i], self.batch_in[idx] = self.batch_in[idx], self.batch_in[i]
                    self.batch_out[i], self.batch_out[idx] = self.batch_out[idx], self.batch_out[i]
                    self.eof[i], self.eof[idx] = self.eof[idx], self.eof[i]
                    state_list_0[i], state_list_0[idx] = state_list_0[idx], state_list_0[i]
                    state_list_1[i], state_list_1[idx] = state_list_1[idx], state_list_1[i]
                    
        while 1 in self.eof:
            self.batch_data.pop()
            self.batch_in.pop()
            self.batch_out.pop()
            self.eof.pop()
            self.batch_size -= 1
            
            state_list_0.pop()
            state_list_1.pop()
        if len(self.eof)==0:
            self.epoch_end=1
            return
        states = (torch.cat(state_list_0,dim=1),torch.cat(state_list_1,dim=1))
        self.states = states
        del states
        del state_list_0,state_list_1
        
    def last_of_list(self,lst,c):
        # index of last element in a list that satisfies a condition c
        idx = len(lst) - 1 - next((i for i,x in enumerate(reversed(lst)) if x==c), len(lst))
        return idx
    
    def unk_minibatch(self, minibatch, vocab):
        # minibatch: np array where OOV words are given external vocab (e.g. 50003)
        # for a numpy array minibatch, put all unks to zero
        unk_idx = vocab.w2i['<UNK>']
        vocab_idxs = np.array(minibatch<vocab.count,dtype=int)
        oov_idxs = np.array(minibatch>=vocab.count,dtype=int) * unk_idx
        out = np.multiply(vocab_idxs, minibatch) # all OOV words are set to 0
        return out + oov_idxs # OOV words are instead set to UNK
    
    def get_type_idx(self):
        from slimit.lexer import Lexer
        lexer = Lexer()
        self.type2idx = dict()
        for i,tok in enumerate(lexer.tokens):
            self.type2idx[tok]=i