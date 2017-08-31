import numpy as np
import spacy
import torch
from torch.autograd import Variable
import os
from collections import Counter
import glob
from spacy import attrs
from slimit.lexer import Lexer
import re

def lex(text,output_type='all'):
    # lexes string, stops when it starts making TypeErrors
    # input: f.read()
    out_list = []
    lexer = Lexer()
    lexer.input(text)
    while True:
        try:
            token = lexer.token()
            if not token:
                break # break if end of token
            if output_type=='value':
                try:
                    out_list.append(token.value)
                except AttributeError:
                    break
            elif output_type=='type':
                try:
                    out_list.append(token.type)
                except AttributeError:
                    break
            else:
                out_list.append(token)
        except TypeError:
            break
        except AttributeError:
            break
    return out_list

def remove_comments(string):
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    # first group captures quoted strings (double or single)
    # second group captures comments (//single-line or /* multi-line */)
    regex = re.compile(pattern, re.MULTILINE|re.DOTALL)
    def _replacer(match):
        # if the 2nd group (capturing comments) is not None,
        # it means we have captured a non-quoted (real) comment string.
        if match.group(2) is not None:
            return "" # so we will return empty to remove the comment
        else: # otherwise, we will return the 1st group
            return match.group(1) # captured quoted-string
    return regex.sub(_replacer, string)

# Truncated Backpropagation 
def detach(states):
    return [state.detach() for state in states]

# changes a numpy array to a Variable of LongTensor type
def numpy_to_var(x,is_int=True):
    if is_int:
        x = torch.LongTensor(x)
    else:
        x = torch.Tensor(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def calc_running_avg_loss(loss, running_avg_loss, step, decay=0.99):
    if running_avg_loss==0:
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1-decay) * loss
    running_avg_loss = min(running_avg_loss,12) # clip
    return running_avg_loss

def to_cuda(item):
    if torch.cuda.is_available():
        return item.cuda()
    else:
        return item

def num_to_var(item):
    # numpy array to Variable
    if item.dtype==int:
        out = Variable(torch.LongTensor(item))
    else:
        out = Variable(torch.Tensor(item))
    return to_cuda(out)

def pack_padding(targets, outputs):
    # targets_np: Variable.cuda() of size [b x seq]
    # outputs: Variable.cuda() of size [b x seq x vocab]
    b,s,v = outputs.size() # get necessary sizes
    targets_np = targets.data.cpu().numpy()
    valid_outs = [] # valid outputs for each sequence
    for i, length in enumerate((targets_np!=0).sum(1)):
        valid_outs.append(np.arange(0,length)+i*s)
    valid_outs = np.hstack(valid_outs)
    valid_outs = torch.LongTensor(valid_outs).cuda()
    targets = targets.view(-1,1)
    outputs = outputs.view(b*s,v)
    return targets[valid_outs],outputs[valid_outs]
        
def decoder_initial(batch_size, sos_idx):
    decoder_in = torch.LongTensor(np.ones(batch_size,dtype=int))*sos_idx
    s = None
    w = None
    if torch.cuda.is_available():
        decoder_in = decoder_in.cuda()
    decoder_in = Variable(decoder_in)
    return decoder_in, s, w    
        