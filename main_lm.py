import os
import torch 
import torch.nn as nn
import numpy as np
import argparse
import pickle
from torch.autograd import Variable
from packages.vocab import Vocab
from packages.batch import Batch
from slimit.lexer import Lexer
from packages.functions import detach, to_cuda, num_to_var, pack_padding

# arguments related to the dataset
parser = argparse.ArgumentParser()

# about running mode
parser.add_argument("--mode",type=str, help='whether "train" or "test" data')
parser.add_argument("--model",type=str, help='which model to use')
parser.add_argument("--dataset_name",type=str, default='data_lexed', help='name of preprocessed dataset')
parser.add_argument("--dataset_dir",type=str, default= '/home/irteam/users/data/150kJavaScript/', 
                    help='name of preprocessed dataset')
parser.add_argument("--save_dir",type=str, default= 'None', 
                    help='where to save models / where saved models are')
parser.add_argument("--startfrom",type=int, default=-1, help='whether starting from a previous model')
parser.add_argument("--cudnn",type=bool, default=False, help='use CUDNN backend')

parser.add_argument("--default",type=bool, default=False, 
                    help='just stick with default settings')

# model information
parser.add_argument("--embed_size",type=int, default=150, help='embedding size')
parser.add_argument("--hidden_size",type=int, default=300, help='hidden size')
parser.add_argument("--num_layers",type=int, default=1, help='number of LSTM layers')
parser.add_argument("--num_epochs",type=int, default=10, help='number of epochs to run')
parser.add_argument("--batch_size",type=int, default=180, help='size of minibatch')
parser.add_argument("--seq_length",type=int, default=50, help='length of sequence size')
parser.add_argument("--out_seq",type=int, default=10, help='length of output sequence for encoder-decoder')
parser.add_argument("--lr",type=float, default=0.002, help='learning rate size')
parser.add_argument("--vocab_size",type=int, default=50000, help='vocab_size')
parser.add_argument("--max_oovs",type=int, default=30, help='max number of oovs per sample')
parser.add_argument("--seq2seq",type=str, default='stmt', help='which type of seq2seq to apply\
                    :\n[m2m] - many to many (in: 0~n / out: 1~n+1)\
                     \n[m2o] - many to one (in: 0~n / out: n+1)\
                     \n[ed]  - encoder-decoder (in: 0~n / out: n+1~??')

args = parser.parse_args()
print(args)
# if args.default:
#     args = get_args(args)
# else:
#     print(args)

if args.cudnn==True:
    from torch.backends import cudnn
    cudnn.benchmark=True


if args.model=='bigru':
    from models.bigru.model import biGRU
    from models.bigru.solver import train,test,autocomplete
if args.model=='bigru_attn':
    from models.bigru_attn.model import biGRU_attn
    from models.bigru.solver import train,test,autocomplete
elif args.model=='copynet':
    from models.copynet.model import CopyEncoder, CopyDecoder
    from models.copynet.solver import train,test,autocomplete

if (args.mode=='train') | (args.mode == 'test'):
    with open(os.path.join(args.dataset_dir,args.mode,('file_list_%s.txt'%args.dataset_name))) as f:
        file_list = f.read().split('\n')
else:
    file_list = []
        
batch = Batch(file_dir=os.path.join(args.dataset_dir,args.mode,args.dataset_name),
              file_list=file_list,
              batch_size=args.batch_size,
              in_seq=args.seq_length,
              out_seq=args.out_seq,
              max_oovs=args.max_oovs,
              seq2seq=args.seq2seq)

idx2id = np.load('vocab/data_lexed/idx2id.npy').item()
idx2tok = np.load('vocab/data_lexed/idx2tok.npy').item()
idx2reg = np.load('vocab/data_lexed/idx2reg.npy').item()
word_list = [x for (_,x) in idx2tok.items()] + [x for (_,x) in idx2reg.items()][:1000] + \
        [x for (_,x) in idx2id.items()][:args.vocab_size]
vocab = Vocab(args.vocab_size)
vocab.add_to_vocab(word_list)
print("Vocab size: %d" %vocab.count)

# import lexer
lexer = Lexer()

# set directory to save model
if args.save_dir=='None':
    save_dir = os.path.join('saved_weights',args.model)
else:
    save_dir = os.path.join('saved_weights',args.save_dir)

print("Saving models in %s..." %save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# load model

if args.startfrom==-1:
    print("Loading new %s model..." %args.model)
    if args.model == 'bigru':
        model = biGRU(vocab.max_size, embed_size=args.embed_size, hidden_size=args.hidden_size,
                         num_layers = args.num_layers)
        model.cuda()
    if args.model == 'bigru_attn':
        model = biGRU_attn(vocab.max_size, embed_size=args.embed_size, hidden_size=args.hidden_size,
                         num_layers = args.num_layers)
        model.cuda()
    elif args.model == 'copynet':
        encoder = CopyEncoder(args.vocab_size, args.embed_size, args.hidden_size)
        decoder = CopyDecoder(args.vocab_size, args.embed_size, args.hidden_size, args.max_oovs)
        encoder.cuda()
        decoder.cuda()
        model = (encoder, decoder)
else:
    print("Loading %s model from %d epochs..." %(args.model, args.startfrom))
    if (args.model == 'bigru') | (args.model == 'bigru_attn'):
        model = torch.load(os.path.join(save_dir,'saved_model_%d_epochs.pckl'%args.startfrom))
        model.cuda()
    elif args.model == 'copynet':
        encoder = torch.load(os.path.join(save_dir,'saved_encoder_%d_epochs.pckl'%args.startfrom))
        decoder = torch.load(os.path.join(save_dir,'saved_decoder_%d_epochs.pckl'%args.startfrom))
        encoder.cuda()
        decoder.cuda()
        model = (encoder, decoder)
args.startfrom += 1
# # set loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


######################### training ###########################
if args.mode=='train':
    train(args,batch,vocab,model,save_dir)
elif args.mode=='test':
    test(args,batch,vocab,model,save_dir)
elif args.mode=='autocomplete':
    autocomplete(args,batch,vocab,model,save_dir)    
