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
from packages.functions import detach, to_cuda, num_to_var, pack_padding, decoder_initial

class Args(object):
    mode = 'train'
    model = 'copynet'
    dataset_name = 'data_lexed'
    dataset_dir = '/home/irteam/users/data/150kJavaScript/'
    save_dir = 'None'
    startfrom = 0
    
    embed_size = 150
    hidden_size = 300
    num_layers = 1
    num_epochs = 10
    batch_size = 100
    seq_length = 50
    out_seq = 1
    lr = 0.002
    vocab_size = 50000
    max_oovs = 30
    seq2seq = 'ed'
args = Args()

if args.model=='lm':
    from models.lstm.model import RNNLM
    from models.lstm.solver import train,test
elif args.model=='copynet':
    from models.copynet.model import CopyEncoder, CopyDecoder
    from models.copynet.solver import train

with open(os.path.join(args.dataset_dir,args.mode,('file_list_%s.txt'%args.dataset_name))) as f:
    file_list = f.read().split('\n')

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
    save_dir = args.save_dir

print("Saving models in %s..." %save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
if args.startfrom==0:
    print("Loading new %s model..." %args.model)
    if args.model == 'lm':
        model = RNNLM(vocab.max_size, embed_size=args.embed_size, hidden_size=args.hidden_size,
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
    if args.model == 'lm':
        model = torch.load(os.path.join(save_dir,'saved_model_%d_epochs.pckl'%args.startfrom))
        model.cuda()
    elif args.model == 'copynet':
        encoder.cuda()
        decoder.cuda()
        model = (encoder, decoder)
# # set loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def train(args, batch, vocab, model, save_dir):
    
    # split model into encoder and decoder
    encoder,decoder = model
    
    # set loss function and optimizer
    criterion = nn.NLLLoss()
    opt_e = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    opt_d = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    
    step = 99999
    total_files = len(batch.full_list)
    print("Total number of files to read: %d" %total_files)

    for epoch in range(args.startfrom,args.num_epochs):
        print("===================== Epoch %d =====================" %(epoch+args.startfrom))

        batch.next_epoch(args.batch_size) # initialize batch data
        batch.initialize_states(args.num_layers, args.hidden_size)
        total_steps = step
        step=0
        while(batch.epoch_end==0):
            step+=1
            # update the minibatch inputs / outputs
            encoder.zero_grad()
            decoder.zero_grad()
            # get next minibatch
            batch.get_minibatch(0)
            
            # get inputs and targets from batch object
            for i in range(len(batch.batch_in)):
                in_line = batch.batch_in[i]
                out_line = batch.batch_out[i]
                oov2idx, idx2oov = vocab.create_oov_list(in_line+out_line, batch.max_oovs)
                batch.oov2idx_list.append(oov2idx)
                batch.idx2oov_list.append(idx2oov)
              
            inputs_oov_np = np.array([vocab.word_list_to_idx_list(line,batch.oov2idx_list[i]) for
                                 i,line in enumerate(batch.batch_in)],dtype=int)
            targets_oov_np = np.array([vocab.word_list_to_idx_list(line,batch.oov2idx_list[i]) for
                                 line in batch.batch_out],dtype=int)
            inputs_unk_np = np.array([vocab.word_list_to_idx_list(line) for
                                 i,line in enumerate(batch.batch_in)],dtype=int)
            targets_unk_np = np.array([vocab.word_list_to_idx_list(line) for
                                 line in batch.batch_out],dtype=int)
#             for i in range(batch.batch_size):
#                 print('IN ',i,": ",' '.join(batch.batch_in[i]))
#                 print('OUT ',i,": ",' '.join(batch.batch_out[i]))
#             print('\n')
            
            # whether to use teacher forcing
            if np.random.random_sample(size=1)[0]<(epoch*1.0/args.num_epochs):
                teacher_force = True
            else:
                teacher_force = False
                
            inputs = Variable(torch.LongTensor(inputs_unk_np)).cuda()
            targets = Variable(torch.LongTensor(targets_unk_np)).cuda()

            # run model to get outputs
            encoded, _ = encoder(inputs)
            
            decoder_in, s, w = decoder_initial(inputs.size(0),vocab.w2i['<SOS>'])
#             decoder_in = targets[:,0]
            
            for j in range(targets.size(1)):
                if j==0:
                    outputs,s,w = decoder(input_idx=decoder_in, encoded=encoded,
                        encoded_idx=inputs_oov_np,prev_state=s,
                        weighted=w, order=j)
                else:
                    tmp_out,s,w = decoder(input_idx=decoder_in, encoded=encoded,
                        encoded_idx=inputs_oov_np, prev_state=s,
                        weighted=w, order=j)
                    outputs = torch.cat([outputs,tmp_out],dim=1)
                
                # if teacher_force:
                #     decoder_in = out[:,-1,:].max(1)[1].squeeze()
                # else:
                decoder_in = targets[:,j] # train with ground truth
            
            targets = Variable(torch.LongTensor(targets_oov_np)).cuda()
            targets, outputs = pack_padding(targets, outputs)
#             print("Target size: ",targets.size())
#             print(' '.join([str(x.data[0]) for x in targets[:100]]))
#             print("Output size: ",outputs.size())
#             print('==============================')
            loss = criterion(torch.log(outputs), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm(encoder.parameters(),0.5)
            torch.nn.utils.clip_grad_norm(decoder.parameters(),0.5)
            opt_e.step()
            opt_d.step()

            batch.next_minibatch()
#             print("next minibatch done?")
#             for i in range(batch.batch_size):
#                 print("remaining file list: ", len(batch.batch_data[i]))
            if step%10==0:
                print ('Epoch [%d/%d], Files: [%d/%d],  Loss: %.3f, Steps: [%d/%d], Perplexity: %5.2f' %
               (epoch+args.startfrom, args.num_epochs, total_files-len(batch.file_list),
                total_files, loss.data[0], step, total_steps, np.exp(loss.data[0])))
        # save model at end of each epoch
        torch.save(f=os.path.join(save_dir,'saved_encoder_%d_epochs.pckl')%(epoch+args.startfrom),obj=encoder)
        torch.save(f=os.path.join(save_dir,'saved_decoder_%d_epochs.pckl')%(epoch+args.startfrom),obj=decoder)
        
print(args)
if args.mode=='train':
    train(args,batch,vocab,model,save_dir)