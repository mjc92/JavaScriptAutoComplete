# solver for copynet
import torch
from torch import nn, optim
from torch.autograd import Variable
import random
from packages.functions import detach, pack_padding, decoder_initial
import numpy as np
import os
import time

def train(args, batch, vocab, model, save_dir):
    
    # split model into encoder and decoder
    encoder,decoder = model
    
    # set loss function and optimizer
    criterion = nn.NLLLoss()
    opt_e = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    opt_d = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    
    step = 99999
    # batch.full_list = batch.full_list[-1*args.batch_size:]
    total_files = len(batch.full_list)
    print("Total number of files to read: %d" %total_files)
    for epoch in range(args.startfrom,args.num_epochs):
        print("===================== Epoch %d =====================" %(epoch))
        batch.seq_length = args.seq_length + epoch * 3 - 30
        random.shuffle(batch.full_list)
        batch.next_epoch(args.batch_size) # initialize batch data
        batch.initialize_states(args.num_layers, args.hidden_size)
        total_steps = step
        step=0
        while(batch.epoch_end==0):
            step+=1
            
            # print(' '.join([str(len(x)) for x in batch.batch_data]))
            # update the minibatch inputs / outputs
            encoder.zero_grad()
            decoder.zero_grad()
            # get next minibatch
            batch.get_minibatch(0)
            
            batch.next_minibatch()

            if batch.epoch_end==1:
                break            
            
            # get inputs and targets from batch object
            batch.oov2idx_list = [dict() for i in range(len(batch.batch_in))]
            batch.idx2oov_list = [dict() for i in range(len(batch.batch_in))]

            for i in range(len(batch.batch_in)):
                in_line = batch.batch_in[i]
                out_line = batch.batch_out[i]
                oov2idx, idx2oov = vocab.create_oov_list(in_line+out_line, batch.max_oovs)
                batch.oov2idx_list[i] = oov2idx
                batch.idx2oov_list[i] = idx2oov

            # make sure all lengths of batch_in and batch_out are same
            inputs_oov_np = np.array([vocab.word_list_to_idx_list(line,batch.oov2idx_list[i]) for
                                 i,line in enumerate(batch.batch_in)],dtype=int)
            targets_oov_np = np.array([vocab.word_list_to_idx_list(line,batch.oov2idx_list[i]) for
                                 line in batch.batch_out],dtype=int)
            inputs_unk_np = np.array([vocab.word_list_to_idx_list(line) for
                                 i,line in enumerate(batch.batch_in)],dtype=int)
            targets_unk_np = np.array([vocab.word_list_to_idx_list(line) for
                                 line in batch.batch_out],dtype=int)
            
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
            
            loss = criterion(torch.log(outputs), targets.view(-1))
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm(encoder.parameters(),0.5)
            torch.nn.utils.clip_grad_norm(decoder.parameters(),0.5)
            
            opt_e.step()
            opt_d.step()
            

            print ('Epoch [%d/%d], Files: [%d/%d],  Loss: %.3f, Steps: [%d/%d], Perplexity: %5.2f'
                   %(epoch, args.num_epochs, total_files-len(batch.file_list),
                total_files, loss.data[0], step, total_steps, np.exp(loss.data[0])))
            # save intermediate model
            if step % 100==0:
                print("Saving model at %d steps..." %step)
                torch.save(f=os.path.join(save_dir,'saved_encoder_%d_epochs.pckl')%(epoch),obj=encoder)
                torch.save(f=os.path.join(save_dir,'saved_decoder_%d_epochs.pckl')%(epoch),obj=decoder)
                print("Model saved.\n")
        # save model at end of each epoch
        torch.save(f=os.path.join(save_dir,'saved_encoder_%d_epochs.pckl')%(epoch),obj=encoder)
        torch.save(f=os.path.join(save_dir,'saved_decoder_%d_epochs.pckl')%(epoch),obj=decoder)
        
def test(args, batch, vocab, model, save_dir):
    
    # split model into encoder and decoder
    encoder,decoder = model
    
    # set loss function and optimizer
    criterion = nn.NLLLoss()
    step = 99999
    total_files = len(batch.full_list)
    print("Total number of files to read: %d" %total_files)
    for epoch in range(1):
        batch.next_epoch(args.batch_size) # initialize batch data
        batch.initialize_states(args.num_layers, args.hidden_size)
        total_steps = step
        step=0
        while(batch.epoch_end==0):
            step+=1
            total = 0
            correct = 0
            # get next minibatch
            batch.get_minibatch(0)
            batch.next_minibatch()
            
            if batch.epoch_end==1:
                break            
            
            # get inputs and targets from batch object
            for i in range(len(batch.batch_in)):
                in_line = batch.batch_in[i]
                out_line = batch.batch_out[i]
                oov2idx, idx2oov = vocab.create_oov_list(in_line+out_line, batch.max_oovs)
                batch.oov2idx_list.append(oov2idx)
                batch.idx2oov_list.append(idx2oov)
            
            # make sure all lengths of batch_in and batch_out are same
            inputs_oov_np = np.array(
                [vocab.word_list_to_idx_list(line,batch.oov2idx_list[i]) 
                 for i,line in enumerate(batch.batch_in)],dtype=int)
            targets_oov_np = np.array(
                [vocab.word_list_to_idx_list(line,batch.oov2idx_list[i]) 
                 for i,line in enumerate(batch.batch_out)],dtype=int)
            inputs_unk_np = np.array([vocab.word_list_to_idx_list(line) 
                for i,line in enumerate(batch.batch_in)],dtype=int)
            targets_unk_np = np.array([vocab.word_list_to_idx_list(line) 
                for i,line in enumerate(batch.batch_out)],dtype=int)
            
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
                
                if teacher_force==True:
                    decoder_in = targets[:,j] # train with ground truth
                else:
                    decoder_in = outputs[:,-1,:].max(1)[1].squeeze()
            
            targets = Variable(torch.LongTensor(targets_oov_np)).cuda()
            out = outputs.max(2)[1] # purely for printing purposes
            # print("======================= Inputs  =======================")
            # print(batch.batch_in[0])
            # print('\n')            
            # print(' '.join([str(x) for x in inputs_oov_np[0]]))
            # print(' '.join(vocab.idx_list_to_word_list(inputs_oov_np[0],
            #                                            batch.idx2oov_list[0])))
            # print("======================= Targets =======================")
            # print(batch.idx2oov_list[0])
            # print(batch.oov2idx_list[0])
            # print(batch.batch_out[0])
            # print('\n')
            # print(' '.join([str(x) for x in targets_oov_np[0]]))
            # print(' '.join(vocab.idx_list_to_word_list(targets_oov_np[0],
            #                                            batch.idx2oov_list[0])))
            # print("======================= Predict =======================")
            # print(' '.join([str(x) for x in out[0].data.cpu().numpy()]))
            # print(' '.join(vocab.idx_list_to_word_list(out[0].data.cpu().numpy(),
            #                                            batch.idx2oov_list[0])))
            # print('\n')

            targets, outputs = pack_padding(targets, outputs)
            loss = criterion(torch.log(outputs), targets.view(-1))

            out = outputs.max(1)[1].unsqueeze(1)
            total_unk = (targets==vocab.w2i['<UNK>']).data.cpu().numpy().sum()
            correct_unk = ((targets==vocab.w2i['<UNK>'])*(targets==out)).data.cpu().numpy().sum()
            total += targets.size(0) - total_unk
            correct += (targets==out).data.cpu().numpy().sum() - correct_unk

            print ('Epoch [%d/%d], Files: [%d/%d],  Accuracy: %.3f, Perplexity: %5.2f' 
                   % (epoch+args.startfrom, args.num_epochs, total_files-len(batch.file_list),
                total_files, correct/total, np.exp(loss.data[0])))
            # save intermediate model
                
def autocomplete(args, batch, vocab, model, save_dir):
    
    # split model into encoder and decoder
    encoder,decoder = model

    from packages.functions import lex
    while(1):
        print('\n')
        input_seq = input("Input code: ")
        print('\n')
        print('============================ Input ============================\n')
        print(input_seq.replace(';',';\n').replace('}','}\n').replace('{','\n{'))
        input_seq = lex(input_seq,'value')
        oov2idx, idx2oov = vocab.create_oov_list(input_seq, batch.max_oovs)   
        inputs_oov_np = np.array(vocab.word_list_to_idx_list(input_seq,oov2idx),dtype=int)
        inputs_unk_np = np.array(vocab.word_list_to_idx_list(input_seq),dtype=int).reshape([1,-1])
        
        # dummy introduced to make up with batch issues
        dummy = np.zeros(inputs_oov_np.shape, dtype=int)
        inputs_oov_np = np.vstack([inputs_oov_np,dummy])
        inputs_unk_np = np.vstack([inputs_unk_np,dummy])

        teacher_force = False
                
        inputs = Variable(torch.LongTensor(inputs_unk_np)).cuda()
        targets = Variable(torch.LongTensor(inputs.size()).zero_()).cuda()
        # run model to get outputs
        encoded, _ = encoder(inputs)
            
        decoder_in, s, w = decoder_initial(inputs.size(0),vocab.w2i['<SOS>'])
            
        for j in range(args.out_seq):
            if j==0:
                outputs,s,w = decoder(input_idx=decoder_in, encoded=encoded,
                    encoded_idx=inputs_oov_np,prev_state=s,
                    weighted=w, order=j)
            else:
                tmp_out,s,w = decoder(input_idx=decoder_in, encoded=encoded,
                    encoded_idx=inputs_oov_np, prev_state=s,
                    weighted=w, order=j)
                outputs = torch.cat([outputs,tmp_out],dim=1)

            if teacher_force==True:
                decoder_in = targets[:,j] # train with ground truth
            else:
                decoder_in = outputs[:,-1,:].max(1)[1].squeeze()
        # print(outputs.max(2)[1].data.cpu().numpy()[0])
        out = outputs.max(2)[1].data.cpu().numpy()[0]
        pred = []
        for i in out:
            if i==vocab.w2i['<EOS>']:
                break
            pred.append(i)
            if i==vocab.w2i[';']:
                break
            
        print("\n============================ Output ===========================\n", 
              ' '.join(vocab.idx_list_to_word_list(pred)))
