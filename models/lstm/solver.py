import torch
from torch import nn, optim
from torch.autograd import Variable
from packages.functions import detach, pack_padding
import numpy as np
import os

def train(args, batch, vocab, model, save_dir):
    # args: from main
    # batch: a Batch object
    # vocab: a Vocab object
    # model: loaded from main
    # epoch: current epoch
    # save_dir: which directory you'll be saving
    
    # set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    step = 99999
    for epoch in range(args.startfrom,args.num_epochs):
        print("=========================== Epoch %d ===========================" %(epoch+args.startfrom))

        batch.next_epoch(args.batch_size) # initialize batch data
        batch.initialize_states(args.num_layers, args.hidden_size)
        total_steps = step
        step=0
        while(batch.epoch_end==0):
            step+=1
            # update the minibatch inputs / outputs
            model.zero_grad()
            # get next minibatch
            batch.get_minibatch(0)
            # get inputs and targets from batch object
            inputs_np = np.array([vocab.word_list_to_idx_list(line) for
                                 line in batch.batch_in],dtype=int)
            targets_np = np.array([vocab.word_list_to_idx_list(line) for
                                 line in batch.batch_out],dtype=int)
            inputs = Variable(torch.LongTensor(inputs_np)).cuda()
            targets = Variable(torch.LongTensor(targets_np)).cuda()

            # run model to get outputs
            outputs, states = model(inputs,batch.states)
            batch.states = detach(states)
            targets, outputs = pack_padding(targets, outputs)
            loss = criterion(outputs, targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(),0.5)
            optimizer.step()

            batch.next_minibatch()
            if step%10==0:
                print ('Epoch [%d/%d], Loss: %.3f, Steps: [%d/%d], Perplexity: %5.2f' %
               (epoch+args.startfrom, args.num_epochs, loss.data[0], step, total_steps, np.exp(loss.data[0])))
        # save model at end of each epoch
        torch.save(f=os.path.join(save_dir,'saved_model_%d_epochs.pckl')%(epoch+args.startfrom),obj=model)
        
def test(args, batch, vocab, model, save_dir):
    # args: from main
    # batch: a Batch object
    # vocab: a Vocab object
    # model: loaded from main
    # epoch: current epoch
    # save_dir: which directory you'll be saving
    
    # set loss function and optimizer
    print("=========================== Testing ===========================")
    initial_files = len(batch.full_list)
    overall_correct = 0
    overall_total = 0
    overall_correct_tok = 0
    overall_correct_id = 0
    overall_total_tok = 0
    overall_total_id = 0
    
    batch.next_epoch(args.batch_size) # initialize batch data
    batch.initialize_states(args.num_layers, args.hidden_size)
    while(batch.epoch_end==0):
        # get next minibatch
        batch.get_minibatch(0)
        # get inputs and targets from batch object
        inputs_np = np.array([vocab.word_list_to_idx_list(line) for
                             line in batch.batch_in],dtype=int)
        targets_np = np.array([vocab.word_list_to_idx_list(line) for
                             line in batch.batch_out],dtype=int)
        inputs = Variable(torch.LongTensor(inputs_np)).cuda()
        targets = Variable(torch.LongTensor(targets_np)).cuda()

        # run model to get outputs
        outputs, states = model(inputs,batch.states)
        input_line = vocab.idx_list_to_word_list(inputs_np[0])
        target_line = vocab.idx_list_to_word_list(targets_np[0])
        output_line = vocab.idx_list_to_word_list(outputs[0].max(1)[1].cpu().data.numpy())
        for tup in zip(input_line, target_line, output_line):
            print(' '.join(list(tup)))
        
        batch.states = detach(states)
        targets, outputs = pack_padding(targets, outputs)
        outputs = outputs.view(targets.size(0),-1).max(1)[1]
        t = targets.squeeze().data.cpu().numpy()
        o = outputs.data.cpu().numpy()
        correct = np.array(t==o,dtype=int)
        correct_tok = np.array(np.multiply(correct,(t<86)), dtype=int)
        total_tok = np.array(t<86, dtype=int)
        correct_id = np.array(np.multiply(correct,(t>=1086)), dtype=int)
        total_id = np.array(t>=1086,dtype=int)
        # for tup in zip(list(correct_tok),list(total_tok),list(correct_id),list(total_id)):
        #     a,b,c,d = tup
        #     tup = (a,b,c,d)
        #     print(' '.join([str(x) for x in list(tup)]))
        total = targets.size(0)
        
        overall_correct += correct.sum()
        overall_correct_tok += correct_tok.sum()
        overall_correct_id += correct_id.sum()
        overall_total += total
        overall_total_tok += total_tok.sum()
        overall_total_id += total_id.sum()
        # print(targets.cpu().data==outputs.cpu().data)
        print('Overall: %d/%d, %1.3f' % (correct.sum(),total,correct.sum()/total*100.0))
        print('Tokens : %d/%d, %1.3f' % (correct_tok.sum(),total_tok.sum(),(correct_tok.sum()/total_tok.sum()*100.0)))
        print('IDs    : %d/%d, %1.3f' % (correct_id.sum(),total_id.sum(),(correct_id.sum()/total_id.sum()*100.0)))
        print('\n\n')
        batch.next_minibatch()
        print("%d/%d files read" %(initial_files-len(batch.file_list),initial_files))
    print("Final..........\n")
    print('Overall: %d/%d, %1.3f' % (overall_correct,overall_total,(overall_correct/overall_total*100.0)))
    print('Tokens : %d/%d, %1.3f' % (overall_correct_tok,overall_total_tok,(overall_correct_tok/overall_total_tok*100.0)))
    print('IDs    : %d/%d, %1.3f' % (overall_correct_id,overall_total_id,(overall_correct_id/overall_total_id*100.0)))
    
