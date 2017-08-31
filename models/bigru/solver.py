# solver for copynet
import torch
from torch import nn, optim
from torch.autograd import Variable
from packages.functions import detach, pack_padding, decoder_initial
import numpy as np
import os

def train(args, batch, vocab, model, save_dir):
    
    # set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    step = 99999
    total_files = len(batch.full_list)
    
    print("Total number of files to read: %d" %total_files)
    for epoch in range(args.startfrom,args.num_epochs):
        print("===================== Epoch %d =====================" %(epoch))

        batch.next_epoch(args.batch_size) # initialize batch data
        batch.initialize_states(args.num_layers, args.hidden_size)
        total_steps = step
        step=0
        while(1):
            step+=1
            # update the minibatch inputs / outputs
            model.zero_grad()
            # get next minibatch
            batch.get_minibatch(0)
            batch.next_minibatch()
            
            if batch.epoch_end==1:
                break            
            
            # make sure all lengths of batch_in and batch_out are same
            inputs_np = np.array([vocab.word_list_to_idx_list(line) for
                                 i,line in enumerate(batch.batch_in)],dtype=int)
            targets_np = np.array([vocab.word_list_to_idx_list(line) for
                                 line in batch.batch_out],dtype=int)
            
            # whether to use teacher forcing
            if np.random.random_sample(size=1)[0]<(epoch*1.0/args.num_epochs):
                teacher_force = True
            else:
                teacher_force = False
                
            inputs = Variable(torch.LongTensor(inputs_np)).cuda()
            targets = Variable(torch.LongTensor(targets_np)).cuda()

            outputs = model(inputs,targets,teacher_force)
            
            targets, outputs = pack_padding(targets, outputs)
            loss = criterion(outputs, targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(),0.5)
            optimizer.step()

            print ('Epoch [%d/%d], Files: [%d/%d],  Loss: %.3f, Steps: [%d/%d],Perplexity: %5.2f' 
                   %(epoch, args.num_epochs, total_files-len(batch.file_list),
                total_files, loss.data[0], step, total_steps, np.exp(loss.data[0])))
            if step % 100==0:
                torch.save(f=os.path.join(save_dir,'saved_model_%d_epochs.pckl')%(epoch+args.startfrom),obj=model)
        # save model at end of each epoch
        torch.save(f=os.path.join(save_dir,'saved_model_%d_epochs.pckl')%(epoch+args.startfrom),obj=model)
        
def test(args, batch, vocab, model, save_dir):
    
    # set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    total_files = len(batch.full_list)
    
    print("Total number of files to read: %d" %total_files)
    batch.next_epoch(args.batch_size) # initialize batch data
    batch.initialize_states(args.num_layers, args.hidden_size)
    overall_total = 0
    overall_correct = 0
    while(1):
        # update the minibatch inputs / outputs
        # get next minibatch
        total = 0
        correct = 0
        batch.get_minibatch(0)
        batch.next_minibatch()

        if batch.epoch_end==1:
            break            

        # make sure all lengths of batch_in and batch_out are same
        inputs_np = np.array([vocab.word_list_to_idx_list(line) for
                             i,line in enumerate(batch.batch_in)],dtype=int)
        targets_np = np.array([vocab.word_list_to_idx_list(line) for
                             line in batch.batch_out],dtype=int)

        # whether to use teacher forcing
        teacher_forcing=False

        inputs = Variable(torch.LongTensor(inputs_np)).cuda()
        targets = Variable(torch.LongTensor(targets_np)).cuda()

        outputs = model(inputs,targets,teacher_forcing)
        idx = 0
        # print("====================== Input ======================") 
        # print(inputs_np[idx])
        # in_string = ' '.join(vocab.idx_list_to_word_list(list(inputs_np[idx])))
        # print(in_string.replace('}','}\n').replace('{','\n{').replace(';',';\n'))
        # print("====================== Output =====================") 
        # print(' '.join([x for x in vocab.idx_list_to_word_list(list(targets_np[idx]))]))
        # print("====================== Predicted ==================") 
        # print(outputs.max(2)[1].data.cpu().numpy()[idx])
        # print(' '.join(vocab.idx_list_to_word_list(
        #     list(outputs.max(2)[1].data.cpu().numpy()[idx]))))
        # print('\n')

        targets, outputs = pack_padding(targets, outputs)
        loss = criterion(outputs, targets.view(-1))
        
        out = outputs.max(1)[1].unsqueeze(1)
        
        total_unk = (targets==vocab.w2i['<UNK>']).data.cpu().numpy().sum()
        correct_unk = ((targets==vocab.w2i['<UNK>'])*(targets==out)).data.cpu().numpy().sum()
        # total += targets.size(0) - total_unk
        total += targets.size(0) - total_unk
        correct += (targets==out).data.cpu().numpy().sum() - correct_unk
        print("%d/%d, accuracy: %1.3f, perplexity: %1.3f" \
              %(correct,total,correct/total,np.exp(loss.data[0])))
        
        overall_total += total
        overall_correct += correct
    # after reading all files, calculate scores
    print("Total: %d/%d, accuracy: %1.3f" 
          %(overall_correct,overall_total,overall_correct/overall_total))

def autocomplete(args, batch, vocab, model, save_dir):
    
    from packages.functions import lex
    while(1):
        print('\n')
        input_seq = input("Input code: ")
        print('\n')
        print('============================ Input ============================\n')
        print(input_seq.replace(';',';\n').replace('}','}\n').replace('{','\n{'))
        input_seq = lex(input_seq,'value')
        inputs_np = np.array(vocab.word_list_to_idx_list(input_seq),dtype=int).reshape(1,-1)
        dummy = np.zeros(inputs_np.shape, dtype=int)
        inputs_np = np.vstack([inputs_np,dummy])
        # whether to use teacher forcing
        teacher_forcing=False

        inputs = Variable(torch.LongTensor(inputs_np)).cuda()
        targets = Variable(torch.LongTensor(2,batch.out_seq).zero_()).cuda()
        outputs = model(inputs,targets,teacher_forcing)        
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
