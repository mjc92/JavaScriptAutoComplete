import os
import pickle
import re
from collections import Counter
import slimit
import argparse
from slimit.lexer import Lexer
from functions import lex


def custom_lex(text):
    # lexes string, stops when it starts making TypeErrors
    # input: f.read()
    out_list = []
    id_list = []
    tok_list = []
    str_list = []
    num_list = []
    lexer = Lexer()
    lexer.input(text)
    while True:
        try:
            token = lexer.token()
            if not token:
                break # break if end of token
            tok_type = token.type
            if tok_type=='ID':
                id_list.append(token.value)
            elif tok_type=='STRING':
                str_list.append(token.value)
            elif tok_type=='NUMBER':
                num_list.append(token.value)
            else:
                tok_list.append(token.value)
            out_list.append(token.value)
        except TypeError:
            break
        except AttributeError:
            break
    return out_list,id_list,tok_list, str_list, num_list


# arguments related to the dataset
parser = argparse.ArgumentParser()

parser.add_argument("--mode",type=str, help='whether "train" or "test" data')
parser.add_argument("--dataset_name",type=str, help='name of preprocessed dataset')
parser.add_argument("--startfrom",type=int, default=0, help='whether starting from middle')

args = parser.parse_args()

# _actual is the result of removing all missing and undecodable files
file_dir = '/home/irteam/users/data/150kJavaScript/'
vocab_dir = '/home/irteam/users/mjchoi/github/JavaScriptAutoComplete/vocab/'
lexer = Lexer()

save_dir = os.path.join(file_dir,args.mode,args.dataset_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
if args.mode=='train':
    list_name = 'programs_training_actual.txt'
elif args.mode=='test':
    list_name = 'programs_eval_actual.txt'
else:
    print("Insufficient mode !")
    import sys
    sys.exit()

vocab_dir = vocab_dir+args.dataset_name
if not os.path.exists(vocab_dir):
    os.makedirs(vocab_dir)
    
with open(file_dir+list_name) as f:
    file_list=f.readlines()
if args.mode=='train':
    if args.startfrom==0:
        id_counter = Counter()
        tok_counter = Counter()
        str_counter = Counter()
        num_counter = Counter()
    else:
        with open(os.path.join(vocab_dir,'id_counter.pckl'),'rb') as f:
            id_counter = pickle.load(f)
        with open(os.path.join(vocab_dir,'tok_counter.pckl'),'rb') as f:
            tok_counter = pickle.load(f)
        with open(os.path.join(vocab_dir,'str_counter.pckl'),'rb') as f:
            str_counter = pickle.load(f)
        with open(os.path.join(vocab_dir,'num_counter.pckl'),'rb') as f:
            num_counter = pickle.load(f)        
for i,file in enumerate(file_list[args.startfrom:]):
    with open(os.path.join(file_dir,file).strip()) as f:
        text = f.read()
    print("processing %s\n" %(file))    
    text = text.split(' ')
    print("file length: %d tokens" % len(text))
    if len(text)>30000:
        text = text[:30000]
    text = ' '.join(text)
    out_list, id_list, tok_list, str_list, num_list=custom_lex(text)
    if args.mode=='train':
        id_counter += Counter(id_list)
        tok_counter += Counter(tok_list)
        str_counter = Counter(str_list)
        num_counter = Counter(num_list)
    with open(os.path.join(save_dir,'file_%d.txt'%(i+1+args.startfrom)),'w') as f:
        f.write(' '.join(out_list))
    print("%d files processed so far\n" %(i+1+args.startfrom))
    
    if ((i+1)%1000==0) & (args.mode=='train'):
        with open(os.path.join(vocab_dir,'id_counter.pckl'),'wb') as f:
            pickle.dump(id_counter,f)
        with open(os.path.join(vocab_dir,'tok_counter.pckl'),'wb') as f:
            pickle.dump(tok_counter,f)
        with open(os.path.join(vocab_dir,'str_counter.pckl'),'wb') as f:
            pickle.dump(str_counter,f)
        with open(os.path.join(vocab_dir,'num_counter.pckl'),'wb') as f:
            pickle.dump(num_counter,f)
        print("\n%d files collected so far\n" %(i+1+args.startfrom))

if args.mode=='train':
    with open(os.path.join(vocab_dir,'id_counter.pckl'),'wb') as f:
        pickle.dump(id_counter,f)
    with open(os.path.join(vocab_dir,'tok_counter.pckl'),'wb') as f:
        pickle.dump(tok_counter,f)
    with open(os.path.join(vocab_dir,'str_counter.pckl'),'wb') as f:
        pickle.dump(str_counter,f)
    with open(os.path.join(vocab_dir,'num_counter.pckl'),'wb') as f:
        pickle.dump(num_counter,f)

out_file_list = os.listdir(os.path.join(file_dir,args.mode,args.dataset_name))
with open(os.path.join(file_dir,args.mode,'file_list_'+args.dataset_name+'.txt'),'w') as f:
    f.write('\n'.join(out_file_list))