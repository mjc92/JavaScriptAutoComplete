import argparse
import os
from slimit.lexer import Lexer

def lex(text,output_type='all'):
    # lexes string, stops when it starts making TypeErrors
    # input: f.read()
    out_list = []
    out_list2 = []
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
            elif output_type=='both':
                try:
                    out_list.append(token.value)
                    out_list2.append(token.type)
                except AttributeError:
                    break
            else:
                try:
                    out_list.append(token)
                except AttributeError:
                    break
        except AttributeError:
            break
        except TypeError:
            break
    if output_type=='both':
        return out_list, out_list2
    return out_list

# arguments related to the dataset
parser = argparse.ArgumentParser()

# about running mode
parser.add_argument("--startfrom",type=int, default=0, help='which file to start from')
parser.add_argument("--file_range",type=int, default=100, help='how many files to look at')
parser.add_argument("--ref_range",type=int, default=10, help='how many lines to refer to')
args = parser.parse_args()

file_dir = '/home/irteam/users/data/150kJavaScript/train/data_lexed'
save_dir = '/home/irteam/users/mjchoi/github/JavaScriptAutoComplete/analysis/sentence_similarities'

score_list = []
id_cnt = 0
str_cnt = 0
id_cnt_correct = 0
str_cnt_correct =0

found_list = []
unfound_list = []

for i in range(args.file_range):
    print("====================================================================")
    print(i)
    print("====================================================================")
    with open(os.path.join(file_dir,'file_%d.txt'%(i+1+args.startfrom))) as f:
        lines = f.read()
    lines = lines.replace(';',';\n').replace('}','}\n').replace('{','\n{').split('\n')
    lines = [x for x in lines if x!=' ']
    if len(lines)<=args.ref_range:
        continue
    for i in range(len(lines)-args.ref_range):
        refs = []
        for j in range(args.ref_range):
            refs.extend(lex(lines[i+j],'value'))
        hyp = lex(lines[i+args.ref_range])
        if len(hyp)==0:
            continue
        for tok in hyp:
            if tok.type=='ID':
                id_cnt+=1
                if tok.value in refs:
                    id_cnt_correct+=1
                    found_list.append(tok.value)
                else:
                    unfound_list.append(tok.value)
            elif tok.type=='STRING':
                str_cnt+=1
                if tok.value in refs:
                    str_cnt_correct+=1
                    found_list.append(tok.value)
                else:
                    unfound_list.append(tok.value)



with open(os.path.join(save_dir,'found_%d_to_%d_ref_%d.txt'%(args.startfrom+1,args.startfrom+args.file_range,
                                                            args.ref_range)),'w') as f:
    out = '\n'.join([str(x) for x in found_list])
    f.write(out)

with open(os.path.join(save_dir,'unfound_%d_to_%d_ref_%d.txt'%(args.startfrom+1,args.startfrom+args.file_range,
                                                            args.ref_range)),'w') as f:
    out = '\n'.join([str(x) for x in unfound_list])
    f.write(out)

with open(os.path.join(save_dir,'log.txt'),'a') as f:
    out = "%d\t%d\t%d\t%d\t%d\t%d\n"%(args.startfrom+1,args.startfrom+args.file_range,args.ref_range,
                               id_cnt_correct,id_cnt,str_cnt_correct,str_cnt)
    f.write(out)

