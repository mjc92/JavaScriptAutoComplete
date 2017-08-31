# import os
# import re
# from collections import Counter
# import slimit
# from slimit.lexer import Lexer
# from functions import lex

# # _actual is the result of removing all missing and undecodable files
# file_dir = '/home/irteam/users/data/150kJavaScript/'
# lexer = Lexer()

# with open(file_dir+'programs_training_actual.txt') as f:
#     file_list=f.readlines()
    
# counter = Counter()
# for i,file in enumerate(file_list):
#     with open(os.path.join(file_dir,file).strip()) as f:
#         text = f.read()
#         words = lex(text)
#         counter += Counter(words)
#     if i%1000==0:
#         print("\n%d files collected so far\n" %i)
# print("All files collected")
# print("Files that could not be lexed: %d" %len(lex_prob_list))
# with open('lex_prob_list.txt','w') as f:
#     f.write('\n'.join(lex_prob_list))
# import pickle
# f = open("counter.pckl",'wb')
# pickle.dump(counter,f)

import pickle
special_tokens = ['<PAD>','<UNK>','<SOS>','<EOS>','<CAM>','<LOW>']
with open('counter.pckl','rb') as f:
    counter = pickle.load(f)
vocab_sizes = [1000,5000,10000,50000]
for vocab_size in vocab_sizes:
    w2i = dict()
    i2w = dict()
    for i,word in enumerate(special_tokens):
        w2i[word]=i
        i2w[i]=word
    for i,tup in enumerate(counter.most_common()[:vocab_size-len(special_tokens)]):
        word,_ = tup
        w2i[word] = i+len(special_tokens)
        i2w[i+len(special_tokens)]=word
    with open('word2idx_%d.pckl'%vocab_size,'wb') as f:
        pickle.dump(w2i,f)
    with open('idx2word_%d.pckl'%vocab_size,'wb') as f:
        pickle.dump(i2w,f)
    print("Saved vocab of %d words"%vocab_size)
