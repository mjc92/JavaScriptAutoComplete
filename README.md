# JavaScript Text Autocomplete model using encoder-decoder neural network w/ copy-based attention

## Description

A model that takes in a portion of previous Javascript codes (~50 tokens) to complete the next sentence (~10 tokens or until a semicolon)

## Model baseline
- model that originates from CopyNet [[link](http://www.aclweb.org/anthology/P16-1154)]
  - seq2seq encoder-decoder model that predicts the next sequence from either generating a new token from scratch or referring to a part in the input to copy relevant parts
  - compared to baseline models, **can handle with OOV words by copying them (less-frequently used IDs in source codes (800,000+))**

## Other ideas for an autocompletion tool
1) when predicting the first token (given input SOS), if a partial string was given as input (e.g. having to predict the next
word 'string' and typed in 'st'), select from the available tokens starting with 'st' to output the next word
2) 

## Experiments
1) next-line prediction
  - give a number of (~50) previous tokens from a code
  - predict up to the next 10 tokens (seq2seq / encoder-decoder)
  - stop if a semicolon (';') is predicted as it stands for EOL (end-of-line) 


## Performance measure
- Accuracy (how many tokens are correctly predicted)
  - Accuracy for identifiers (can our model correctly place an identifier?)
  - Accuracy for non-identifier tokens (can our model understand code structure?)
- Perplexity (how efficiently can our model cut down the number of possibilies for the next tokens?)

## Dataset
- 150k JavaScript Dataset [[link](http://www.srl.inf.ethz.ch/js150.php)]
- 99,815 / 49,000 .js files for training / testing
- Vocabulary size: 50,000
  - 86 tokens for code ('{', '[', ';', 'var' etc.)
  - 1,000 frequently used regex expressions
  - 48,000 most used identifier names (out of 800,000+)
  - all remaining expressions are considered as 'UNK'

## Baselines
1) biGRU-based encoder-decoder
2) biGRU-based encoder-decoder w/ attention
3) Sparse Pointer Network? [[link](https://arxiv.org/abs/1611.08307)]
4) n-gram based language models? (bit outdated though...)
5) other baselines recommended
  - note that since our task is the **first to do line-level code prediction** instead of single-token prediction, it may be insufficient to used SOTA code-completion models that predict the next AST node

## (Expected) Results
1) Our model has less training / testing perplexity compared to baseline models
2) Our model has higher non-identifier & identifier prediction accuracies
3) Our model can copy infrequent tokens from the input and use them as the output wherease baseline models only predict UNK tokens
4) Our model can complete all the way to a semicolon to end a statement

## FAQ
1) Why not use syntax information provided from abstract syntax trees (ASTs)?
  - While ASTs provide structural information of what to use, in many situations it is impossible to create them from
  an incomplete piece of code, as is seen in real-life code completion situations. Many papers that use ASTs for code completion therefore cannot predict from a raw piece of source code, but instead predict the next node from a partial AST [([link](https://openreview.net/pdf?id=rJbPBt9lg)], [[link](http://www.srl.inf.ethz.ch/papers/oopsla16-dt.pdf)]).
  - Meanwhile, our model can be applied to any incomplete piece of code without having to construct a particular AST, and can obtain similar structural information using a lexer. This goes under the assumption that a RNN-based model can learn the structural information of a code **even only with lexed results** (e.g. closing brackets, ending with a semicolon)

2) How is it different from Sparse Pointer Networks [[link](https://arxiv.org/abs/1611.08307)]?
  - SPNs focus on predicting Python code; ours is on JavaScript, a more frequently used programming language
  - In the dataset used for SPNs, **the original data is not preserved** because all identifier names are changed to anonymized versions such as 'attribute184' or 'function42'. However, our model can successfully use unknown tokens by assigning them temporary OOV vocab indices and copying them from the input sequence. Also, the names of identifiers such as functions should be preserved as a specific function determines the structure of the subsequent tokens to be used (e.g. the function 'sum' should end as 'sum()', whereas 'exp' should have an argument 'exp(x)')
  - Our model does not explicitly store a memory of the previous identifiers. Rather, it uses the previous input sequence as a kind of a lookup memory where relevant parts can be copied to the output. This is a more flexible setting as there is no need to manage a fixed memory set.

## Minor issues
1) CopyNet is taking too long to train... (13+ hrs per epoch)
