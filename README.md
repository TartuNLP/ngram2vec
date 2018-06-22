# ngram2vec
Embeddings for n-grams via sampling.

## Learning word2vec model with word2vec (gensim word2vec)
Parameters for extraction are inside the learnmdl.py, they can be changed in the file.

```console
$ python3 learnmdl.py preproc.data.en model.en
```
Now model.en can be loaded in and used with python (gensim word2vec model).

## Extracting only ngrams
Extracting only ngrams is good, because mainly fasttext with python is very slow and C++ code is used to train. Additionally, adds modularity - extracting ngrams once and then training whatever embeddings: different embeddings (word2vec, glove, fasttext etc) or different hyperparameters...

### Example use with fasttext (C++ compiled):

```console
$ python3 extract_ngrams.py data.clean.en data.ngrams.en
$ ./fasttext cbow -input data.ngram.en  -thread 16 -dim 300 ngram.mdl.d3.en 
```
