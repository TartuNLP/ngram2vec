#!/usr/bin/env python3

import re
import sys
import logging

import ngram

from gensim.models import Word2Vec

dataFile = sys.argv[1]
modelFile = sys.argv[2]

logging.basicConfig(level = logging.INFO)

lines = ngram.SentenceNgramSampler(dataFile)

model = Word2Vec(lines, workers=16, sg=1, hs=1, iter=25)

model.wv.save_word2vec_format(modelFile, binary = True)
