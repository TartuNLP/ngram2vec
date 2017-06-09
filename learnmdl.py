#!/usr/bin/env python3

import re
import sys
import logging

import ngramiter

from gensim.models import Word2Vec

dataFile = sys.argv[1]
modelFile = sys.argv[2]

logging.basicConfig(level = logging.INFO)

lines = ngramiter.CorpusNgramIterator(dataFile)

model = Word2Vec(lines, workers=16, sg=1, hs=1)

model.wv.save_word2vec_format(modelFile, binary = True)
