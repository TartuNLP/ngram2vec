#!/usr/bin/env python3

import re
import sys
import logging

import ngram

from gensim.models import Word2Vec

#raw / use factor X
#filter freqs
#filter factors

if __name__ == "__main__":
	dataFile = sys.argv[1]
	modelFile = sys.argv[2]
	
	srcFactor = None
	freqFilter = [5, 30, 50]
	firstPosFilter = None
	lastPosFilter = None
	somePosFilter = None
	beta = 0.125
	
	logging.basicConfig(level = logging.INFO)

	lines = ngram.SentenceNgramSampler(dataFile, minCounts = freqFilter, srcFactor = srcFactor, firstPosFilter = firstPosFilter, lastPosFilter = lastPosFilter, atLeastOnePosFilter = somePosFilter, ngramThresholdBeta = beta)

	model = Word2Vec(lines, workers=20, sg=1, hs=1, iter=10)

	model.wv.save_word2vec_format(modelFile, binary = True)
