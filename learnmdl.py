#!/usr/bin/env python3

import re
import sys
import logging

import ngram

from datetime import datetime
from gensim.models import Word2Vec

def debug(msg):
	sys.stderr.write("{0}: {1}\n".format(str(datetime.now()), msg))

if __name__ == "__main__":
	dataFile = sys.argv[1]
	modelFile = sys.argv[2]
	
	#Factored Estonian data:
	#tokFactor = 0
	#posFactor = 2
	#firstPosFilter = "A,S,H"
	#lastPosFilter = "S,H"
	
	#Unfactored data:
	tokFactor = None
	posFactor = None
	firstPosFilter = None
	lastPosFilter = None
	
	freqFilter = [5, 30, 50, 70, 90]
	somePosFilter = None
	crazyBigMFCorpus = False
	beta = 0.125
	epochs = 0
	
	logging.basicConfig(level = logging.INFO)

	lines = ngram.SentenceNgramSampler(dataFile, minCounts = freqFilter, tokFactor = tokFactor, posFactor = posFactor, firstPosFilter = firstPosFilter, lastPosFilter = lastPosFilter, atLeastOnePosFilter = somePosFilter, ngramThresholdBeta = beta, crazyBigMFCorpus = crazyBigMFCorpus)
	
	if len(freqFilter) > 1:
		debug("Initializing")
		for line in lines:
			pass

	if epochs > 0:
		model = Word2Vec(workers=60, sg=1, hs=1, iter=10, min_count=freqFilter[0])
		
		debug("Building vocab")
		model.build_vocab(lines)
		
		debug("Learning")
		for i in range(epochs):
			model.train(lines, total_examples = len(lines), epochs = 1)
			model.save(modelFile + ".trainable." + str(i))
			model.wv.save_word2vec_format(modelFile + "." + str(i), binary = True)
			debug("Iteration {0} done".format(i))
