#!/usr/bin/env python3

import sys
import pickle
from ngramiter import CorpusNgramIterator
from trainvec import trainEmbeddings

import logging

if __name__ == "__main__":
	logging.basicConfig(level=logging.DEBUG)
	
	ngramIter = CorpusNgramIterator(sys.argv[1], 128, minCounts = [5, 40, 80], stopAtNCorpusPasses = 20)
	
	embeddings = trainEmbeddings(ngramIter.getVocSize(), ngramIter)
	
	with open(sys.argv[2], 'wb') as outFh:
		pickle.dump({ 'embeddings': embeddings.tolist(), 'dicts': ngramIter.getDicts() }, outFh, pickle.HIGHEST_PROTOCOL)
