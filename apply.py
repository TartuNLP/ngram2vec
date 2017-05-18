#!/usr/bin/env python3

import json
import sys
import numpy as np

class Ngram2Vec:
	def __init__(self, filename):
		with open(filename, 'r') as inFh:
			rawModel = json.load(inFh)
			self.__dict__.update(rawModel['dicts'])
			self.embeddings = np.array(rawModel['embeddings'])
	
	def wordform(self, nidx):
		return " ".join([self.idx2word[str(wIdx)] for wIdx in self.nidx2list[str(nidx)]])
	
	def most_similar(self, wordIdx, k = 10):
		v = self.embeddings[wordIdx]
		sim = np.dot(self.embeddings, v)
		
		if k == 1:
			return max(...
		else:
			idxs = sorted(range(len(mdl.nidx2list)), key=lambda i: -sim[i])
			return idxs[1:1+k]
