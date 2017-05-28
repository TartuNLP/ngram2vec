#!/usr/bin/env python3

import pickle
import sys
import numpy as np

def hash2list(hashVal, wordBits = 20):
	runningIdx = hashVal
	result = []
	
	while runningIdx > 0:
		result.append(runningIdx % (1 << wordBits))
		runningIdx = runningIdx >> wordBits
	
	return result

def list2hash(wordIdxList, wordBits = 20):
	currBits = 0
	
	result = 0
	
	for wordIdx in wordIdxList:
		result += wordIdx << currBits
		currBits += wordBits
	
	return result

class Ngram2Vec:
	def __init__(self, filename):
		with open(filename, 'rb') as inFh:
			rawModel = pickle.load(inFh)
			self.__dict__.update(rawModel['dicts'])
			self.embeddings = np.array(rawModel['embeddings'])
	
	def idx2ngram(self, nidx):
		return [self.idx2word[wIdx] for wIdx in self.nidx2list[nidx]]
	
	def ngram2idx(self, words):
		wordIdxs = [self.word2idx[w] for w in words]
		hsh = list2hash(wordIdxs)
		return self.hash2nidx[hsh]
	
	def most_similar_idx(self, ngramIdx, k = 5):
		v = self.embeddings[ngramIdx]
		
		sim = np.dot(self.embeddings, v)
		
		idxs = enumerate(sim)
		
		if k == 1:
			return [ max(idxs, key=lambda x: 0 if x[0] == ngramIdx else x[1]) ]
		else:
			idxs = sorted(idxs, key=lambda x: -x[1])
			return idxs[1:1+k]
	
	def most_similar(self, ngram, k = 5):
		rawRes = self.most_similar_idx(self.ngram2idx(ngram), k)
		
		return [(self.idx2ngram(w), f) for w, f in rawRes]
	
	def __contains__(self, ngram):
		try:
			_ = self.ngram2idx(ngram)
			return True
		except KeyError as e:
			return False
	
	def __getitem__(self, ngram):
		return self.embeddings[self.ngram2idx(ngram)]
