#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Mark Fishel

import re
import numpy as np
import random
import pickle

import logging
logger = logging.getLogger('ngram iter')

from collections import defaultdict, deque, Counter

class CorpusNgramIterator:
	batchCount = 0
	epochCount = 0
	
	ngramDict = defaultdict(lambda: defaultdict(int))
	
	data = []
	
	currSntIdx = 0
	currGenIdx = 0

	def __init__(self, filename, minCounts = [5, 50, 80], numSntGen = 3, ngramInclThreshold = 0.5):
		# read corpus once to fill the ngram dict
		self.maxNgramLen = len(minCounts)
		self.numSntGen = numSntGen
		self.ngramInclThreshold = ngramInclThreshold
		
		self._readData(filename)
		
		self._filterNgrams(minCounts)
	
	def _readData(self, filename):
		logger.info("Reading data")
		
		with open(filename, 'r') as fh:
			idx = 0
			
			for rawline in fh:
				sentence = [t for t in rawline.strip().lower().split() if re.search(r'\w', t)]
				
				self.data.append(sentence)
				
				for ngram, spec in self.ngrams(sentence):
					self.ngramDict[len(spec) - 1][ngram] += 1
				
				idx += 1
				
				if not idx % 10000:
					logger.info("Read {0} sentences".format(idx))
	
	def _filterNgrams(self, minCounts):
		logger.info("Filtering n-grams")
		
		for nlen in range(1, self.maxNgramLen):
			before = len(self.ngramDict[nlen])
			self.ngramDict[nlen] = { k: v for k, v in self.ngramDict[nlen].items() if v > minCounts[nlen] }
			after = len(self.ngramDict[nlen])
			logger.info("Filtered n-grams of length {0} from {1} down to {2}".format(nlen + 1, before, after))
	
	def ngrams(self, srcSnt):
		for idx, tok in enumerate(srcSnt):
			for nlen in range(1, self.maxNgramLen):
				if idx - nlen >= 0:
					spec = range(idx - nlen, idx + 1)
					
					yield "__".join([srcSnt[i] for i in spec]), set(spec)
	
	def _getNonoverlappingNgrams(self, ngramsAndSpecs):
		result = []
		covMap = set()
		
		for ngram, spec in ngramsAndSpecs:
			if ngram in self.ngramDict[len(spec) - 1] and random.random() < self.ngramInclThreshold and not (spec & covMap):
				result.append(spec)
				covMap.update(spec)
		
		return result
	
	def _applyJoinOps(self, sentence, toJoin):
		result = sentence
		
		for op in sorted(toJoin, key=lambda x: -min(x)):
			result = result[:min(op)] + ["__".join([sentence[i] for i in sorted(op)])] + result[max(op)+1:]
		
		return result
	
	def __next__(self):
		try:
			srcSnt = self.data[self.currSntIdx]
		except IndexError:
			self.currSntIdx = 0
			self.currGenIdx = 0
			raise StopIteration
		
		ngramsAndSpecs = list(self.ngrams(srcSnt))
		random.shuffle(ngramsAndSpecs)
		
		if len([1 for n, s in ngramsAndSpecs if n in self.ngramDict[len(s)-1]]) > 0:
			self.currGenIdx += 1
			
			if self.currGenIdx >= self.numSntGen:
				self.currGenIdx = 0
				self.currSntIdx += 1
			
			toJoin = self._getNonoverlappingNgrams(ngramsAndSpecs)
			
			
			result = self._applyJoinOps(srcSnt, toJoin)
			
			return result
		else:
			self.currSntIdx += 1
			self.currGenIdx = 0
			
			return srcSnt
	
	def __iter__(self):
		return self