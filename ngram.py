#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Mark Fishel

import re
import numpy as np
import random
import pickle
import math

import logging

from collections import defaultdict, deque, Counter
from operator import itemgetter

logger = logging.getLogger('ngram iter')

class SentenceNgramSampler:
	batchCount = 0
	epochCount = 0
	
	ngramDict = defaultdict(lambda: defaultdict(int))
	
	currSntIdx = None
	storedData = []
	
	firstIter = True

	def __init__(self, filename, minCounts = [None, 30, 50], ngramThresholdBeta = 0.125,
				firstPosFilter = None, lastPosFilter = None, atLeastOnePosFilter = None,
				crazyBigMFCorpus = False, tokFactor = None, posFactor = None):
		
		self.maxNgramLen = len(minCounts)
		self.minCounts = minCounts
		
		self.ngramThresholdBeta = ngramThresholdBeta
		
		self.firstPosFilter = self._maybeReadFilter(firstPosFilter)
		self.lastPosFilter = self._maybeReadFilter(lastPosFilter)
		self.atLeastOnePosFilter = self._maybeReadFilter(atLeastOnePosFilter)
		
		self.tokFactor = tokFactor
		self.posFactor = posFactor
		
		self.crazyBigMFCorpus = crazyBigMFCorpus
		self.filename = filename
		
		self.fileHandle = open(filename, 'r')
	
	def __next__(self):
		result = None
		
		while not result:
			factoredSnt = self._getNextSentence()
			
			ngramsAndSpecs = list(self.ngrams(factoredSnt))
			random.shuffle(ngramsAndSpecs)
			
			toJoin = self._getNonoverlappingNgrams(ngramsAndSpecs)
			
			result = self._applyJoinOps(factoredSnt, toJoin)
		
		return result
	
	def _maybeReadFilter(self, rawFilterSpec):
		if rawFilterSpec is None:
			return None
		else:
			return set(rawFilterSpec.split(","))
	
	def _getFactors(self, rawToken):
		if self.tokFactor is None:
			return (rawToken, None)
		else:
			factors = rawToken.split("|")
			
			f1 = factors[self.tokFactor].lower()
			f2 = None if self.posFactor is None else factors[self.posFactor]
			
			return (f1, f2)
	
	def _cleanSentence(self, rawSnt):
		result = [self._getFactors(t) for t in rawSnt.strip().split()]
		
		return [(t, p) for t, p in result if re.search(r'[a-zäöüõšž]', t)]
	
	def _tryGetNext(self):
		#either first iteration, or re-reading the file every time
		if (self.currSntIdx is None):
			rawSnt = next(self.fileHandle)
			return self._cleanSentence(rawSnt)
		
		#or reading from data in memory
		else:
			snt = self.storedData[self.currSntIdx]
			self.currSntIdx += 1
			return snt
	
	def _handleEndOfFile(self):
		self.fileHandle.close()
		
		if self.firstIter:
			self._filterDict()
		
		self.firstIter = False
		
		if self.crazyBigMFCorpus:
			self.fileHandle = open(self.filename, 'r')
		else:
			self.currSntIdx = 0
			
		raise StopIteration
	
	def _handleEndOfList(self):
		self.currSntIdx = 0
		raise StopIteration
	
	def _updateNgramDict(self, fsnt):
		#update ngram freq counter
		for ngram, spec in self.ngrams(fsnt):
			nlen = len(spec) - 1
			
			if self._acceptableNgram(fsnt, spec):
				self.ngramDict[nlen][ngram] += 1
	
	def _getNextSentence(self):
		try:
			factoredSnt = self._tryGetNext()
		
		except IndexError:
			self._handleEndOfList()
		
		except StopIteration:
			self._handleEndOfFile()
		
		if self.firstIter:
			self._updateNgramDict(factoredSnt)
			
			if not self.crazyBigMFCorpus:
				self.storedData.append(factoredSnt)
		
		return factoredSnt
	
	def _acceptableNgram(self, fsnt, ngramSpec):
		if self.posFactor is None:
			return True
		
		factors = [fsnt[i][1] for i in sorted(ngramSpec)]
		
		firstOk = (self.firstPosFilter is None or factors[0] in self.firstPosFilter)
		lastOk = (self.lastPosFilter is None or factors[-1] in self.lastPosFilter)
		someOk = (self.atLeastOnePosFilter is None or set(factors) & self.atLeastOnePosFilter)
		
		result = (firstOk and lastOk and someOk)
		
		#print(factors, firstOk, lastOk, someOk, result)
		
		return result
	
	def _filterDict(self):
		for nlen in self.ngramDict:
			before = len(self.ngramDict[nlen])
			self.ngramDict[nlen] = { k: v for k, v in self.ngramDict[nlen].items() if v >= self.minCounts[nlen] }
			after = len(self.ngramDict[nlen])
			logger.info("Filtered {0}-grams from {1} down to {2}".format(nlen + 1, before, after))
	
	def ngrams(self, fSnt):
		for idx in range(len(fSnt)):
			for nlen in range(1, self.maxNgramLen):
				if idx - nlen >= 0:
					spec = range(idx - nlen, idx + 1)
					
					yield "__".join([fSnt[i][0] for i in spec]), set(spec)
	
	def _getNonoverlappingNgrams(self, ngramsAndSpecs):
		result = []
		covMap = set()
		
		for ngram, spec in ngramsAndSpecs:
			nlen = len(spec) - 1
			
			if ngram in self.ngramDict[nlen] and self.ngramDict[nlen][ngram] >= self.minCounts[nlen]:
				if not (spec & covMap):
					threshold = math.exp((-math.log(self.ngramDict[nlen][ngram]))*self.ngramThresholdBeta)
					
					if random.random() < threshold:
						result.append(spec)
						covMap.update(spec)
		
		return result
	
	def _applyJoinOps(self, sentence, toJoin):
		result = [t for t, _ in sentence]
		
		for op in sorted(toJoin, key=lambda x: -min(x)):
			result = result[:min(op)] + ["__".join([sentence[i][0] for i in sorted(op)])] + result[max(op)+1:]
		
		return result
	
	def __iter__(self):
		return self
