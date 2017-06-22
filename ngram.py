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
		srcSnt = self._getNextSentence()
		
		#apply ngram joining and return result
		
		ngramsAndSpecs = list(self.ngrams(srcSnt))
		random.shuffle(ngramsAndSpecs)
		
		toJoin = self._getNonoverlappingNgrams(ngramsAndSpecs)
		
		result = self._applyJoinOps(srcSnt, toJoin)
		
		return result
	
	def _maybeReadFilter(self, rawFilterSpec):
		if rawFilterSpec is None:
			return None
		else:
			return set(rawFilterSpec.split(","))
	
	def _cleanSentence(self, rawSnt):
		halfReady = [t for t in rawSnt.strip().lower().split() if re.search(r'\w', t)]
		
		if self.tokFactor is None:
			return halfReady, None
		else:
			factors = [t.split("|") for t in halfReady]
			
			tokResult = map(itemgetter(self.tokFactor), factors)
			
			if self.posFactor is None:
				posResult = None
			else:
				posResult = map(itemgetter(self.posFactor), factors)
			
			return tokResult, posResult
	
	def _getNextSentence(self):
		try:
			#either first iteration, or re-reading the file every time
			if (self.currSntIdx is None):
				rawSnt = next(self.fileHandle)
				srcSnt, fltFactors = self._cleanSentence(rawSnt)
			
			#or reading from data in memory
			else:
				srcSnt = self.storedData[self.currSntIdx]
				fltFactors = None
				self.currSntIdx += 1
		
		#ran out of data
		except IndexError:
			self.currSntIdx = 0
			raise StopIteration
		
		#end of file
		except StopIteration:
			self.fileHandle.close()
			
			if self.firstIter:
				self._filterDict()
			
			self.firstIter = False
			
			if self.crazyBigMFCorpus:
				self.fileHandle = open(self.filename, 'r')
			else:
				self.currSntIdx = 0
				
			raise StopIteration
		
		if self.firstIter:
			#update ngram freq counter
			for ngram, spec in self.ngrams(srcSnt):
				nlen = len(spec) - 1
				
				if self._acceptableNgram(fltFactors, spec):
					self.ngramDict[nlen][ngram] += 1
			
			if not self.crazyBigMFCorpus:
				self.storedData.append(srcSnt)
		
		#ready
		return srcSnt
	
	def _acceptableNgram(self, fltFactors, ngramSpec):
		if self.posFactor is None:
			return True
		
		factors = [fltFactors[i] for i in sorted(ngramSpec)]
		
		return (self.firstPosFilter is None or factors[0] in self.firstPosFilter) and (self.lastPosFilter is None or factors[-1] in self.lastPosFilter) and (self.atLeastOnePosFilter is None or set(factors) & self.atLeastOnePosFilter)
	
	def _filterDict(self):
		for nlen in self.ngramDict:
			before = len(self.ngramDict[nlen])
			self.ngramDict[nlen] = { k: v for k, v in self.ngramDict[nlen].items() if v >= self.minCounts[nlen] }
			after = len(self.ngramDict[nlen])
			logger.info("Filtered {0}-grams from {1} down to {2}".format(nlen + 1, before, after))
	
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
			nlen = len(spec) - 1
			
			if ngram in self.ngramDict[nlen] and self.ngramDict[nlen][ngram] >= self.minCounts[nlen]:
				if not (spec & covMap):
					threshold = math.exp((-math.log(self.ngramDict[nlen][ngram]))*self.ngramThresholdBeta)
					
					if random.random() < threshold:
						result.append(spec)
						covMap.update(spec)
		
		return result
	
	def _applyJoinOps(self, sentence, toJoin):
		result = sentence
		
		for op in sorted(toJoin, key=lambda x: -min(x)):
			result = result[:min(op)] + ["__".join([sentence[i] for i in sorted(op)])] + result[max(op)+1:]
		
		return result
	
	def __iter__(self):
		return self
