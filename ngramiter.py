import re
import numpy as np
import random
import pickle

from nvecs import hash2list, list2hash

import logging
logger = logging.getLogger('ngram iter')

from collections import defaultdict, deque, Counter

class CorpusNgramIterator:
	numData = []
	ngramBuffer = deque()
	currSnt = 0
	
	wordBits = 20
	
	__UNK__ = 1
	
	word2idx = { '__UNK__': __UNK__ }
	idx2word = {}
	
	nidx2list = {}
	hash2nidx = {}
	
	batchCount = 0
	epochCount = 0
	
	thisIsIt = False
	
	def __init__(self, filename, miniBatchSize, minCounts = [5, 50, 80], contextWidth = 5, numSamples = 5, stopAtNBatches = False, stopAtNEpochs = 5):
		self.miniBatchSize = miniBatchSize
		self.contextWidth = contextWidth
		self.numSamples = numSamples
		self.maxNgramLen = len(minCounts)
		
		self.stopAtNEpochs = stopAtNEpochs
		self.stopAtNBatches = stopAtNBatches
		
		self.readRawData(filename, minCounts)
	
	def getVocSize(self):
		return len(self.nidx2list)
			
	def readRawData(self, filename, minCounts):
		"""
		read the file, compose a dictionary of counts and indexes and save data as indexes
		"""
		logger.info("Counting words (pass 1)")
		freqDict = self.countWords(filename)
		
		logger.info("Filtering word dicts")
		wordList = self.getFilteredWordList(freqDict, minCounts[0])
		
		self.initWordDicts(wordList)
		
		logger.info("Reading data as indexes (pass 2)")
		self.readIndexedTaggedCorpus(filename)
		
		logger.info("Counting n-grams")
		ngramFreqDict = self.countNgrams()
		
		logger.info("Filtering n-gram dicts")
		self.initNgramDicts(minCounts, ngramFreqDict)
	
	def getDicts(self):
		return { 'word2idx': self.word2idx,
			'idx2word': self.idx2word,
			'nidx2list': self.nidx2list,
			'hash2nidx': self.hash2nidx }
	
	def toks(self, line, pos=False):
		if pos:
			#return [rawTok.split("|") for rawTok in line.lower().strip().split(" ")]
			return [(w, None) for w in line.lower().strip().split(" ")]
		else:
			#return [rawTok.split("|")[0] for rawTok in line.lower().strip().split(" ")]
			return line.lower().strip().split(" ")
	
	def countWords(self, filename):
		with open(filename, 'r') as fh:
			result = Counter([token for line in fh for token in self.toks(line) if re.search(r'\w', token)])
			logger.debug("Number of all words: {0}".format(len(result)))
			return result
	
	def getFilteredWordList(self, freqDict, wordMinCounts):
		# dict header -- not using index 0 for hashing purposes
		# sort by decreasing frequency
		# filter out words below the threshold
		# filter out words without alphanumeric characters
		result = ["__NONE__", "__UNK__"] + \
			[word for word in sorted(freqDict, key=lambda x: -freqDict[x]) \
			if (freqDict[word] >= wordMinCounts) \
			and (re.search(r'\w', word))]
		
		logger.debug("Number of filtered words: {0}".format(len(result)))
		
		return result
	
	def initWordDicts(self, wordList):
		self.idx2word = dict(enumerate(wordList))
		
		self.word2idx = dict([(w, i) for i, w in enumerate(wordList)])
		
		idxs = range(len(wordList))
		
		self.hash2nidx = { i: i for i in idxs }
		self.nidx2list = { i: [i,] for i in idxs }
	
	def readIndexedTaggedCorpus(self, filename):
		self.numData = list() 
		
		with open(filename, 'r', encoding='utf8') as fh:
			for line in fh:
				tokens = self.toks(line, pos=True)
				
				filtTaggedIdxs = [( self.word2idx.get(word) or self.__UNK__, tag ) \
						for word, tag in tokens if re.search(r'\w', word)]
				
				self.numData.append(filtTaggedIdxs)
	
	def countNgrams(self):
		ngramFreqDict = defaultdict(lambda: defaultdict(int))
		
		#fltPos = set(['adj', 'noun', 'verb', 'adv', 'propn'])
		fltPos = set(['adj', 'noun', 'adv', 'propn'])
		
		for taggedIndexes in self.numData:
			for sntIdx in range(len(taggedIndexes)):
				for ngramLen in range(1, self.maxNgramLen):
					if sntIdx - ngramLen >= 0:
						ngramIdxs, ngramTags = zip(*[taggedIndexes[i] for i in range(sntIdx - ngramLen, sntIdx + 1)])
						ngramTagSet = set(ngramTags)
						
						#if (not self.__UNK__ in ngramIdxs) and (ngramTagSet & fltPos):
						if (not self.__UNK__ in ngramIdxs):
							hashVal = list2hash(ngramIdxs, self.wordBits)
							ngramFreqDict[ngramLen][hashVal] += 1
		
		for ngramLen in range(1, self.maxNgramLen):
			logger.debug("Number of all n-grams of length {0}: {1}".format(ngramLen, len(ngramFreqDict[ngramLen])))
		
		return ngramFreqDict
	
	def initNgramDicts(self, minCounts, ngramFreqDict):
		baseIdx = len(self.word2idx)
		
		for nlen in range(1, self.maxNgramLen):
			ndict = ngramFreqDict[nlen]
			
			total = len(ndict)
			hashVals = [hashVal for hashVal in sorted(ndict, key=lambda x: -ndict[x]) if ndict[hashVal] >= minCounts[nlen]]
			logger.debug("Filtered length-{0} ngrams from {1} down to {2}".format(nlen, total, len(hashVals)))
			
			nidxs = list(range(baseIdx, baseIdx + len(hashVals)))
			
			thisHash2nidx = dict(zip(hashVals, nidxs))
			
			thisNidx2list = { nidx: hash2list(hashVal, self.wordBits) for (nidx, hashVal) in zip(nidxs, hashVals) }
			
			self.hash2nidx.update(thisHash2nidx)
			self.nidx2list.update(thisNidx2list)
			
			baseIdx += len(hashVals)
	
	def generateSkipGramPairs(self, seq):
		seqLen = len(seq)
		
		for idx, elem in enumerate(seq):
			if self.epochCount > 1:
				maxLen = self.maxNgramLen
			elif self.epochCount > 0:
				maxLen = min(self.maxNgramLen, 2)
			else:
				maxLen = 1
			
			for ngramLen in range(maxLen):
				if idx + ngramLen < seqLen:
					rIdx = idx + ngramLen + 1
					
					rawCore = seq[idx: rIdx]
					try:
						hashVal = list2hash(rawCore)
						coreIdx = self.hash2nidx[hashVal]
						
						context = seq[max(idx - self.contextWidth, 0): idx] + seq[rIdx: min(rIdx + self.contextWidth, seqLen)]
						
						for contextIdx in random.sample(context, min(self.numSamples, len(context))):
							if not contextIdx in self.idx2word:
								contextIdx = 100 + contextIdx
							
							yield (coreIdx, contextIdx)
					except KeyError:
						pass
	
	def fillBuffer(self):
		skipGrams = list(self.generateSkipGramPairs([w for w, t in self.numData[self.currSnt]]))
		
		random.shuffle(skipGrams)
		
		self.ngramBuffer.extend(skipGrams)
		
		self.currSnt = self.currSnt + 1
		
		if self.currSnt >= len(self.numData):
			self.currSnt = 0
			self.epochCount += 1
			logger.debug("Finished epoch number {0}".format(self.epochCount))
			
			if self.stopAtNEpochs and self.epochCount >= self.stopAtNEpochs:
				print("corpus passes =", self.epochCount)
				self.thisIsIt = True
	
	def getNextSkipGramPair(self):
		retry = 100
		
		while retry > 0:
			try:
				res = self.ngramBuffer.popleft()
			except IndexError:
				retry -= 1
				self.fillBuffer()
				continue
			else:
				break
		
		if retry == 0:
			raise Exception("Failed to refill the buffer after 100 attempts")
		
		return res
		
	def __next__(self):
		if self.stopAtNBatches and self.batchCount >= self.stopAtNBatches:
			print("minibatches =", self.batchCount)
			self.thisIsIt = True
		
		if self.thisIsIt:
			raise StopIteration
		
		batch = np.ndarray(shape=(self.miniBatchSize), dtype=np.int32)
		labels = np.ndarray(shape=(self.miniBatchSize, 1), dtype=np.int32)
		
		for i in range(self.miniBatchSize):
			batch[i], labels[i, 0] = self.getNextSkipGramPair()
		
		self.batchCount += 1
		
		if not self.batchCount % 10000:
			logger.debug("Finished batch number {0}".format(self.batchCount))
		
		return (batch, labels)
		
	def __iter__(self):
		return self
