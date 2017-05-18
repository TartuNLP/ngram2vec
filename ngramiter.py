import re
import spacy
import numpy as np
import random

from collections import defaultdict, deque, Counter

class CorpusNgramIterator:
	numData = []
	ngramBuffer = deque()
	currSnt = 0
	
	wordBits = 20
	
	__UNK__ = 1
	
	word2idx = { '__UNK__': 1 }
	idx2word = {}
	
	nidx2list = {}
	hash2nidx = {}
	
	nlp = spacy.load('en')
	
	batchCount = 0
	corpusPassCount = 0
	
	thisIsIt = False
	
	def __init__(self, filename, miniBatchSize, minCounts = [5, 50, 80], contextWidth = 5, numSamples = 5, stopAtNBatches = False, stopAtNCorpusPasses = 5):
		self.miniBatchSize = miniBatchSize
		self.contextWidth = contextWidth
		self.numSamples = numSamples
		self.maxNgramLen = len(minCounts)
		
		self.stopAtNCorpusPasses = stopAtNCorpusPasses
		self.stopAtNBatches = stopAtNBatches
		
		print("Reading data")
		self.readRawData(filename, minCounts)
		print("Done")
	
	def getVocSize(self):
		return len(self.nidx2list)
			
	def readRawData(self, filename, minCounts):
		"""
		read the file, compose a dictionary of counts and indexes and save data as indexes
		"""
		freqDict = self.getFreqDict(filename)
		
		words = ["__NONE__", "__UNK__"] + [word for word in sorted(freqDict, key=lambda x: -freqDict[x]) if (freqDict[word] >= minCounts[0]) and (re.search(r'\w', word))]
		
		self.idx2word = dict(enumerate(words))
		
		self.word2idx = dict([(w, i) for i, w in enumerate(words)])
		
		self.hash2nidx = { i: i for i in range(len(words)) }
		self.nidx2list = { i: [i,] for i in range(len(words)) }
		
		ngramFreqDict = self.numsAndNgrams(filename, freqDict)
		
		#for x in ngramFreqDict[1]:
		#	if ngramFreqDict[1][x] >= minCounts[0]:
		#		print("DDD", "_".join([self.idx2word[i] for i in self.hash2list(x)]), ngramFreqDict[1][x])
		
		#for x in ngramFreqDict[2]:
		#	if ngramFreqDict[2][x] >= minCounts[0]:
		#		print("DDDD", "_".join([self.idx2word[i] for i in self.hash2list(x)]), ngramFreqDict[2][x])
		
		baseIdx = len(words)
		
		print("DEBUG words:", baseIdx)
		
		for nlen in sorted(ngramFreqDict):
			ndict = ngramFreqDict[nlen]
			
			hashVals = [hashVal for hashVal in sorted(ndict, key=lambda x: -ndict[x]) if ndict[hashVal] >= minCounts[nlen]]
			nidxs = list(range(baseIdx, baseIdx + len(hashVals)))
			
			thisHash2nidx = dict(zip(hashVals, nidxs))
			
			thisNidx2list = { nidx: self.hash2list(hashVal) for (nidx, hashVal) in zip(nidxs, hashVals) }
			
			self.hash2nidx.update(thisHash2nidx)
			self.nidx2list.update(thisNidx2list)
			
			print("DEBUG ngrams len", nlen, ":", len(hashVals))
			
			baseIdx += len(hashVals)
	
	def getDicts(self):
		return { 'word2idx': self.word2idx,
			'idx2word': self.idx2word,
			'nidx2list': self.nidx2list,
			'hash2nidx': self.hash2nidx }
	
	def toks(self, line, pos=False):
		doc = self.nlp.tokenizer(line.lower())
		
		if pos:
			self.nlp.tagger(doc)
			return [(w.text, w.pos_) for w in doc]
		else:
			return [w.text for w in doc]
		
	
	def getFreqDict(self, filename):
		fh = open(filename, 'r')
		result = Counter([token for line in fh for token in self.toks(line) if re.search(r'\w', token)])
		fh.close()
		return result
	
	def numsAndNgrams(self, filename, freqDict):
		ngramFreqDict = defaultdict(lambda: defaultdict(int))
		
		fltPos = set(['ADJ', 'NOUN', 'VERB', 'ADV', 'PROPN'])
		
		with open(filename, 'r', encoding='utf8') as fh:
			for line in fh:
				tokens = self.toks(line, pos=True)
				
				numSentence = [(self.word2idx[token] if token in self.word2idx else self.__UNK__) for token, pos in tokens if re.search(r'\w', token)]
				poses = [pos for token, pos in tokens if re.search(r'\w', token)]
				self.numData.append(numSentence)
				
				for sntIdx in range(len(numSentence)):
					for ngramLen in range(1, self.maxNgramLen):
						if sntIdx - ngramLen >= 0:
							tokenIdxs = [numSentence[i] for i in range(sntIdx - ngramLen, sntIdx + 1)]
							tokenPoss = set([poses[i] for i in range(sntIdx - ngramLen, sntIdx + 1)])
							
							if (not self.__UNK__ in tokenIdxs) and (tokenPoss & fltPos):
								ngramFreqDict[ngramLen][self.list2hash(tokenIdxs)] += 1
		
		return ngramFreqDict
	
	def prepareDicts(self, minCounts, idxFreq):
		self.idx2word = dict(zip(self.word2idx.values(), self.word2idx.keys()))
		
		for idx, freq in idxFreq[0].items():
			if freq < minCounts[0]:
				del self.idx2word[idx]
			else:
				self.ngram2raw[idx] = idx
		
		numWords = len(self.idx2word) + 1
		
		for ngramLen, minCount in enumerate(minCounts[1:]):
			for rawIdx, freq in idxFreq[ngramLen].items():
				if freq >= minCount:
					cleanIdx = numWords
					numWords += 1
					
					self.ngram2raw[cleanIdx] = rawIdx
					self.raw2ngram[rawIdx] = cleanIdx
	
	def generateSkipGramPairs(self, seq):
		seqLen = len(seq)
		
		for idx, elem in enumerate(seq):
			maxLen = self.maxNgramLen if self.corpusPassCount >= 5 else 1
			for ngramLen in range(self.maxNgramLen):
				if idx + ngramLen < seqLen:
					rIdx = idx + ngramLen + 1
					
					rawCore = seq[idx: rIdx]
					try:
						hashVal = self.list2hash(rawCore)
						coreIdx = self.hash2nidx[hashVal]
						
						context = seq[max(idx - self.contextWidth, 0): idx] + seq[rIdx: min(rIdx + self.contextWidth, seqLen)]
						
						for contextIdx in random.sample(context, min(self.numSamples, len(context))):
							if not contextIdx in self.idx2word:
								contextIdx = 100 + contextIdx
							
							yield (coreIdx, contextIdx)
					except KeyError:
						pass
	
	def fillBuffer(self):
		skipGrams = list(self.generateSkipGramPairs(self.numData[self.currSnt]))
		
		random.shuffle(skipGrams)
		
		self.ngramBuffer.extend(skipGrams)
		
		self.currSnt = self.currSnt + 1
		
		if self.currSnt >= len(self.numData):
			self.currSnt = 0
			self.corpusPassCount += 1
			
			if self.stopAtNCorpusPasses and self.corpusPassCount >= self.stopAtNCorpusPasses:
				print("corpus passes =", self.corpusPassCount)
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
		
		return (batch, labels)
		
	def __iter__(self):
		return self
	
	def hash2list(self, hashVal):
		runningIdx = hashVal
		result = []
		
		while runningIdx > 0:
			result.append(runningIdx % (1 << self.wordBits))
			runningIdx = runningIdx >> self.wordBits
		
		return result
	
	def list2hash(self, wordIdxList):
		currBits = 0
		
		result = 0
		
		for wordIdx in wordIdxList:
			result += wordIdx << currBits
			currBits += self.wordBits
		
		return result
