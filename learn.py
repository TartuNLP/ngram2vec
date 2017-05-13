#!/usr/bin/env python3

import re
import sys
import math
import random

import numpy as np
import tensorflow as tf

from collections import defaultdict, deque, Counter
from six.moves import xrange

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
	
	def __init__(self, filename, miniBatchSize, minCounts = [5, 50, 50], contextWidth = 2, numSamples = 4):
		self.miniBatchSize = miniBatchSize
		self.contextWidth = contextWidth
		self.numSamples = numSamples
		self.maxNgramLen = len(minCounts)
		
		self.readRawData(filename, minCounts)
	
	def getVocSize(self):
		return len(self.nidx2list)
			
	def readRawData(self, filename, minCounts):
		"""
		read the file, compose a dictionary of counts and indexes and save data as indexes
		"""
		freqDict = self.getFreqDict(filename)
		
		words = [None, "__UNK__"] + [word for word in sorted(freqDict, key=lambda x: -freqDict[x]) if freqDict[word] >= minCounts[0]]
		
		self.idx2word = dict(enumerate(words))
		
		self.word2idx = dict([(w, i) for i, w in enumerate(words)])
		
		self.hash2nidx = { i: i for i in range(len(words)) }
		self.nidx2list = { i: [i,] for i in range(len(words)) }
		
		ngramFreqDict = self.numsAndNgrams(filename, freqDict)
		
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
	
	def getFreqDict(self, filename):
		fh = open(filename, 'r')
		result = Counter([token for line in fh for token in line.split()])
		fh.close()
		return result
	
	def numsAndNgrams(self, filename, freqDict):
		ngramFreqDict = defaultdict(lambda: defaultdict(int))
		
		with open(filename, 'r', encoding='utf8') as fh:
			for line in fh:
				tokens = line.split()
				
				numSentence = [(self.word2idx[token] if token in self.word2idx else self.__UNK__) for token in tokens]
				self.numData.append(numSentence)
				
				for sntIdx in range(len(numSentence)):
					for ngramLen in range(1, self.maxNgramLen):
						if sntIdx - ngramLen >= 0:
							tokenIdxs = [numSentence[i] for i in range(sntIdx - ngramLen, sntIdx + 1)]
							
							if not self.__UNK__ in tokenIdxs:
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
		
		self.currSnt = (self.currSnt + 1) % len(self.numData)
	
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
		batch = np.ndarray(shape=(self.miniBatchSize), dtype=np.int32)
		labels = np.ndarray(shape=(self.miniBatchSize, 1), dtype=np.int32)
		
		for i in range(self.miniBatchSize):
			batch[i], labels[i, 0] = self.getNextSkipGramPair()
		
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

def trainEmbeddings(vocabulary_size, miniBatchIterator, embedding_size = 256, num_iters = 1e6):
	batch_size = miniBatchIterator.miniBatchSize
	#embedding_size = 256  # Dimension of the embedding vector.
	#skip_window = 2       # How many words to consider left and right.
	#num_skips = 2         # How many times to reuse an input to generate a label.

	# We pick a random validation set to sample nearest neighbors. Here we limit the
	# validation samples to the words that have a low numeric ID, which by
	# construction are also the most frequent.
	valid_size = 32     # Random set of words to evaluate similarity on.
	valid_window = 1000  # Only pick dev samples in the head of the distribution.
	valid_examples = np.random.choice(valid_window, valid_size, replace=False)
	num_sampled = 128    # Number of negative examples to sample.

	graph = tf.Graph()

	with graph.as_default():

		# Input data.
		train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
		train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
		valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

		# Ops and variables pinned to the CPU because of missing GPU implementation
		with tf.device('/cpu:0'):
			# Look up embeddings for inputs.
			embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
			embed = tf.nn.embedding_lookup(embeddings, train_inputs)

			# Construct the variables for the NCE loss
			nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
			nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

		# Compute the average NCE loss for the batch.
		# tf.nce_loss automatically draws a new sample of the negative labels each
		# time we evaluate the loss.
		loss = tf.reduce_mean(
			tf.nn.nce_loss(weights=nce_weights,
								biases=nce_biases,
								labels=train_labels,
								inputs=embed,
								num_sampled=num_sampled,
								num_classes=vocabulary_size))

		# Construct the SGD optimizer using a learning rate of 1.0.
		optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

		# Compute the cosine similarity between minibatch examples and all embeddings.
		norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
		normalized_embeddings = embeddings / norm
		valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
		similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

		# Add variable initializer.
		init = tf.global_variables_initializer()

	with tf.Session(graph=graph) as session:
		# We must initialize all variables before we use them.
		init.run()
		print('Initialized')

		average_loss = 0
		for step in xrange(num_iters):
			batch_inputs, batch_labels = next(miniBatchIterator)
		 
			feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

			# We perform one update step by evaluating the optimizer op (including it
			# in the list of returned values for session.run()
			_, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
			average_loss += loss_val

			if step % 10000 == 0:
				if step > 0:
					average_loss /= 10000
				# The average loss is an estimate of the loss over the last 2000 batches.
				print('Average loss at step ', step, ': ', average_loss)
				average_loss = 0

			# Note that this is expensive (~20% slowdown if computed every 500 steps)
			if step % 100000 == 0:
				sim = similarity.eval()
				for i in xrange(valid_size):
					valid_word = reverse_dictionary[valid_examples[i]]
					top_k = 8  # number of nearest neighbors
					nearest = (-sim[i, :]).argsort()[1:top_k + 1]
					log_str = 'Nearest to %s:' % valid_word
					for k in xrange(top_k):
						close_word = reverse_dictionary[nearest[k]]
						log_str = '%s %s,' % (log_str, close_word)
					print(log_str)
		final_embeddings = normalized_embeddings.eval()
		print("done")

	s = time.time()
	simDemand = np.dot(final_embeddings, final_embeddings[ngram_dict[str([dictionary['down'], dictionary['the'], dictionary['road']])]])
	e = time.time()
	print("time to multiply:", e - s)
	s = e
	simDemandWords = [(reverse_dictionary[i], s) for i, s in enumerate(simDemand)]
	e = time.time()
	print("time to label:", e - s)
	s = e
	simDemandWords.sort(key=lambda x: -x[1])
	e = time.time()
	print("time to sort:", e - s)
	s = e
	print([w for w, _ in simDemandWords[:10]])

	import json
	with open('3grams-emb256-w2.dat', 'w') as fh:
		 json.dump([dictionary, reverse_dictionary, ngram_dict, final_embeddings.tolist()], fh)

if __name__ == "__main__":
	ngramIter = CorpusNgramIterator(sys.argv[1], 16, minCounts = [5, 5, 5], contextWidth = 1, numSamples = 1)
	
	embeddings = trainEmbeddings(ngramIter.getVocSize(), ngramIter, num_iters = 10)
