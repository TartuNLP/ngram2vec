#!/usr/bin/env python3

import nvecs
import sys
import logging
import math
import gensim

from collections import defaultdict
from apply_bpe import BPE

logger = logging.getLogger('paraphrase.py')

class State:
	def __init__(self, expansions, ngram = list(), prev = None, covVec = set(), simProb = 0.0, lmProb = 0.0):
		self.lmProb = lmProb
		self.simProb = simProb
		self.covVec = covVec
		self.ngram = ngram
		self.prev = prev
		
		self.expansions = [exp for exp in expansions if self.compatible(exp)]
	
	def __repr__(self):
		return self.getKey() + ": " + str(self.getProb())

	def getProb(self):
		return self.simProb + self.lmProb

	def getKey(self):
		return str(self.covVec) + "//" + str(self.getFullForm())

	def nextStates(self):
		for inNgram, inNgramSpec, outNgramScoreList in self.expansions:
			for outNgram, simProb in outNgramScoreList:
				yield State(self.expansions, prev = self, ngram = outNgram, covVec = self.combineCovVec(inNgramSpec), simProb = self.simProb + math.log(simProb))

	def compatible(self, expansion):
		return not expansion[1] & self.covVec

	def combineCovVec(self, ngramSpec):
		return self.covVec | ngramSpec

	def isEnd(self, query):
		return self.covVec == set(range(len(query)))

	def getFullForm(self):
		result = []
		state = self
		while state.prev != None:
			result = state.ngram + result
			state = state.prev
		return result
	
	def getExplanation(self):
		result = []
		state = self
		
		while state.prev != None:
			result = [[state.ngram, state.covVec]] + result
			state = state.prev
		
		prevCov = set()
		
		for resultElem in result:
			x = resultElem[1]
			resultElem[1] -= prevCov
			prevCov |= x
		
		return ", ".join([str(a) + "/" + str(b) for a, b in result])

def ngrams(seq, simMdl, qn, maxNgramLen = 3):
	for i in range(len(seq)):
		#thisUniGram = [seq[i]]
		
		#yield thisUniGram, set(i), [ ( thisUniGram, 0.1 ) ]
		
		for l in range(maxNgramLen):
			if i - l >= 0:
				currSpec = range(i - l, i + 1)
				currNgram = [seq[i] for i in currSpec]
				
				if currNgram in simMdl:
					yield currNgram, set(currSpec), simMdl.most_similar(currNgram, k = qn)

def paraphrase(simMdl, query, n = 5, qn = 10):
	logger.debug("Paraphrasing " + str(query))
	grid = defaultdict(lambda : dict())

	scoredNgramPairs = list(ngrams(query, simMdl, qn))
	logger.debug("search space: " + "\n".join(["> " + str(x) for x in scoredNgramPairs]))
	
	startState = State(scoredNgramPairs)

	grid[0] = { startState.getKey(): startState }
	
	currLev = 0
	
	results = []

	while grid[currLev]:
		logger.debug("Processing level {0}".format(currLev))
		for state in sorted(grid[currLev].values(), key=lambda x: -x.getProb())[:n]:
			logger.debug("   Processing level {0} state {1}".format(currLev, state))
			
			if state.isEnd(query):
				expl = state.getExplanation()
				fullForm = state.getFullForm()
				prob = state.getProb()
				
				logger.debug("      End state: {0}, {1}".format(prob, expl))
				
				results.append((fullForm, prob, expl))
			else:
				deadEnd = True
				
				logger.debug("      Next states:")
				
				for nextState in state.nextStates():
					lev = currLev + len(nextState.ngram)
					key = nextState.getKey()
					
					logger.debug("      --> {0}".format(nextState))
					
					deadEnd = False
					
					if not key in grid[lev] or grid[lev][key].getProb() < nextState.getProb():
						grid[lev][key] = nextState
				
				if deadEnd:
					logger.debug("      Dead end")
					
			logger.debug("Finished processing level {0} state {1}".format(currLev, state))
			logger.debug("---------------------------------------")
		logger.debug("Finished processing level {0}".format(currLev))
		logger.debug("=======================================")
		
		currLev += 1
	
	return sorted(results, key=lambda x: -x[1])

def bpeSplit(query, bpeMdlFile):
	logger.info("Loading BPE model")
	with open(bpeMdlFile, 'r') as codes:
		bpeMdl = BPE(codes, separator = '')
	
	splitQuery = bpeMdl.segment(query).split()
	logger.debug("Query '{0}' split with BPE as {1}".format(query, str(splitQuery)))
	
	return splitQuery

def test(query = "houseofcards", simMdlFile = "opensubs-bpe.trivecs", bpeMdlFile = "opensubs-bpe.mdl"):
	logger.info("Loading similarity model")
	simMdl = nvecs.Ngram2Vec(simMdlFile)
	
	splitQuery = bpeSplit(query, bpeMdlFile)
	
	logger.info("Loading LM")
	lmMdl = gensim.models.Word2Vec.load('gensim-bpe-hs.mdl')
	
	logger.info("Paraphrasing")
	for phrase, prob, expl in paraphrase(simMdl, splitQuery, n = 20, qn = 20):
		#logger.info("{0} / {1}: {2} ({3})".format(prob, lmMdl.score(phrase), " ".join(phrase), expl))
		print("{0} / {1}: {2} ({3})".format(prob, lmMdl.score([phrase]), " ".join(phrase), expl))

if __name__ == "__main__":
	#logging.basicConfig(level = logging.INFO)
	test()
