#!/usr/bin/env python3

import sys
import logging
import math
import rnnlm
from gensim.models import KeyedVectors

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

	def nextStates(self, lmMdl):
		for inNgram, inNgramSpec, outNgramScoreList in self.expansions:
			for outNgram, simProb in outNgramScoreList:
				newstate = State(self.expansions, prev = self, ngram = outNgram, covVec = self.combineCovVec(inNgramSpec), simProb = self.simProb + math.log(simProb))
				newstate.lmProb = rnnlm.score(newstate.getFullForm(), lmMdl)
				#print(newstate.ngram, newstate.getFullForm(), newstate.lmProb)
				yield newstate

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

def ngrams(seq, simMdl, qn, maxNgramLen = 4):
	for i in range(len(seq)):
		#thisUniGram = [seq[i]]
		
		#yield thisUniGram, set(i), [ ( thisUniGram, 0.1 ) ]
		
		for l in range(maxNgramLen):
			if i - l >= 0:
				currSpec = range(i - l, i + 1)
				currNgram = [seq[i] for i in currSpec]
				
				currNgramStr = "__".join(currNgram)
				
				if currNgramStr in simMdl:
					yield currNgram, set(currSpec), [(ngram.split("__"), prob) for ngram, prob in simMdl.most_similar(currNgramStr, topn = qn)]

def paraphrase(query, simMdl, lmMdl, n = 5, qn = 10):
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
				
				for nextState in state.nextStates(lmMdl):
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

def loadBpe(bpeMdlFile):
	with open(bpeMdlFile, 'r') as codes:
		bpeMdl = BPE(codes, separator = '')
	return bpeMdl

def bpeSplit(query, bpeMdl):
	result = []
	for segm in query:
		result.append(bpeMdl.segment(segm))
	
	return (" ".join(result)).split()

if __name__ == "__main__":
	logging.basicConfig(level = logging.INFO)
	
	try:
		simMdlFile = sys.argv[1]
		bpeMdlFile = sys.argv[2]
		lmMdlFile = sys.argv[3]
		dictFile = sys.argv[4]
	except IndexError:
		print("Usage: paraphrase.py  ngramMdl  bpeMdl  languageMdl  dictMdl")
	else:
		logger.info("Loading similarity model")
		simMdl = KeyedVectors.load_word2vec_format(simMdlFile, binary=True)
		
		a = simMdl.most_similar(list(simMdl.vocab)[5])
		logger.debug(str(a))
		
		logger.info("Loading LM")
		lmMdl = rnnlm.loadModels(lmMdlFile, dictFile)
		
		logger.info("Loading BPE model")
		bpeMdl = loadBpe(bpeMdlFile)
		
		logger.info("Ready to paraphrase (enter 'quit' to quit)")
		
		query = "-"
		
		while query != ["quit"]:
			sys.stdout.write("\nQuery: ")
			
			query = input().lower().split()
			
			splitQuery = bpeSplit(query, bpeMdl)
			
			print("(split as " + "|".join(splitQuery) + ")")
			results = paraphrase(splitQuery, simMdl, lmMdl, n = 5, qn = 10)
			for phrase, prob, expl in results[:5]:
				print("{2} (p={1} / {3})".format("".join(phrase), prob, "|".join(phrase), expl))
