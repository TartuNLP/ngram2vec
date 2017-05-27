#!/usr/bin/env python3

import nvecs
import sys
from collections import defaultdict

class State:
	def __init__(self, ngram = list(), prev = None, covVec = set(), simProb = 0.0, lmProb = 0.0):
		self.lmProb = lmProb
		self.simProb = simProb
		self.covVec = covVec
		self.ngram = ngram
		self.prev = prev
	
	def __repr__(self):
		return self.getKey() + ": " + str(self.getProb())

	def getProb(self):
		return self.simProb + self.lmProb

	def getKey(self):
		return str(self.covVec) + "//" + str(self.getFullForm())

	def nextStates(self, ngramMdl, query):
		for inNgram, inNgramSpec in self.compatibleNgrams(query, ngramMdl):
			for outNgram, simProb in ngramMdl["".join(inNgram)]:
				yield State(prev = self, ngram = outNgram, covVec = self.combineCovVec(inNgramSpec), simProb = self.simProb + simProb)

	def compatibleNgrams(self, query, ngramMdl):
		maxNgramLen = 3

		for i in range(len(query)):
			for l in range(maxNgramLen):
				if i - l >= 0:
					currIdx = set(range(i - l, i + 1))
					currNgram = [query[i] for i in currIdx]

					if not currIdx & self.covVec and "".join(currNgram) in ngramMdl:
						yield currNgram, currIdx

	def combineCovVec(self, ngramSpec):
		return self.covVec | ngramSpec

	def isEnd(self, query):
		return self.covVec == set(range(len(query)))

	def getFullForm(self):
		result = []
		state = self
		while state.prev != None:
			#print("D", state.ngram, result)
			result = state.ngram + result
			state = state.prev
		return result

def paraphrase(ngramModel, query, n = 5, qn = 20):
	grid = defaultdict(lambda : dict())

	startState = State()

	grid[0] = { startState.getKey(): startState }
	
	currLev = 0
	
	results = []

	while grid[currLev]:
		for state in sorted(grid[currLev].values(), key=lambda x: -x.getProb())[:n]:
			if state.isEnd(query):
				#yield state.getFullForm(), state.getProb()
				results.append(state.getFullForm(), state.getProb())
			else:
				for nextState in state.nextStates(ngramModel, query):
					lev = currLev + len(nextState.ngram)
					key = nextState.getKey()
					
					if not key in grid[lev] or grid[lev][key].getProb() < nextState.getProb():
						grid[lev][key] = nextState

		currLev += 1
	
	return sorted(results, key=lambda x: -x[1])

if __name__ == "__main__":
	query = [x for x in "ABC"]

	mdl = { "A": [ (["a"], -1), (["x", "y"], -2) ], "BC": [ (["b", "c"], -1) ], "AB": [ (["a", "b"], -3) ], "C": [ ( ["c"], -0.5 ), ( ["z"], -0.1 ) ] }

	for x paraphrase(mdl, query, n = 15):
		 print(v, p)
