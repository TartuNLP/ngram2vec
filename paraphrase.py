#!/usr/bin/env python3

import nvecs
import sys

def paraphrase(ngramModel, query, n = 3):
	

if __name__ == "__main__":
	query = "pen and paper".split()
	
	mdl = nvecs.Ngram2Vec(sys.argv[1])
	
	print(query)
	
	for para in paraphrase(mdl, query, n = 5):
		print(para)
