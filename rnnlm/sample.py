#!/usr/bin/env python3

import sys
import rnnlm
import pickle
from keras.models import load_model

if __name__ == "__main__":
	modelInFile = sys.argv[1]
	dictInFile = sys.argv[2]
	try:
		numToSample = int(sys.argv[3])
	except IndexError:
		numToSample = 1
	
	mdls = rnnlm.loadModels(modelInFile, dictInFile)
	
	for _ in range(numToSample):
		raw, prob = rnnlm.sample(mdls)
		
		decoded = [str(mdls[1]['i2w'][i]) for i in raw]
		print(" ".join(decoded) + " (" + str(prob) + ")")
