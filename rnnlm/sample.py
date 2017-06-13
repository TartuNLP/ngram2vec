#!/usr/bin/env python3

import sys
import rnnlm
import pickle
from keras.models import load_model

if __name__ == "__main__":
	modelInFile = sys.argv[1]
	dictInFile = sys.argv[2]
	
	with open(dictInFile, 'rb') as fh:
		dicts = pickle.load(fh)
	
	lm = load_model(modelInFile)
	
	raw, prob = rnnlm.sample(lm, dicts)
	
	decoded = [str(dicts['i2w'][i]) for i in raw]
	print(" ".join(decoded) + " (" + str(prob) + ")")
