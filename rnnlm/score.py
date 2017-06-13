#!/usr/bin/env python3

import sys
import rnnlm
import pickle
from keras.models import load_model

if __name__ == "__main__":
	dataFile = sys.argv[1]
	dictInFile = sys.argv[2]
	modelInFile = sys.argv[3]
	
	with open(dictInFile, 'rb') as fh:
		dicts = pickle.load(fh)
	
	lm = load_model(modelInFile)
	
	textData = rnnlm.file2text(dataFile, maxLen = dicts['m'])
	
	for snt in textData:
		print(rnnlm.score(lm, snt, dicts))
