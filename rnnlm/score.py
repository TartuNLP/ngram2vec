#!/usr/bin/env python3

import sys
import rnnlm
import pickle
from keras.models import load_model

if __name__ == "__main__":
	try:
		dataFile = sys.argv[1]
		dictInFile = sys.argv[2]
		modelInFile = sys.argv[3]
	except:
		print("Usage: score.py  data_file  dict_file  model_file")
	else:
		mdl = rnnlm.loadModels(modelInFile, dictInFile)
		
		textData = rnnlm.file2text(dataFile)
		
		for snt in textData:
			print(snt, rnnlm.score(snt, mdl))
