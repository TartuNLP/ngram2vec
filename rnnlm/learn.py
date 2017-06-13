#!/usr/bin/env python3

import sys
import rnnlm
import pickle
from keras.models import load_model

if __name__ == "__main__":
	if len(sys.argv) == 4:
		#learn from scratch
		dataFile = sys.argv[1]
		dictOutFile = sys.argv[2]
		modelOutFile = sys.argv[3]
		
		vocSize = 10000
		maxLen = 50
		
		textData = rnnlm.file2text(dataFile, maxLen = maxLen)
		w2i, i2w = rnnlm.text2dicts(textData, vocSize)
		inputs, outputs = rnnlm.text2numio(textData, w2i, maxLen)
		
		with open(sys.argv[3], 'wb') as fh:
			pickle.dump({ 'w2i': w2i, 'i2w': i2w, 'v': vocSize, 'm': maxLen }, fh, protocol=pickle.HIGHEST_PROTOCOL)
		
		lm = rnnlm.initModel(vocSize, maxLen)
		
	elif len(sys.argv) == 5:
		#continue learning
		dataFile = sys.argv[1]
		dictInFile = sys.argv[2]
		modelInFile = sys.argv[3]
		modelOutFile = sys.argv[4]
		
		with open(dictInFile, 'rb') as fh:
			dicts = pickle.load(fh)
		
		textData = rnnlm.file2text(dataFile, maxLen = dicts['m'])
		inputs, outputs = rnnlm.text2numio(textData, dicts['w2i'], dicts['m'])
		
		lm = load_model(modelInFile)
	else:
		raise Exception("AAAAA")
	
	rnnlm.learn(lm, inputs, outputs)
	lm.save(modelOutFile)
