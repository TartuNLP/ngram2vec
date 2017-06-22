import sys
import numpy as np
import math
import pickle

from collections import defaultdict, Counter

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
from keras.utils import to_categorical

from keras.models import load_model

SOS = 1
EOS = 2
OOV = 3

def file2text(filename, maxLen = 50):
	if filename == '-':
		fh = sys.stdin
	else:
		fh = open(filename, 'r')
	
	result = []
	
	for line in fh:
		toks = line.strip().lower().split()
		
		if toks and len(toks) < maxLen:
			result.append(toks)
	
	if filename != '-':
		fh.close()
	
	return result

def text2dicts(textData, vocSize = 10000):
	idx2word = { 0: None, SOS: "__s__", EOS: "__/s__", OOV: "UNK" }
	word2idx = dict(zip(idx2word.values(), idx2word.keys()))
	
	freq = defaultdict(int)

	for toks in textData:
		for tok in toks:
			freq[tok] += 1
	
	freq = { k: v for k, v in sorted(freq.items(), key=lambda x: -x[1])[:(vocSize - len(idx2word))] }
	
	for tok in sorted(freq, key=lambda x: -freq[x]):
		idx = len(idx2word)
		word2idx[tok] = idx
		idx2word[idx] = tok
	
	return word2idx, idx2word
	
def text2numio(textData, word2idx, maxLen):
	numSnts = len(textData)
	inputs = np.zeros([numSnts, maxLen], dtype='int32')
	outputs = np.zeros([numSnts, maxLen, 1], dtype='int32')
	
	for i, toks in enumerate(textData):
		inputs[i,0] = SOS
		
		for j, tok in enumerate(toks):
			try:
				idx = word2idx[tok]
			except KeyError:
				idx = OOV
			
			inputs[i, j+1] = idx
			outputs[i, j, 0] = idx
		
		outputs[i, len(toks), 0] = EOS
	
	return inputs, outputs

def initModel(vocSize, maxLen):
	model = Sequential()
	
	model.add(Embedding(input_dim = vocSize, output_dim = 128, input_length = maxLen))
	
	model.add(LSTM(256, input_shape=(maxLen, 128), return_sequences=True))
	model.add(Dropout(0.2))
	
	model.add(Dense(vocSize))
	model.add(Activation('softmax'))
	
	model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
	
	return model

def learn(mdl, inputs, outputs):
	mdl.fit(inputs, outputs, epochs=3, batch_size=32)

def renorm(pd):
	raw = [p**2 for p in pd]
	raw[OOV] = 0
	s = sum(raw)
	return [p/s for p in raw]

def sample(mdls):
	(mdl, dicts) = mdls
	
	baseInput = np.zeros([1,50], dtype='int32')
	
	result = []
	w = SOS
	
	prob = 0.0
	
	for i in range(50):
		baseInput[0, i] = w
		
		pd = mdl.predict(baseInput)[0, i]
		
		#w = max(enumerate(pd), key=lambda x: x[1] if x[0] != OOV else 0)[0]
		w = np.random.choice(dicts['v'], p=renorm(pd))
		prob += math.log(pd[w])
		
		if w == EOS:
			break
		
		result.append(w)
	
	return result, prob/(len(result)+1)

def score(snt, models, skipEOS = False):
	(mdl, dicts) = models
	
	inputs, outputs = text2numio([snt], dicts['w2i'], dicts['m'])
	
	hyps = mdl.predict(inputs)
	
	result = 0
	length = 0
	
	for j, pVec in enumerate(hyps[0]):
		inp = inputs[0, j]
		outp = outputs[0, j, 0]
		
		if inp == 0 or (skipEOS and outp == EOS):
			break
		
		length += 1
		result += math.log(pVec[outp])
		
	#return result / length
	return result / 10

def loadModels(modelFile, dictFile):
	mdl = load_model(modelFile)
	
	with open(dictFile, 'rb') as fh:
		dicts = pickle.load(fh)
	
	return (mdl, dicts)
