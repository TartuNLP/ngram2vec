#!/usr/bin/env python3

import sys
import spacy

nlp = spacy.load('en')

with open(sys.argv[1], 'r') as fh:
	for line in fh:
		doc = nlp.tokenizer(line)
		nlp.tagger(doc)
		
		
