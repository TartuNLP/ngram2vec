import re
import sys
import logging

import ngram

from datetime import datetime

def debug(msg):
	sys.stderr.write("{0}: {1}\n".format(str(datetime.now()), msg))

if __name__ == "__main__":
	dataFile = sys.argv[1]
	outFile = sys.argv[2]
	logging.basicConfig(level = logging.INFO)
	beta = 0.125
	freqFilter = [20,120,90]

	lines = ngram.SentenceNgramSampler(dataFile, minCounts=freqFilter, ngramThresholdBeta=beta)
  
	with open(outFile, "w") as f:
		for i in lines:
			f.write(" ".join(i) + "\n")
