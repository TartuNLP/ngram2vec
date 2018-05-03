import sys

import ngram

if __name__ == "__main__":
	dataFile = sys.argv[1]
	outFile = sys.argv[2]
	beta = 0.125
	freqFilter = [20,120,90]

	lines = ngram.SentenceNgramSampler(dataFile, minCounts=freqFilter, ngramThresholdBeta=beta)
  
	with open(outFile, "w") as f:
		for i in lines:
			f.write(" ".join(i) + "\n")
	print("Job finished")
