import re
import math
import random

import numpy as np
import tensorflow as tf

from collections import defaultdict
from six.moves import xrange

maxNgramSize = 3

def read_data(filename):
    with open(filename, 'r', encoding='utf8') as fh:
        #return [line.split() for line in fh]
        for line in fh:
            yield line.split()

def iterNgrams(seqOfSeq, maxSize = 3, minSize = 1):
    assert minSize > 0
    
    for seqIdx, seq in enumerate(seqOfSeq):
        for i in range(minSize - 1, len(seq)):
            for ngramLen in range(minSize - 1, min(maxSize, i + 1)):
                ngram = seq[i-ngramLen : i+1]
                if not ngram:
                    msg = "Empty ngram in {0}".format(str(seq))
                    raise Exception()
                yield ngram, i - ngramLen, seqIdx

def decodeNgram(idxs, revDict):
    return "_".join([revDict[idx] for idx in idxs])

def build_dataset(snts):
    """Process raw inputs into a dataset."""

    dictionary = { 'UNK': 0 }
    data = list()
    
    for snt in snts:
        datasnt = list()
        
        for word in snt:
            if re.search(r'\w', word):
                if not word in dictionary:
                    dictionary[word] = len(dictionary)

                datasnt.append(dictionary[word])
        data.append(datasnt)
    print("reading done")
    ngram_dictionary = dict()

    for ngram, _, _ in iterNgrams(data, minSize=2, maxSize = maxNgramSize):
        nrep = str(ngram)
        if not nrep in ngram_dictionary:
            ngram_dictionary[nrep] = len(ngram_dictionary) + len(dictionary)

    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    reversed_ngram_dictionary = dict(zip(ngram_dictionary.values(), [eval(k) for k in ngram_dictionary.keys()]))
    reversed_dictionary.update({k: decodeNgram(v, reversed_dictionary) for k, v in reversed_ngram_dictionary.items()})
    
    return data, dictionary, reversed_dictionary, ngram_dictionary

def iterWindows(seqOfSeq, ngramDict, window_size=2, max_ngram=3):
    for ngram, nStart, seqIdx in iterNgrams(seqOfSeq, maxSize = max_ngram):
        seq = seqOfSeq[seqIdx]
        nLen = len(ngram)
        ngramRep = ngram[0] if nLen == 1 else ngramDict[str(ngram)]
        outputContext = seq[max(nStart-window_size, 0):nStart] + seq[nStart+nLen:nStart+nLen+window_size]
        for predictMe in outputContext:
            yield ngramRep, predictMe

def xgenerate_batch(iterator, batch_size, num_skips, skip_window, prevBuff = []):
    
    resultBuffer = list(prevBuff)
    
    try:
        while len(resultBuffer) < batch_size:
            resultBuffer.append(next(iterator))
    except StopIteration:
        pass
    
    if len(resultBuffer) < batch_size:
        return resultBuffer, None
    else:
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        for i in range(batch_size):
            batch[i], labels[i, 0] = resultBuffer[i]
    
        return (batch, labels)

if __name__ == "__main__":
	snts = read_data("europarl.en")
	datax, dictionary, reverse_dictionary, ngram_dict = build_dataset(snts)

	#del vocabulary  # Hint to reduce memory.
	print('Sample data', datax[0], [reverse_dictionary[i] for i in datax[0]])
	rdi = list(reverse_dictionary.items())
	print('Words and n-grams:', len(reverse_dictionary), rdi[:3], rdi[-5:-2])
	vocabulary_size = len(reverse_dictionary)
	print(vocabulary_size)

	batch_size = 128
	embedding_size = 256  # Dimension of the embedding vector.
	skip_window = 2       # How many words to consider left and right.
	num_skips = 2         # How many times to reuse an input to generate a label.

	# We pick a random validation set to sample nearest neighbors. Here we limit the
	# validation samples to the words that have a low numeric ID, which by
	# construction are also the most frequent.
	valid_size = 32     # Random set of words to evaluate similarity on.
	valid_window = 1000  # Only pick dev samples in the head of the distribution.
	valid_examples = np.random.choice(valid_window, valid_size, replace=False)
	num_sampled = 128    # Number of negative examples to sample.

	graph = tf.Graph()

	with graph.as_default():

		# Input data.
		train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
		train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
		valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

		# Ops and variables pinned to the CPU because of missing GPU implementation
		with tf.device('/cpu:0'):
			# Look up embeddings for inputs.
			embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
			embed = tf.nn.embedding_lookup(embeddings, train_inputs)

			# Construct the variables for the NCE loss
			nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
			nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

		# Compute the average NCE loss for the batch.
		# tf.nce_loss automatically draws a new sample of the negative labels each
		# time we evaluate the loss.
		loss = tf.reduce_mean(
			tf.nn.nce_loss(weights=nce_weights,
								biases=nce_biases,
								labels=train_labels,
								inputs=embed,
								num_sampled=num_sampled,
								num_classes=vocabulary_size))

		# Construct the SGD optimizer using a learning rate of 1.0.
		optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

		# Compute the cosine similarity between minibatch examples and all embeddings.
		norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
		normalized_embeddings = embeddings / norm
		valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
		similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

		# Add variable initializer.
		init = tf.global_variables_initializer()

	num_steps = 2500001

	itr = iterWindows(datax, ngram_dict, window_size = skip_window, max_ngram=maxNgramSize)

	with tf.Session(graph=graph) as session:
		# We must initialize all variables before we use them.
		init.run()
		print('Initialized')

		average_loss = 0
		for step in xrange(num_steps):
			batch_inputs, batch_labels = xgenerate_batch(itr, batch_size, num_skips, skip_window)
			
			while batch_labels == None:
				itr = iterWindows(datax, ngram_dict, window_size = skip_window, max_ngram=maxNgramSize)
				print("restarted iterator")
				batch_inputs, batch_labels = xgenerate_batch(itr, batch_size, num_skips, skip_window, prevBuff = batch_inputs)
		 
			feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

			# We perform one update step by evaluating the optimizer op (including it
			# in the list of returned values for session.run()
			_, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
			average_loss += loss_val

			if step % 10000 == 0:
				if step > 0:
					average_loss /= 10000
				# The average loss is an estimate of the loss over the last 2000 batches.
				print('Average loss at step ', step, ': ', average_loss)
				average_loss = 0

			# Note that this is expensive (~20% slowdown if computed every 500 steps)
			if step % 100000 == 0:
				sim = similarity.eval()
				for i in xrange(valid_size):
					valid_word = reverse_dictionary[valid_examples[i]]
					top_k = 8  # number of nearest neighbors
					nearest = (-sim[i, :]).argsort()[1:top_k + 1]
					log_str = 'Nearest to %s:' % valid_word
					for k in xrange(top_k):
						close_word = reverse_dictionary[nearest[k]]
						log_str = '%s %s,' % (log_str, close_word)
					print(log_str)
		final_embeddings = normalized_embeddings.eval()
		print("done")

	s = time.time()
	simDemand = np.dot(final_embeddings, final_embeddings[ngram_dict[str([dictionary['down'], dictionary['the'], dictionary['road']])]])
	e = time.time()
	print("time to multiply:", e - s)
	s = e
	simDemandWords = [(reverse_dictionary[i], s) for i, s in enumerate(simDemand)]
	e = time.time()
	print("time to label:", e - s)
	s = e
	simDemandWords.sort(key=lambda x: -x[1])
	e = time.time()
	print("time to sort:", e - s)
	s = e
	print([w for w, _ in simDemandWords[:10]])

	import json
	with open('3grams-emb256-w2.dat', 'w') as fh:
		 json.dump([dictionary, reverse_dictionary, ngram_dict, final_embeddings.tolist()], fh)
