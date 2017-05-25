import numpy as np
import tensorflow as tf
import math

from six.moves import xrange

def trainEmbeddings(vocabulary_size, miniBatchIterator, embedding_size = 256, learning_rate = 1.0, num_neg_sampled = 64):
	batch_size = miniBatchIterator.miniBatchSize
	#embedding_size = 256  # Dimension of the embedding vector.
	#skip_window = 2       # How many words to consider left and right.
	#num_skips = 2         # How many times to reuse an input to generate a label.

	# We pick a random validation set to sample nearest neighbors. Here we limit the
	# validation samples to the words that have a low numeric ID, which by
	# construction are also the most frequent.
	valid_size = 40     # Random set of words to evaluate similarity on.
	valid_window = 500  # Only pick dev samples in the head of the distribution.
	#valid_examples = np.random.choice(valid_window, valid_size, replace=False)
	
	vstart = len(miniBatchIterator.word2idx)
	
	valid_examples = np.array(np.random.choice(valid_window, 20, replace=False).tolist() + \
		(vstart + np.random.choice(valid_window, 25, replace=False)).tolist())
	#valid_examples = 200 + np.random.choice(valid_window, valid_size, replace=False)
	
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
								num_sampled=num_neg_sampled,
								num_classes=vocabulary_size))

		# Construct the SGD optimizer using a learning rate of 1.0.
		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

		# Compute the cosine similarity between minibatch examples and all embeddings.
		norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
		normalized_embeddings = embeddings / norm
		valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
		similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

		# Add variable initializer.
		init = tf.global_variables_initializer()

	with tf.Session(graph=graph) as session:
		# We must initialize all variables before we use them.
		init.run()
		print('Initialized')

		average_loss = 0
		step = 0
		#for step in xrange(num_iters):
		for batch_inputs, batch_labels in miniBatchIterator:
			step += 1
		 
			feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

			# We perform one update step by evaluating the optimizer op (including it
			# in the list of returned values for session.run()
			_, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
			average_loss += loss_val

			if step % 2000 == 0:
				if step > 0:
					average_loss /= 2000
				# The average loss is an estimate of the loss over the last 2000 batches.
				print('Average loss at step ', step, ': ', average_loss)
				average_loss = 0

			# Note that this is expensive (~20% slowdown if computed every 500 steps)
			if step % 100000 == 0:
				sim = similarity.eval()
				for i in xrange(valid_size):
					#valid_word = miniBatchIterator.idx2word[valid_examples[i]]
					
					#valid_word = str(miniBatchIterator.nidx2list[valid_examples[i]]) + "/"
					#print(valid_word)
					
					valid_word = "_".join([miniBatchIterator.idx2word[wIdx] for wIdx in miniBatchIterator.nidx2list[valid_examples[i]]])
					
					top_k = 8  # number of nearest neighbors
					nearest = (-sim[i, :]).argsort()[1:top_k + 1]
					log_str = 'Nearest to %s:' % valid_word
					for k in xrange(top_k):
						#close_word = miniBatchIterator.idx2word[nearest[k]]
						close_word = "_".join([miniBatchIterator.idx2word[wIdx] for wIdx in miniBatchIterator.nidx2list[nearest[k]]])
						log_str = '%s %s (%s),' % (log_str, close_word, sim[i, nearest[k]])
					print(log_str)
		final_embeddings = normalized_embeddings.eval()
		return final_embeddings
