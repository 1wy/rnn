
# coding: utf-8
# in this example, we add attention mechanism to the seq2seq model.

x = [[5, 7, 8], [6, 3], [3], [1]]
import helpers
xt, xlen = helpers.batch(x)

import numpy as np
import tensorflow as tf

tf.reset_default_graph()
sess = tf.InteractiveSession()

PAD = 0
EOS = 1

vocab_size = 10
input_embedding_size = 20

encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units

# define inputs
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
encoder_inputs_length = tf.placeholder(shape=(None), dtype=tf.int32, name='encoder_inputs_length')

# initialize the embedding
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

# encoder
encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

encoder_outputs, encoder_final_state = tf.nn.bidirectional_dynamic_rnn(
	encoder_cell, encoder_cell, encoder_inputs_embedded,
	sequence_length=encoder_inputs_length,
	dtype=tf.float32, time_major=True,
)


# concat the bidirectional hidden units
(state_c_fw, state_h_fw), (state_c_bw, state_h_bw) = encoder_final_state

state_c = tf.concat([state_c_fw, state_c_bw], 1)
state_h = tf.concat([state_h_fw, state_h_bw], 1)
# project the final encoder vector
state_c = tf.contrib.layers.fully_connected(state_c, decoder_hidden_units, activation_fn=tf.nn.tanh)
state_h = tf.contrib.layers.fully_connected(state_h, decoder_hidden_units, activation_fn=tf.nn.tanh)

encoder_final_state = tf.contrib.rnn.LSTMStateTuple(state_c, state_h)
encoder_outputs = tf.concat(encoder_outputs, 2)

# decoder
decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)
encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))

# here we unroll the decoder for len(encoder_input)+2.
# +2 additional steps, +1 leading <EOS> token for decoder inputs
decoder_lengths = encoder_inputs_length + 3

eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)

Wo = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32, name='Wo')
bo = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32, name='bo')
# Loop initial state is function of only encoder_final_state and embeddings:
def loop_fn_initial():
	initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
	initial_input = eos_step_embedded
	initial_cell_state = encoder_final_state
	initial_cell_output = None
	initial_loop_state = None  # we don't need to pass any additional information
	return (initial_elements_finished,
			initial_input,
			initial_cell_state,
			initial_cell_output,
			initial_loop_state)

def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
	def get_next_input():
		output_logits = tf.add(tf.matmul(previous_output, Wo), bo)
		prediction = tf.argmax(output_logits, axis=1)
		next_input = tf.nn.embedding_lookup(embeddings, prediction)
		return next_input

	elements_finished = (time >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
	# defining if corresponding sequence has ended

	finished = tf.reduce_all(elements_finished)  # -> boolean scalar
	input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
	state = previous_state
	output = previous_output
	loop_state = None

	return (elements_finished,
			input,
			state,
			output,
			loop_state)

def loop_fn(time, previous_output, previous_state, previous_loop_state):
	if previous_state is None:    # time == 0
		assert previous_output is None and previous_state is None
		return loop_fn_initial()
	else:
		return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.stack()


decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, Wo), bo)
decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))

decoder_prediction = tf.argmax(decoder_logits, 2)


stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
	labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
	logits=decoder_logits,
)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())


batch_size = 100

batches = helpers.random_sequences(length_from=3, length_to=8,
								   vocab_lower=2, vocab_upper=10,
								   batch_size=batch_size)

print('head of the batch:')
for seq in next(batches)[:10]:
	print(seq)

def next_feed():
	batch = next(batches)
	encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
	decoder_targets_, _ = helpers.batch(
		[(sequence) + [EOS] + [PAD] * 2 for sequence in batch]
	)
	return {
		encoder_inputs: encoder_inputs_,
		encoder_inputs_length: encoder_input_lengths_,
		decoder_targets: decoder_targets_,
	}

loss_track = []

max_batches = 3001
batches_in_epoch = 1000

try:
	for batch in range(max_batches):
		fd = next_feed()
		_, l = sess.run([train_op, loss], fd)
		loss_track.append(l)

		if batch == 0 or batch % batches_in_epoch == 0:
			print('batch {}'.format(batch))
			print('  minibatch loss: {}'.format(sess.run(loss, fd)))
			predict_ = sess.run(decoder_prediction, fd)
			for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
				print('  sample {}:'.format(i + 1))
				print('    input     > {}'.format(inp))
				print('    predicted > {}'.format(pred))
				if i >= 2:
					break
			print()

except KeyboardInterrupt:
	print('training interrupted')

import matplotlib.pyplot as plt
plt.plot(loss_track)
plt.show()
print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))