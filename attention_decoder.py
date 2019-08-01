from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn
from tensorflow.python.ops import nn
from tensorflow.python.ops import init_ops

from tensorflow.contrib.layers import conv2d, fully_connected
from functools import partial
from tensorflow.contrib.legacy_seq2seq import attention_decoder

def mask(sequence_lengths):
	""" Mask function used by recurrent decoder with attention.

	Given a vector with varying lengths, produces an explicit matrices with
	  True/False values. E.g.
	  > mask([1,2,3])
	  [[True, False, False],
	   [True, True, False],
	   [True, True, True]]
	Args:
	  sequence_lengths: An int32/int64 vector of size n.
	Return:
	  true_false: A [n, max_len(sequence_lengths)] sized Tensor.
	"""
	# based on this SO answer: http://stackoverflow.com/a/34138336/118173
	batch_size = array_ops.shape(sequence_lengths)[0]
	max_len = math_ops.reduce_max(sequence_lengths)

	lengths_transposed = array_ops.expand_dims(sequence_lengths, 1)

	rng = math_ops.range(max_len)
	rng_row = array_ops.expand_dims(rng, 0)
	true_false = math_ops.less(rng_row, lengths_transposed)
	return true_false


def rnn_decoder_attention(cell, num_attention_units,
                          attention_inputs, decoder_inputs, initial_state,
                          decoder_length, decoder_fn, attention_length=None,
                          weight_initializer=None, encoder_projection=None,
                          parallel_iterations=None, swap_memory=False,
                          time_major=False, scope=None):
	""" Dynamic RNN decoder with attention for a sequence-to-sequence model
	specified by RNNCell 'cell'.

	The 'rnn_decoder_attention' is similar to the
	'tf.python.ops.rnn.dynamic_rnn'. As the decoder does not make any
	assumptions of sequence length of the input or how many steps it can decode,
	since 'dynamic_rnn_decoder' uses dynamic unrolling. This allows
	'attention_inputs' and 'decoder_inputs' to have [None] in the sequence
	length of the decoder inputs.

	The parameters attention_inputs and  decoder_inputs are nessesary for both
	training and evaluation. During training all of attention_inputs and a slice
	of decoder_inputs is feed at every timestep. During evaluation
	decoder_inputs it is only feed at time==0, as the decoder needs the
	'start-of-sequence' symbol, known from Bahdanau et al., 2014
	https://arxiv.org/abs/1409.0473, at the beginning of decoding.

	The parameter  initial_state is used to initialize the decoder RNN.
	As default a linear transformation with a tf.nn.tanh linearity is used.
	By a linear transformation we can have different number of units between
	the encoder and decoder.

	The parameter sequence length is nessesary as it determines how many
	timesteps to decode for each sample. TODO: Could make it optional for
	training.

	The parameter attention_length is used for masking the alpha values
	computes over the attention_input. Is set to None (default) no mask is
	computed.

	Extensions of interest:
	- Support time_major=True for attention_input (not using conv2D)
	- Look into rnn.raw_rnn so we don't need to handle zero states
	- Make 'alpha' usable
	- Don't use decoder_inputs for evaluation
	- Make a attention class to allow custom attention functions
	- Multi-layered decoder
	- Beam search

	Args:
	  cell: An instance of RNNCell.
	  num_attention_units: The number of units used for attention.
	  attention_inputs: The encoded inputs.
		The input used to attend over at every timestep, must be of size
		[batch_size, seq_len, features]
	  decoder_inputs: The inputs for decoding (embedded format).
		If `time_major == False` (default), this must be a `Tensor` of shape:
		  `[batch_size, max_time, ...]`.
		If `time_major == True`, this must be a `Tensor` of shape:
		  `[max_time, batch_size, ...]`.
		The input to `cell` at each time step will be a `Tensor` with dimensions
		  `[batch_size, ...]`.
	  initial_state: An initial state for the decoder's RNN.
		Must be [batch_size, num_features], where num_features does not have to
		match the cell.state_size. As a projection is performed at the beginning
		of the decoding.
	  decoder_length: An int32/int64 vector sized `[batch_size]`.
	  decoder_fn: A function that takes a state and returns an embedding.
		Here is an example of a `decoder_fn`:
		def decoder_fn(embeddings, weight, bias):
		  def dec_fn(state):
			prev = tf.matmul(state, weight) + bias
			return tf.gather(embeddings, tf.argmax(prev, 1))
		  return dec_fn
	  encoder_projection: (optional) given that the encoder might have a
		different size than the decoder, we project the intial state as
		described in Bahdanau, 2014 (https://arxiv.org/abs/1409.0473).
		The optional `encoder_projection` is a
		`tf.contrib.layers.fully_connected` with
		`activation_fn=tf.python.ops.nn.tanh`.
	  weight_initializer: (optional) An initializer used for attention.
	  attention_length: (optional) An int32/int64 vector sized `[batch_size]`.
	  parallel_iterations: (Default: 32).  The number of iterations to run in
		parallel.  Those operations which do not have any temporal dependency
		and can be run in parallel, will be.  This parameter trades off
		time for space.  Values >> 1 use more memory but take less time,
		while smaller values use less memory but computations take longer.
	  swap_memory: Transparently swap the tensors produced in forward inference
		but needed for back prop from GPU to CPU.  This allows training RNNs
		which would typically not fit on a single GPU, with very minimal (or no)
		performance penalty.
	  time_major: The shape format of the `inputs` and `outputs` Tensors.
		If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
		If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
		Using `time_major = True` is a bit more efficient because it avoids
		transposes at the beginning and end of the RNN calculation.  However,
		most TensorFlow data is batch-major, so by default this function
		accepts input and emits output in batch-major form.
	  scope: VariableScope for the created subgraph;
		defaults to "decoder_attention".
	Returns:
	  A pair (outputs_train, outputs_eval) where:
		outputs_train/eval: the RNN output 'Tensor'
		  If time_major == False (default), this will be a `Tensor` shaped:
			`[batch_size, max_time, cell.output_size]`.
		  If time_major == True, this will be a `Tensor` shaped:
			`[max_time, batch_size, cell.output_size]`.
		NOTICE: output_train is commonly used for calculating loss.
	Raises:
	  #TODO Put up some raises
	"""

	with vs.variable_scope(scope or "decoder") as varscope:
		# Project initial_state as described in Bahdanau et al. 2014
		# https://arxiv.org/abs/1409.0473
		if encoder_projection is None:
			encoder_projection = partial(fully_connected, activation_fn=math_ops.tanh)
		state = encoder_projection(initial_state, cell.output_size)
		# Setup of RNN (dimensions, sizes, length, initial state, dtype)
		# Setup dtype
		dtype = state.dtype
		if not time_major:
			# [batch, seq, features] -> [seq, batch, features]
			decoder_inputs = array_ops.transpose(decoder_inputs, perm=[1, 0, 2])
		# Get data input information
		batch_size = array_ops.shape(decoder_inputs)[1]
		attention_input_depth = int(attention_inputs.get_shape()[2])
		decoder_input_depth = int(decoder_inputs.get_shape()[2])
		attention_max_length = array_ops.shape(attention_inputs)[1]
		# Setup decoder inputs as TensorArray
		decoder_inputs_ta = tensor_array_ops.TensorArray(dtype, size=0,
		                                                 dynamic_size=True)
		decoder_inputs_ta = decoder_inputs_ta.unpack(decoder_inputs)

		print "attention_input_depth,", attention_input_depth
		print "decoder_input_depth,", decoder_input_depth
		# Setup attention weight
		if weight_initializer is None:
			weight_initializer = init_ops.truncated_normal_initializer(stddev=0.1)
		with vs.variable_scope("attention") as attnscope:
			v_a = vs.get_variable('v_a',
			                      shape=[num_attention_units],
			                      initializer=weight_initializer)
			W_a = vs.get_variable('W_a',
			                      shape=[cell.output_size, num_attention_units],
			                      initializer=weight_initializer)

		# Encode attention_inputs for attention
		hidden = array_ops.reshape(attention_inputs, [-1, attention_max_length,
		                                              1, attention_input_depth])
		part1 = conv2d(hidden, num_attention_units, (1, 1))
		part1 = array_ops.squeeze(part1, [2])  # Squeeze out the third dimension

		def context_fn(state, inp):
			with vs.variable_scope("attention") as attnscope:
				part2 = math_ops.matmul(state, W_a)  # [batch, attn_units]
				part2 = array_ops.expand_dims(part2, 1)  # [batch, 1, attn_units]
				cmb_attn = part1 + part2  # [batch, seq, attn_units]
				e = math_ops.reduce_sum(v_a * math_ops.tanh(cmb_attn), [2])  # [batch, seq]
				alpha = nn.softmax(e)
				# Mask
				if attention_length is not None:
					alpha = math_ops.to_float(mask(attention_length)) * alpha
					alpha = alpha / math_ops.reduce_sum(alpha, [1], keep_dims=True)
				# [batch, features]
				context = math_ops.reduce_sum(array_ops.expand_dims(alpha, 2)
				                              * attention_inputs, [1])
				context.set_shape([None, attention_input_depth])
				con = array_ops.concat(1, (inp, context))
				print "con,", con.get_shape()
				return con, alpha

		# loop function train
		def loop_fn_train(time, cell_output, cell_state, loop_state):
			print "@@@TRAIN@@@"
			emit_output = cell_output
			if cell_output is None:
				next_cell_state = state  # Use projection of prev encoder state
			else:
				next_cell_state = cell_state
			elements_finished = (time >= decoder_length)  # TODO handle seq_len=None
			finished = math_ops.reduce_all(elements_finished)

			next_input, _ = control_flow_ops.cond(
				finished,
				# Handle zero states
				lambda: (array_ops.zeros([batch_size,
				                          decoder_input_depth + attention_input_depth], dtype=dtype),
				         array_ops.zeros([batch_size, attention_max_length],
				                         dtype=dtype)),
				# Read data and calculate attention
				lambda: context_fn(next_cell_state, decoder_inputs_ta.read(time)))
			next_input.set_shape(
				[None, decoder_input_depth + attention_input_depth])  # it loses its shape at some point
			next_loop_state = None
			return (elements_finished, next_input, next_cell_state,
			        emit_output, next_loop_state)

		# loop function eval
		def loop_fn_eval(time, cell_output, cell_state, loop_state):
			print "@@@EVAL@@@"
			emit_output = cell_output
			if cell_output is None:
				next_cell_state = state
			else:
				next_cell_state = cell_state
			elements_finished = (time >= decoder_length)  # TODO handle seq_len=None
			finished = math_ops.reduce_all(elements_finished)
			varscope.reuse_variables()
			next_input, _ = control_flow_ops.cond(
				finished,
				# Handle zero states
				lambda: (array_ops.zeros([batch_size,
				                          decoder_input_depth + attention_input_depth], dtype=dtype),
				         array_ops.zeros([batch_size, attention_max_length],
				                         dtype=dtype)),
				# Read data and calculate attention
				lambda: control_flow_ops.cond(math_ops.greater(time, 0),
				                              lambda: context_fn(next_cell_state,
				                                                 decoder_fn(next_cell_state)),
				                              lambda: context_fn(next_cell_state,
				                                                 decoder_inputs_ta.read(0))))
			# next_input loses its shape at some point
			next_input.set_shape([None, decoder_input_depth + attention_input_depth])
			next_loop_state = None
			print "next_input,", next_input.get_shape()
			return (elements_finished, next_input, next_cell_state,
			        emit_output, next_loop_state)

		# Run raw_rnn function
		outputs_ta_train, _, _ = rnn.raw_rnn(cell, loop_fn_train)
		varscope.reuse_variables()
		outputs_ta_eval, _, _ = rnn.raw_rnn(cell, loop_fn_eval)
		outputs_train = outputs_ta_train.pack()
		outputs_eval = outputs_ta_eval.pack()
		if not time_major:
			outputs_train = array_ops.transpose(outputs_train, perm=[1, 0, 2])
			outputs_eval = array_ops.transpose(outputs_eval, perm=[1, 0, 2])
		return outputs_train, outputs_eval
