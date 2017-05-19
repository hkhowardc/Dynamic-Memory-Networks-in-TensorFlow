import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear


class AnswerGRUWrapper(RNNCell):

    def __init__(self, gru, q_vec, vocab_size):
        """Create a Answer GRU cell which concat y(t-1) with q_vec
        Args:
          gru: an GRUCell
          q_vec: a 2-D [batch size x q_vec_dim] tensor 
          vocab_size: integer, the size of the output after projection.
        Raises:
          TypeError: if cell is not an RNNCell.
          ValueError: if vocab_size is not positive.
        """

        if not isinstance(gru, tf.contrib.rnn.GRUCell):
            raise TypeError("The parameter cell is not GRUCell.")

        if vocab_size < 1:
            raise ValueError("Parameter vocab_size must be > 0: %d." % vocab_size)

        self._cell = gru
        self._q_vec = q_vec
        self._vocab_size = vocab_size

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._vocab_size

    def __call__(self, inputs, state, scope=None):
        """
        Concate the inputs with self._q_vec and pass as inputs into the enclosing GRUCell, 
        then project to self._vocab_size
        """

        # Implement a(t) = GRU([y(t-1), q], q(t-1))
        #   [batch_size x vocab_size] + [batch_size x hidden_size(q)]
        #       => [batch_size x vocab_size + hidden_size(q)]
        combined_inputs = tf.concat([inputs, self._q_vec], 1)

        # Implements a(t) = GRU([y(t - 1), q], q(t - 1))
        # invoke the nested RNNCell (GRUCell)
        #   RNNCell __call__(inputs, state):
        #       params:
        #           inputs: [batch_size x vocab_size + hidden_size(q)]
        #           state:  [batch_size x hidden_size(a)]
        #       return: (Outputs, New State)
        #           Outputs:    [batch_size x hidden_size(a)]
        #           New State:  [batch_size x hidden_size(a)]
        output, res_state = self._cell(combined_inputs, state)

        # Implements y(t) = softmax(W(a)*a(t))
        #   Shape: [batch_size x hidden_size(a)]
        #               => [batch_size x vocab_size]
        projected = _linear(output,
                            output_size=self._vocab_size,
                            bias=False)

        return projected, res_state
