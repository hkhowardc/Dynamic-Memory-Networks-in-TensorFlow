import sys

import numpy as np

import tensorflow as tf
from attention_gru_cell import AttentionGRUCell


import babi_input


class Config(object):
    """Holds model hyperparams and data information."""

    batch_size = 100
    # embed_size = 80
    embed_size = 100
    # hidden_size = 80
    hidden_size = 256

    # max_epochs = 256
    max_epochs = 4096

    # early_stopping = 20
    early_stopping = 200

    dropout = 0.9
    lr = 0.001
    l2 = 0.001

    cap_grads = False
    max_grad_val = 10
    noisy_grads = False

    # word2vec_init = False
    word2vec_init = True
    embedding_init = np.sqrt(3) 

    # set to zero with strong supervision to only train gates
    strong_supervision = False
    beta = 1

    drop_grus = False

    anneal_threshold = 1000
    anneal_by = 1.5

    num_hops = 3
    num_attention_features = 4

    max_allowed_inputs = 130
    num_train = 9000

    floatX = np.float32

    babi_id = "1"
    babi_test_id = ""

    train_mode = True

    # sequence answer output
    seq_answer = False


def _add_gradient_noise(t, stddev=1e-3, name=None):
    """Adds gradient noise as described in http://arxiv.org/abs/1511.06807
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks."""
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)


# from https://github.com/domluna/memn2n
def _position_encoding(sentence_size, embedding_size):
    """Position encoding described in section 4.1 in "End to End Memory Networks" (http://arxiv.org/pdf/1503.08895v5.pdf)"""
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)


class DMNPlus(object):

    def load_data(self, debug=False):
        """Loads train/valid/test data and sentence encoding"""
        if self.config.train_mode:
            self.train, self.valid, self.word_embedding, self.max_q_len, self.max_input_len, self.max_sen_len, self.num_supporting_facts, self.vocab_size = babi_input.load_babi(self.config, split_sentences=True)
        else:
            self.test, self.word_embedding, self.max_q_len, self.max_input_len, self.max_sen_len, self.num_supporting_facts, self.vocab_size = babi_input.load_babi(self.config, split_sentences=True)
        self.encoding = _position_encoding(self.max_sen_len, self.config.embed_size)

        # TODO: remove hardcode later
        # TODO: Add <EOS> as RNN Output later
        if self.config.seq_answer:
            # Max num of words in answer sentences (no need to add <GO> signal)
            self.max_a_len = 3

    def add_placeholders(self):
        """add data placeholder to graph"""
        self.question_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_q_len))
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_input_len, self.max_sen_len))

        self.question_len_placeholder = tf.placeholder(tf.int32, shape=(None,))
        self.input_len_placeholder = tf.placeholder(tf.int32, shape=(None,))

        if self.config.seq_answer:
            # sequence answer output
            self.answer_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_a_len))
        else:
            self.answer_placeholder = tf.placeholder(tf.int64, shape=(None,))

        self.rel_label_placeholder = tf.placeholder(tf.int32, shape=(None, self.num_supporting_facts))

        self.dropout_placeholder = tf.placeholder(tf.float32)

        self.gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)

        # apply droput to grus if flag set
        if self.config.drop_grus:
            self.gru_cell = tf.contrib.rnn.DropoutWrapper(self.gru_cell,
                                                          input_keep_prob=self.dropout_placeholder,
                                                          output_keep_prob=self.dropout_placeholder)

    def get_predictions(self, output):
        preds = tf.nn.softmax(output)

        print('[get_predictions|get_predictions] preds.shape: %s' % preds.shape)
        print('[get_predictions|get_predictions] preds: %s' % preds)

        # for both single answer and sequence answer, the last dimension is the unit logits
        pred = tf.argmax(preds, -1)

        print('[DEBUG|get_predictions] pred<after argmax>.shape: %s' % pred.shape)
        print('[DEBUG|get_predictions] pred<after argmax>: %s' % pred)

        return pred

    def add_loss_op(self, output):
        """Calculate loss"""
        # optional strong supervision of attention with supporting facts
        gate_loss = 0
        if self.config.strong_supervision:
            for i, att in enumerate(self.attentions):
                labels = tf.gather(tf.transpose(self.rel_label_placeholder), 0)
                gate_loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=att, labels=labels))

        if self.config.seq_answer:
            # TODO Consider using legacy_seq2seq.sequence_loss() or other weighting techniques for free-text answers
            #   Ref: https://github.com/suriyadeepan/practical_seq2seq/blob/master/seq2seq_wrapper.py
            #        http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features

            # For loss of simple answers, no need to do weighting
            # Check examples on xentropy loss for RNN output, most of them flatten RNN output first
            output = tf.reshape(output, [-1, self.vocab_size])
            expected = tf.reshape(self.answer_placeholder, [-1])
            print('[DEBUG|add_loss_op] output<after reshape>.shape: %s' % output.shape)
            print('[DEBUG|add_loss_op] output<after reshape>: %s' % output)
            print('[DEBUG|add_loss_op] expected<after reshape>.shape: %s' % expected.shape)
            print('[DEBUG|add_loss_op] expected<after reshape>: %s' % expected)

            output_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=expected))

            loss = self.config.beta * output_loss + gate_loss
        else:
            loss = self.config.beta*tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=self.answer_placeholder)) + gate_loss

        # add l2 regularization for all variables except biases
        for v in tf.trainable_variables():
            if 'bias' not in v.name.lower():
                loss += self.config.l2*tf.nn.l2_loss(v)

        tf.summary.scalar('loss', loss)

        return loss
        
    def add_training_op(self, loss):
        """Calculate and apply gradients"""
        opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        gvs = opt.compute_gradients(loss)

        # optionally cap and noise gradients to regularize
        if self.config.cap_grads:
            gvs = [(tf.clip_by_norm(grad, self.config.max_grad_val), var) for grad, var in gvs]
        if self.config.noisy_grads:
            gvs = [(_add_gradient_noise(grad), var) for grad, var in gvs]

        train_op = opt.apply_gradients(gvs)
        return train_op

    def get_question_representation(self, embeddings):
        """Get question vectors via embedding and GRU"""

        # TODO hardcode CPU first
        with tf.device("/cpu:0"):
            questions = tf.nn.embedding_lookup(embeddings, self.question_placeholder)

        _, q_vec = tf.nn.dynamic_rnn(self.gru_cell,
                                     questions,
                                     dtype=np.float32,
                                     sequence_length=self.question_len_placeholder)

        return q_vec

    def get_input_representation(self, embeddings):
        """Get fact (sentence) vectors via embedding, positional encoding and bi-directional GRU"""
        # get word vectors from embedding
        # TODO hardcode CPU first
        with tf.device("/cpu:0"):
            inputs = tf.nn.embedding_lookup(embeddings, self.input_placeholder)

        # use encoding to get sentence representation
        inputs = tf.reduce_sum(inputs * self.encoding, 2)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.gru_cell,
                                                     self.gru_cell,
                                                     inputs,
                                                     dtype=np.float32,
                                                     sequence_length=self.input_len_placeholder)

        # f<-> = f-> + f<-
        fact_vecs = tf.reduce_sum(tf.stack(outputs), axis=0)

        # apply dropout
        fact_vecs = tf.nn.dropout(fact_vecs, self.dropout_placeholder)

        return fact_vecs

    def get_attention(self, q_vec, prev_memory, fact_vec, reuse):
        """Use question vector and previous memory to create scalar attention for current fact"""
        with tf.variable_scope("attention", reuse=True):

            features = [fact_vec * q_vec,
                        fact_vec * prev_memory,
                        tf.abs(fact_vec - q_vec),
                        tf.abs(fact_vec - prev_memory)]

            feature_vec = tf.concat(features, 1)

            attention = tf.layers.dense(feature_vec,
                                        self.config.embed_size,
                                        activation=tf.nn.tanh,
                                        reuse=reuse)

            attention = tf.layers.dense(attention,
                                        1,
                                        activation=None,
                                        reuse=reuse)
            
        return attention

    def generate_episode(self, memory, q_vec, fact_vecs, hop_index):
        """Generate episode by applying attention to current fact vectors through a modified GRU"""

        attentions = [tf.squeeze(
            self.get_attention(q_vec, memory, fv, bool(hop_index) or bool(i)), axis=1)
            for i, fv in enumerate(tf.unstack(fact_vecs, axis=1))]

        attentions = tf.transpose(tf.stack(attentions))
        self.attentions.append(attentions)
        attentions = tf.nn.softmax(attentions)
        attentions = tf.expand_dims(attentions, axis=-1)

        reuse = True if hop_index > 0 else False
        
        # concatenate fact vectors and attentions for input into attGRU
        gru_inputs = tf.concat([fact_vecs, attentions], 2)

        with tf.variable_scope('attention_gru', reuse=reuse):
            _, episode = tf.nn.dynamic_rnn(AttentionGRUCell(self.config.hidden_size),
                                           gru_inputs,
                                           dtype=np.float32,
                                           sequence_length=self.input_len_placeholder)

        return episode

    def add_answer_module(self, rnn_output, q_vec):
        """Linear softmax answer module"""

        rnn_output = tf.nn.dropout(rnn_output, self.dropout_placeholder)

        output = tf.layers.dense(tf.concat([rnn_output, q_vec], 1),
                                 self.vocab_size,
                                 activation=None)
        print('[DEBUG|add_answer_module] output.shape: %s' % output.shape)
        print('[DEBUG|add_answer_module] output: %s' % output)

        return output

    def add_seq_answer_module(self, rnn_output, q_vec):
        """Sequential answer module
        
        Answer Module should be implemented as follows, refers to https://arxiv.org/abs/1506.07285 section 2.4:
            y(t) = softmax(W(a)a(t))
            a(t) = GRU([y(t-1), q], q(t-1))
        
        DMN+ answer sequence output implementation refers to:
            https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/blob/master/dmn_batch.py

        Implementing RNN using tf.scan() refers to:
            http://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
                    
        For RNN Decoder <GO> signal as GRU's x(0), refers to:
            http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/
            https://arxiv.org/abs/1506.03099 section 4.3
            https://github.com/suriyadeepan/practical_seq2seq/blob/master/seq2seq_wrapper.py
            https://github.com/chiphuyen/tf-stanford-tutorials/blob/master/assignments/chatbot/model.py
            https://github.com/tensorflow/models/tree/master/tutorials/rnn/translate
        """

        # TODO consider add into seperate vocab size for answer
        # TODO add a seperate integer (not zero, which represents <PAD>/<EOS> for now) to represent <GO>
        # TODO also add a seperate integer to represent <EOS> and use zero to represents <PAD>
        # TODO try embeds decoder_inputs before passing to decoder RNN, refers to embedding_rnn_seq2seq in
        #   https://github.com/tensorflow/tensorflow/blob/master/
        #       tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py

        rnn_output = tf.nn.dropout(rnn_output, self.dropout_placeholder)

        # step function to execute GRU
        def gru_step_func(prev_step_output, current_elem):
            """
            Refers to http://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html

            :param prev_step_output: (y(t-1), a(t-1)) or the initializer (y(0), a(0))
                                     y(0) is <Go> signal, a(0) is the last memory (m(T(M)))                                     
            :param current_elem: (q_vec)
            :return: same structure as the initializer a(0)
            """

            # previous y, y(t-1): [batch_size x self.config.vocab_size]
            # previous a, a(t-1): [batch_size x self.config.hidden_size(a)]
            prev_y, prev_a = prev_step_output
            q = current_elem

            # [batch_size x self.config.vocab_size] + [batch_size x self.config.hidden_size]
            #   => [batch_size x self.config.vocab_size + self.config.hidden_size]
            gru_inputs = tf.concat([prev_y, q], 1)

            # RNNCell __call__(inputs, state):
            #   params:
            #       inputs: [batch_size x input_size], input_size = self.config.vocab_size + self.config.hidden_size(q)
            #       state:  [batch_size x self.state_size], self.state_size = self.config.hidden_size(a)
            #   return: (Outputs, New State)
            #       Outputs:    [batch_size x self.output_size], self.output_size = self.config.hidden_size(a)
            #       New State:  [batch_size x self.state_size] self.config.hidden_size(a)
            _, a = self.gru_cell(inputs=gru_inputs, state=prev_a)

            # Implements y(t) = softmax(W(a)*a(t))
            # Use dense layer to do matrix multiplication
            y = tf.layers.dense(a,
                                units=self.vocab_size,
                                activation=tf.nn.softmax,
                                use_bias=False)

            return y, a

        # For decoder inputs with GO signals (represented by zero for now)
        #   Shape: [batch_size x self.config.vocab_size]
        rnn_decoder_init_y_go = tf.zeros(shape=(tf.shape(self.answer_placeholder)[0], self.vocab_size))
        print('[DEBUG|add_seq_answer_module] rnn_decoder_init_y_go.shape: %s' % rnn_decoder_init_y_go.shape)
        print('[DEBUG|add_seq_answer_module] rnn_decoder_init_y_go: %s' % rnn_decoder_init_y_go)

        # Since every step needs to concate y(t-1) and q, let's pass q_vec as inputs of tf.scan() to determine
        #   Shape: [answer length x batch_size x self.config.hidden_size(q)]
        rnn_inputs = tf.stack([q_vec for _ in range(self.max_a_len)])
        print('[DEBUG|add_seq_answer_module] rnn_inputs.shape: %s' % rnn_inputs.shape)
        print('[DEBUG|add_seq_answer_module] rnn_inputs: %s' % rnn_inputs)

        # GRU's initial state is last memory a(0) = m(T(M))
        #   Shape: [batch_size x self.config.hidden_size(m)]
        last_memory = rnn_output
        print('[DEBUG|add_seq_answer_module] last_memory.shape: %s' % last_memory.shape)
        print('[DEBUG|add_seq_answer_module] last_memory: %s' % last_memory)

        # initial values of tf.scan()
        init_y_and_a = (rnn_decoder_init_y_go, last_memory)

        y_rnn_outputs, _ = tf.scan(fn=gru_step_func,
                                   elems=rnn_inputs,
                                   initializer=init_y_and_a,
                                   name='answer_decoder_rnn')
        print('[DEBUG|add_seq_answer_module] y_rnn_outputs.shape: %s' % y_rnn_outputs.shape)
        print('[DEBUG|add_seq_answer_module] y_rnn_outputs: %s' % y_rnn_outputs)

        # Shape: [answer length x batch_size x self.config.vocab_size] =>
        #           [batch_size x answer length x self.config.vocab_size]
        y_rnn_outputs = tf.transpose(y_rnn_outputs, [1, 0, 2])
        print('[DEBUG|add_seq_answer_module] y_rnn_outputs<after transpose>.shape: %s' % y_rnn_outputs.shape)
        print('[DEBUG|add_seq_answer_module] y_rnn_outputs<after transpose>: %s' % y_rnn_outputs)

        return y_rnn_outputs

    def inference(self):
        """Performs inference on the DMN model"""

        # set up embedding
        # TODO try lock the embedding weights if it is initialized with glove vectors (i.e. word2vec_init=True)
        embeddings = tf.Variable(self.word_embedding.astype(np.float32), name="Embedding")
         
        # input fusion module
        with tf.variable_scope("question", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get question representation')
            q_vec = self.get_question_representation(embeddings)

        with tf.variable_scope("input", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get input representation')
            fact_vecs = self.get_input_representation(embeddings)

        # keep track of attentions for possible strong supervision
        self.attentions = []

        # memory module
        with tf.variable_scope("memory", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> build episodic memory')

            # generate n_hops episodes
            prev_memory = q_vec

            for i in range(self.config.num_hops):
                # get a new episode
                print('==> generating episode', i)
                episode = self.generate_episode(prev_memory, q_vec, fact_vecs, i)

                # untied weights for memory update
                with tf.variable_scope("hop_%d" % i):
                    prev_memory = tf.layers.dense(tf.concat([prev_memory, episode, q_vec], 1),
                                                  self.config.hidden_size,
                                                  activation=tf.nn.relu)

            output = prev_memory

        # pass memory module output through linear answer module
        with tf.variable_scope("answer", initializer=tf.contrib.layers.xavier_initializer()):
            if self.config.seq_answer:
                # sequence answer output
                output = self.add_seq_answer_module(output, q_vec)
            else:
                output = self.add_answer_module(output, q_vec)

        return output

    # def count_unordered_matches(self, pred_sent, expected_sent):
    #
    #     expected_word_set = set(expected_sent.tolist())
    #
    #     match_count = 0
    #     for word_idx in pred_sent:
    #         if word_idx in expected_word_set:
    #             match_count += 1
    #
    #     print('[DEBUG] %s vs %s => %s' % (pred_sent, expected_sent, match_count))
    #
    #     return match_count

    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=2, train=False):
        config = self.config
        dp = config.dropout
        if train_op is None:
            train_op = tf.no_op()
            dp = 1
        total_steps = int(len(data[0]) / config.batch_size)
        total_loss = []
        accuracy = 0
        
        # shuffle data
        p = np.random.permutation(len(data[0]))
        qp, ip, ql, il, im, a, r = data
        qp, ip, ql, il, im, a, r = qp[p], ip[p], ql[p], il[p], im[p], a[p], r[p] 

        # Test Debugging
        # for t_i in range(3):
        #     print("Data[%s] qp: %s, ip: %s, ql: %s, il: %s, im: %s, a: %s, r: %s" % (t_i, qp[t_i], ip[t_i], ql[t_i], il[t_i], im[t_i], a[t_i], r[t_i]))

        for step in range(total_steps):
            index = list(range(step * config.batch_size, (step + 1) * config.batch_size))
            feed = {
                self.question_placeholder: qp[index],
                self.input_placeholder: ip[index],
                self.question_len_placeholder: ql[index],
                self.input_len_placeholder: il[index],
                self.answer_placeholder: a[index],
                self.rel_label_placeholder: r[index],
                self.dropout_placeholder: dp
            }
            loss, pred, summary, _ = session.run(
              [self.calculate_loss, self.pred, self.merged, train_op], feed_dict=feed)

            if train_writer is not None:
                train_writer.add_summary(summary, num_epoch * total_steps + step)

            answers = a[step * config.batch_size: (step + 1) * config.batch_size]

            if self.config.seq_answer:
                # For simplicity, the accuracy is defined as exact, full sequence match
                #
                # TODO for accuracy of more variable free-text consider weighting
                #   Ref: http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features

                # unordered_matches = [self.count_unordered_matches(pred_st, answers_st)
                #                      for pred_st, answers_st in zip(pred, answers)]
                # accuracy_unordered = np.sum(unordered_matches) / float(len(answers) * self.max_a_len)
                # print('accuracy_unordered (total %s answers): %s' % (len(answers), accuracy_unordered))

                # for pred_st, answers_st in zip(pred, answers):
                #     print('%s vs %s => %s' % (pred_st, answers_st, np.array_equiv(pred_st, answers_st)))

                matches = [np.array_equiv(pred_st, answers_st) for pred_st, answers_st in zip(pred, answers)]
                accuracy += np.sum(matches) / float(len(answers))
            else:
                accuracy += np.sum(pred == answers) / float(len(answers))

            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                  step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')

        return np.mean(total_loss), accuracy/float(total_steps)

    def __init__(self, config):
        # TODO Hardcode GPU first
        with tf.device("/gpu:0"):
            self.config = config
            self.variables_to_save = {}
            self.load_data(debug=False)
            self.add_placeholders()
            self.output = self.inference()
            self.pred = self.get_predictions(self.output)
            self.calculate_loss = self.add_loss_op(self.output)
            self.train_step = self.add_training_op(self.calculate_loss)
            self.merged = tf.summary.merge_all()
