import tensorflow as tf
import numpy as np
import argparse


# hardcode first
task_9_mapping = {
    None: 0,
    "mary": 1,
    "is": 2,
    "no": 3,
    "longer": 4,
    "in": 5,
    "the": 6,
    "bedroom": 7,
    "daniel": 8,
    "moved": 9,
    "to": 10,
    "hallway": 11,
    "sandra": 12,
    "bathroom": 13,
    "not": 14,
    "office": 15,
    "went": 16,
    "john": 17,
    "kitchen": 18,
    "travelled": 19,
    "back": 20,
    "garden": 21,
    "yes": 22,
    "journeyed": 23
}


def encode_word(mapping, word):
    if word.lower() in mapping:
        return mapping[word.lower()]
    else:
        return mapping[None]


def encode_sentence(sent_text, word_map, max_length):
    tokens = sent_text.split()

    sent_as_int = np.zeros(shape=(max_length, ), dtype=np.int32)
    for t_idx, token in enumerate(tokens):
        sent_as_int[t_idx] = encode_word(word_map, token)

    return sent_as_int


def encode_sentences(sent_texts, word_map, max_sent, max_length):
    sents_as_int = np.zeros(shape=(max_sent, max_length), dtype=np.int32)
    for s_idx, sent_text in enumerate(sent_texts):
        sents_as_int[s_idx] = encode_sentence(sent_text, word_map, max_length)

    return sents_as_int


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--babi_task_id", help="specify babi task 1-20 (default=1)")
parser.add_argument("-t", "--dmn_type", help="specify type of dmn (default=original)")
args = parser.parse_args()

dmn_type = args.dmn_type if args.dmn_type is not None else "plus"

if dmn_type == "plus":
    from dmn_plus import Config
    config = Config()
else:
    raise NotImplementedError(dmn_type + ' DMN type is not currently implemented')

if args.babi_task_id is not None:
    config.babi_id = args.babi_task_id

config.strong_supervision = False

config.train_mode = False

print('Testing DMN ' + dmn_type + ' on babi task', config.babi_id)

# create model
with tf.variable_scope('DMN') as scope:
    if dmn_type == "plus":
        from dmn_plus import DMNPlus
        model = DMNPlus(config)

print('==> initializing variables')
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as session:
    session.run(init)

    print('==> restoring weights')
    saver.restore(session, 'weights/task' + str(model.config.babi_id) + '.weights')

    print('==> running DMN')

    q_list = []
    i_list = []

    q_len_list = []
    i_len_list = []

    a_list = []

    rel_label_list = []

    # hard code encoded input for task 9
    # q_list.append(np.array([2, 17, 5, 6, 18], dtype=np.int32))
    # i_list.append(np.array([
    #                         [12, 2, 5, 6, 7, 0, 0],
    #                         [17, 2, 5, 6, 21, 0, 0],
    #                         [8,  2, 14, 5, 6, 15, 0],
    #                         [17, 16, 10, 6, 15, 0, 0],
    #                         [17, 2, 14, 5, 6, 15, 0],
    #                         [17, 16, 20, 10, 6, 18, 0],
    #                         [0, 0, 0, 0, 0, 0, 0],
    #                         [0, 0, 0, 0, 0, 0, 0],
    #                         [0, 0, 0, 0, 0, 0, 0],
    #                         [0, 0, 0, 0, 0, 0, 0]
    #                      ], dtype=np.int32))
    # q_len_list.append(5)
    # i_len_list.append(6)
    # a_list.append(22)
    # rel_label_list.append(np.array([0], dtype=np.int32))

    q_text = "Is John in the kitchen"
    input_texts = [
        "John went to the kitchen",
        "Sandra is not in the bedroom",
        "Mary is no longer in the bathroom",
    ]

    q = encode_sentence(q_text, task_9_mapping, model.max_q_len)
    input = encode_sentences(input_texts, task_9_mapping, model.max_input_len, model.max_sen_len)

    q_list.append(q)
    i_list.append(input)
    q_len_list.append(5)
    i_len_list.append(2)
    a_list.append(encode_word(task_9_mapping, "no"))
    rel_label_list.append(np.array([0], dtype=np.int32))

    qp = np.stack(q_list)
    ip = np.stack(i_list)
    ql = np.stack(q_len_list)
    il = np.stack(i_len_list)
    a = np.stack(a_list)
    r = np.stack(rel_label_list)

    feed = {
        model.question_placeholder: qp,
        model.input_placeholder: ip,
        model.question_len_placeholder: ql,
        model.input_len_placeholder: il,
        model.answer_placeholder: a,
        model.rel_label_placeholder: r,
        model.dropout_placeholder: 1
    }
    pred = session.run(
        [model.pred], feed_dict=feed)

    print('Question: ', q_text)
    print('Input: ')
    for text in input_texts:
        print(text if text is not None else '<blank line>')
    print('Prediction:', pred[0], ' => ', ('yes' if pred[0] == 22 else 'no' if pred[0] == 3 else pred))
