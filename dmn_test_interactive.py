import tensorflow as tf
import numpy as np
import argparse

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
    q_list.append(np.array([2, 17, 5, 6, 18], dtype=np.int32))
    i_list.append(np.array([
                            [12, 2, 5, 6, 7, 0, 0],
                            [17, 2, 5, 6, 21, 0, 0],
                            [8,  2, 14, 5, 6, 15, 0],
                            [17, 16, 10, 6, 15, 0, 0],
                            [17, 2, 14, 5, 6, 15, 0],
                            [17, 16, 20, 10, 6, 18, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]
                         ], dtype=np.int32))
    q_len_list.append(5)
    i_len_list.append(6)
    a_list.append(22)
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
    loss, pred, summary = session.run(
        [model.calculate_loss, model.pred, model.merged], feed_dict=feed)

    print('')
    print('Prediction:', pred)
