from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)
pd.options.mode.chained_assignment = None
'''
PREPROCESSING
'''

df_train = pd.read_csv('dtrain.csv')
df_test = pd.read_csv('dtest.csv')


def clean(df, thresh):
	# DROP SHIT
	df.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)

	# CLEAN FURTHER
	lst = ['Mr.', 'Mrs.', 'Master.', 'Miss.']
	count = 0
	for i in range(len(df)):
		i += thresh

		# EXTRACT NAME
		l = df.Name[i].split(' ')
		for j in l:
			if j.strip() in lst:
				df.Name[i] = j.strip()[:-1]
				count += 1
				break
			df.Name[i] = ''

		# FILL NAME
		if df.Name[i] == '':
			if df.Sex[i] == 'male':
				if df.Age[i] >= 18:
					df.Name[i] = 'Mr'
				else:
					df.Name[i] = 'Master'
			elif df.Sex[i] == 'female':
				if df.Age[i] >= 18:
					df.Name[i] = 'Mrs'
				else:
					df.Name[i] = 'Miss'

		# FILL AGE
		if not (df.Age[i] > 0.001):
			if df.Name[i] == 'Mr' or df.Name[i] == 'Mrs':
				df.Age[i] = np.random.choice(range(20, 50))
			else:
				df.Age[i] = np.random.choice(range(5, 18))

	# MAKE NUMERIC
	df.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': -1, 'C': 0, 'Q': 1}, \
				'Name': {'Mr': -1, 'Mrs': 1, 'Master': -0.5, 'Miss': 0.5}}, inplace=True)

# DATASET FUNCTIONS
def get_train_data():
    df = df_train[:720]
    clean(df, 0)
    x = df.drop(['Survived'], axis=1)
    x = (x - x.mean())/(x.max()-x.min())
    y = df.loc[:,'Survived']
    x = x.values
    y = y.values
    y = np.expand_dims(y, axis=1)
    return x,y

def get_val_data():
    df = df_train[720:]
    clean(df, 720)
    x = df.drop(['Survived'], axis=1)
    x = (x - x.mean())/(x.max()-x.min())
    y = df.loc[:,'Survived']
    x = x.values
    y = y.values
    y = np.expand_dims(y, axis=1)
    return x,y

def get_test_data():
    df = df_test
    clean(df, 0)
    x = df
    x = (x - x.mean())/(x.max()-x.min())
    x = x.values
    return x

# DATA
train_data, train_labels = get_train_data()
val_data, val_labels = get_val_data()
test_data = get_test_data()

# CHECK SHAPES
print ('train_data', train_data.shape, 'train_label', train_labels.shape)
print ('val_data', val_data.shape, 'val_label', val_labels.shape)
print ('test_data', test_data.shape)

# TRAINING PARAMETERS
# epoch = one cycle through the full training dataset

num_epochs = 100000
display_epoch = 2000
num_samples = train_data.shape[0]
num_attrib = train_data.shape[1]
batch_size = 16
lr = 1e-4
num_hidden = 16


# SIGMOID ACTIVATION AND ITS DERIVATIVE
def activate(x):
	return 1 / (1 + np.exp(-x))

# TODO check what the F is it
def d_activation(x):
	return x * (1 - x)


# WEIGHTS AND BIASES
weight_i_h = np.random.random((num_attrib, num_hidden)) - 0.3
weight_h_o = np.random.random((num_hidden, 1)) - 0.1
bias_i_h = np.random.random((num_hidden)) + 0.2
bias_h_o = np.random.random((1)) + 0.3

# EMPTY LISTS TO STORE EPOCH, LOSS AND ACCURACY
ep, lo, ac = [], [], []

# TRAINING
for epoch in range(1, num_epochs + 1):

	for batch in range(int(num_samples / batch_size)):
		# FORWARD PROPAGATON
		input_layer = train_data[batch_size * batch: batch_size * (batch + 1)]
		hidden_layer = activate(np.dot(input_layer, weight_i_h) + bias_i_h)
		output_layer = activate(np.dot(hidden_layer, weight_h_o) + bias_h_o)
		output_train_labels = train_labels[batch_size * batch: batch_size * (batch + 1)]

		loss = output_layer - output_train_labels

		# BACKWARD PROPAGATION
		weight_i_h -= lr * input_layer.T.dot(
			((loss * d_activation(output_layer)).dot(weight_h_o.T)) * d_activation(hidden_layer))
		bias_i_h -= lr * sum(((loss * d_activation(output_layer)).dot(weight_h_o.T)) * d_activation(hidden_layer))
		weight_h_o -= lr * hidden_layer.T.dot(loss * d_activation(output_layer))
		bias_h_o -= lr * sum(loss * d_activation(output_layer))

	# ACCURACY
	x_, y_ = train_data[batch_size * batch: batch_size * (batch + 1)], train_labels[
																	   batch_size * batch: batch_size * (batch + 1)]
	prediction = activate((activate(x_.dot(weight_i_h) + bias_i_h)).dot(weight_h_o) + bias_h_o)
	prediction = (np.round(prediction, decimals=0)).astype(int)
	acc = np.mean(np.equal(y_, prediction))

	# STORE CHECKPOINTS
	if epoch == 1:
		print('Training in progress...')
		print('Epoch:', '%05d' % (epoch), ' Loss: {0:.10f}'.format((sum(abs(loss)))[0]),
			  ' Training Accuracy: {0:.5f}'.format(acc))
	if epoch % display_epoch == 0:
		print('Epoch:', '%05d' % (epoch), ' Loss: {0:.10f}'.format((sum(abs(loss)))[0]),
			  ' Training Accuracy: {0:.5f}'.format(acc))
		ep.append(epoch)
		lo.append((sum(abs(loss)))[0])
		ac.append(acc)

print('Model Trained !')




# # importing modules
# import os
# import numpy as np
# import pandas as pd
# import tensorflow as tf
#
# # # Path to Computation graphs
# # LOGDIR = './graphs'
#
# # todo insert dataset name
# train = pd.read_csv('dtrain.csv')
# test = pd.read_csv('dtest.csv')
#
# # print(data_train.head(892))
# df = pd.concat([train, test], axis=0, sort=True)
# # train.info("dtrain.csv")
# # print(train.shape)
# # print(train.size)
# # print(train.columns)
#
# sess = tf.Session()
#
# LEARNING_RATE = 0.01
# BATCH_SIZE = 1000
# EPOCHS = 10
#
# HL_1 = 1000
# # todo chek if we need to add x0
# INPUT_SIZE = 5 * 12
# N_CLASSES = 2
#
# with tf.name_scope('input'):
#     images = tf.placeholder(tf.float32, [None, INPUT_SIZE], name="feachers")
#     labels = tf.placeholder(tf.float32, [None, N_CLASSES], name="labels")
#
#
# def fc_layer(x, layer, size_out, activation=None):
#     with tf.name_scope(layer):
#         size_in = int(x.shape[1])
#         W = tf.Variable(tf.random_normal([size_in, size_out]), name="weights")
#         b = tf.Variable(tf.constant(-1, dtype=tf.float32, shape=[size_out]), name="biases")
#
#         wx_plus_b = tf.add(tf.matmul(x, W), b)
#         if activation:
#             return activation(wx_plus_b)
#         return wx_plus_b
#
#
# fc_1 = fc_layer(images, 'fc_1', HL_1, tf.nn.relu)
#
# # to prevent overfitting
# dropped = tf.nn.dropout(fc_1, keep_prob=0.9)
#
# # output layer
# y = fc_layer(dropped, 'output', N_CLASSES)
#
# with tf.name_scope('loss'):
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=labels))
#     tf.summary.scalar('loss', loss)
#
# with tf.name_scope('optimizer'):
#     train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
#
# with tf.name_scope('evaluation'):
#     correct = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))
#     tf.summary.scalar('accuracy', accuracy)
#
# # train_writer = tf.summary.FileWriter(os.path.join(LOGDIR, "train"), sess.graph)
# # test_writer = tf.summary.FileWriter(os.path.join(LOGDIR, "test"), sess.graph)
#
# summary_op = tf.summary.merge_all()
#
# init = tf.global_variables_initializer()
# sess.run(init)
#
# with tf.name_scope('training'):
#     step = 0
#     for epoch in range(EPOCHS):
#         print("epoch ", epoch, "\n-----------\n")
#
#         for batch in range(892 / BATCH_SIZE):
# 			step += 1
#
# 			# batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
#
# 			batch_xs = df[pd.notnull(df['Survived'])].drop(['Survived'], axis = 1)
# 			batch_ys = df[pd.notnull(df['Survived'])](['Survived'])
# 			X_test = df[pd.isnull(df['Survived'])].drop(['Survived'], axis=1)
#
#             summary_result, _ = sess.run([summary_op, train], feed_dict={images: batch_xs, labels: batch_ys})
#
#             # train_writer.add_summary(summary_result, step)
#
#             summary_result, acc = sess.run([summary_op, accuracy],
#                                            feed_dict={images: mnist.test.images, labels: mnist.test.labels})
#
#             # test_writer.add_summary(summary_result, step)
#
#             print("Batch ", batch, ": accuracy = ", acc)
#
# # train_writer.close()
# # test_writer.close()
# sess.close()
