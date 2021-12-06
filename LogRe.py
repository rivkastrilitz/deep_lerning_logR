from __future__ import print_function
import numpy as np # linear algebra
import pandas as pd #  CSV file
import matplotlib.pyplot as plt
import tensorflow as tf
import math


# load csv files
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_train.head(3)


full_data = [df_train, df_test]

'''
PREPROCESSING
'''
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
        y= np.expand_dims(y, axis=1)
        return x,y

def get_test_data():
        df = df_test
        clean(df, 0)
        x = df
        x = (x - x.mean())/(x.max()-x.min())
        x = x.values
        return x

    # DATA
X_train, y_train  = get_train_data()
X_val , y_val  = get_val_data()
X_test = get_test_data()

# CHECK SHAPES
print ('train_data', X_train.shape, 'train_label', y_train.shape)
print ('val_data', X_val.shape, 'val_label', y_val.shape)
print ('test_data', X_test.shape)

#number of Labels we are predicting is 2: survived or not_survived
numFeatures = X_train.shape[1]
numLabels = 2


# clear old variables
tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, numFeatures])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)
Lambda = 0.001 #Regularization Parameter
learningRate = tf.train.exponential_decay(learning_rate=1e-2,
                                          global_step= 1,
                                          decay_steps=X_train.shape[0],
                                          decay_rate= 0.95,
                                          staircase=True)


#LOGISTIC REGRESSION MODEL:
#This model uses Sigmoid as the activation function. L2 Regularization is used.
#fully_connected_layer = x*weights + bias
#activation_layer = sigmoid(fully_connected_layer)
#Loss = cross_entropy(activation_layer)+Lambda*L2_loss
#Using GradientDescent Optimizer

def model(x, y, is_training):
        weights = tf.get_variable("weights", shape=[numFeatures, numLabels])
        bias = tf.get_variable("bias", shape=[numLabels])
        y_out = tf.matmul(x, weights) + bias
        return (y_out, weights)


y_out, weights = model(x, y, is_training)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.one_hot(y, 2), logits=y_out))
regularizer = tf.nn.l2_loss(weights)
cost_op = tf.reduce_mean(loss + Lambda * regularizer)
optimizer = tf.train.GradientDescentOptimizer(learningRate)
train_step = optimizer.minimize(cost_op)

# Prediction
prediction = tf.argmax(y_out, 1)

# strat a session
sess = tf.Session()
sess.run(tf.global_variables_initializer())


def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])

    training_now = training is not None

    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [cost_op, correct_prediction, accuracy]
    if training_now:
        variables[-1] = training

    # counter
    iter_cnt = 0
    for e in np.arange(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in np.arange(int(math.ceil(Xd.shape[0] / batch_size))):
            # generate indicies for the batch
            start_idx = (i * batch_size) % Xd.shape[0]
            idx = train_indicies[start_idx:start_idx + batch_size]

            # create a feed dictionary for this batch
            feed_dict = {x: Xd[idx, :],
                         y: yd[idx],
                         is_training: training_now}
            # get batch size
            actual_batch_size = yd[idx].shape[0]
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables, feed_dict=feed_dict)

            # aggregate performance stats
            losses.append(loss * actual_batch_size)
            correct += np.sum(corr)

            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}" \
                      .format(iter_cnt, loss, np.sum(corr) / actual_batch_size))
            iter_cnt += 1
        total_correct = correct / Xd.shape[0]
        total_loss = np.sum(losses) / Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"
              .format(total_loss, total_correct, e + 1))
        if plot_losses and (e == epochs - 1):
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e + 1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss, total_correct

if __name__ == '__main__':

    print('Training')
    run_model(sess, y_out, cost_op, X_train, y_train, 100, 100, 100, train_step, True)
    print('Validation')
    run_model(sess, y_out, cost_op, X_val, y_val, 1, 100)



sess.close()