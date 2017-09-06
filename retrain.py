import tensorflow as tf
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from remtime import *
from collections import deque
from remtime import printTime
# from matplotlib import pyplot
# from mpl_toolkits.mplot3d import Axes3D

# top_param
LEARNING_RATE = 0.002
TS_BATCH_SIZE = 1000
N_EPOCHS = 2000
REG_PENALTY = 0.25
NUM_FEAT = 16

# LEARNING_RATE = 0.001
# BATCH_SIZE = 500	
# TS_BATCH_SIZE = 1000
# N_EPOCHS = 15
# REG_PENALTY = 0.05
# NUM_FEAT = 50


#w_movie = tf.Variable(tf.random_normal([NUM_MOVIES, NUM_FEAT])/np.sqrt(NUM_MOVIES))

def CollabFilterring(user_batch, movie_batch):
	# Access saved Variables directly
	# print(sess.run('cost:0'))
	# This will print 2, which is the value of bias that we saved


	# Now, let's access and create placeholders variables and
	# create feed-dict to feed new data
	weights_ = np.loadtxt("wmovie.csv",delimiter=',').astype(np.float32)
	biases_ = np.loadtxt("bmovie.csv",delimiter=',').astype(np.float32)
	w_user = tf.Variable(tf.random_normal([1, NUM_FEAT])/np.sqrt(1))
	w_movie = weights_

	batch_w_user = tf.nn.embedding_lookup(w_user, user_batch)
	batch_w_movie = tf.nn.embedding_lookup(w_movie, movie_batch)

	bias = tf.Variable(tf.zeros([1]))
	bias_user = tf.Variable(tf.zeros([1]))
	#bias_movie = tf.Variable(tf.zeros([NUM_MOVIES]))
	bias_movie = biases_
	batch_bias_user = tf.nn.embedding_lookup(bias_user, user_batch)
	batch_bias_movie = tf.nn.embedding_lookup(bias_movie, movie_batch)

	output = tf.reduce_sum(tf.multiply(batch_w_user, batch_w_movie), 1)
	output = tf.add(output, bias)
	output = tf.add(output, batch_bias_movie)
	output = tf.add(output, batch_bias_user, name='output')
	cost_reg = REG_PENALTY*tf.add(tf.nn.l2_loss(batch_w_movie), tf.nn.l2_loss(batch_w_user))
	# cost_l2 = tf.reduce_mean(tf.pow(output - rating_batch, 2))
	# cost_reg = 0
	return output, cost_reg

def train_nn(train_data, user_id):

	
	with tf.Session() as sess:
		user_batch = tf.placeholder(tf.int32, [None], name='user_batch')
		movie_batch = tf.placeholder(tf.int32, [None], name='movie_batch')
		rating_batch = tf.placeholder(tf.float32, [None], name='rating_batch')
		prediction, cost_reg = CollabFilterring(user_batch, movie_batch)

		cost_l2 = tf.nn.l2_loss(tf.subtract(prediction, rating_batch))
		cost = tf.add(cost_l2, cost_reg)
		#default learning rate = 0.001
		optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
		saver2 = tf.train.Saver()
		sess.run(tf.global_variables_initializer())
		for epoch in range(N_EPOCHS):
			stime = time.time()
			np.random.shuffle(train_data)	
			_, c, pred_batch = sess.run([optimizer, cost, prediction], feed_dict = {user_batch: train_data[:,2], movie_batch: train_data[:,0], rating_batch: train_data[:,1]})
			
			pred_batch = np.clip(pred_batch, 0, 5.0)
			print("E loss:"+str(round(np.sqrt(np.mean(np.power(pred_batch - train_data[:,1], 2))),2)))
		
		print train_data[:,1]
		print np.around(pred_batch*2)/2


		pred_batch = sess.run([prediction], feed_dict = {user_batch: test_data[:,1], movie_batch: test_data[:,0]})
		pred_batch = np.clip(pred_batch, 0, 5.0)
		print np.around(pred_batch*2)/2
		saver2.save(sess, str(user_id))


data = np.zeros(shape=(5,3))
user_Id = 0
ratings=np.array([2,4.0,4,5,5])
movieId=np.array([822, 895, 1129, 1176, 1942])
data[:,2] = user_Id
data[:,1] = ratings
data[:,0] = movieId.astype(int)
train_data = data
data = np.zeros(shape=(15,2))
user_Id = 0

movieId=np.array([10389, 11289, 470, 517, 8476, 6, 191, 312, 330, 481, 499, 524, 282, 577, 1331])
data[:,1] = user_Id
data[:,0] = movieId.astype(int)

test_data=data

print("Training starts here....")
train_nn(train_data, 0)
