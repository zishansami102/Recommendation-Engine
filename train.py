import tensorflow as tf
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from remtime import *
from collections import deque
from remtime import printTime



LEARNING_RATE = 0.0005
BATCH_SIZE = 500
TS_BATCH_SIZE = 1000
N_EPOCHS = 24
REG_PENALTY = 0.05
NUM_FEAT = 100

NUM_USERS = 6040
NUM_MOVIES = 3952


user_batch = tf.placeholder(tf.int32, [None], name='user_batch')
movie_batch = tf.placeholder(tf.int32, [None], name='movie_batch')
rating_batch = tf.placeholder(tf.float32, [None], name='rating_batch')

##############################################################################################################################
################################################## ------ START HERE --------  ###############################################
##############################################################################################################################


def CollabFilterring(user_batch, movie_batch):

	w_user = tf.Variable(tf.random_normal([NUM_USERS, NUM_FEAT])/np.sqrt(NUM_USERS))
	w_movie = tf.Variable(tf.random_normal([NUM_MOVIES, NUM_FEAT])/np.sqrt(NUM_MOVIES))
	batch_w_user = tf.nn.embedding_lookup(w_user, user_batch)
	batch_w_movie = tf.nn.embedding_lookup(w_movie, movie_batch)

	bias = tf.Variable(tf.zeros([]))
	bias_user = tf.Variable(tf.zeros([NUM_USERS]))
	bias_movie = tf.Variable(tf.zeros([NUM_MOVIES]))
	batch_bias_user = tf.nn.embedding_lookup(bias_user, user_batch)
	batch_bias_movie = tf.nn.embedding_lookup(bias_movie, movie_batch)

	output = tf.reduce_sum(tf.multiply(batch_w_user, batch_w_movie), 1)
	output = tf.add(output, bias)
	output = tf.add(output, batch_bias_movie)
	output = tf.add(output, batch_bias_user, name='output')

	cost_reg = REG_PENALTY*tf.add(tf.nn.l2_loss(batch_w_movie), tf.nn.l2_loss(batch_w_user))
	
	return output, cost_reg

def train_nn(user_batch, movie_batch, rating_batch):
	num_batch_loop = int(NUM_TR_ROW/BATCH_SIZE)

	prediction, cost_reg = CollabFilterring(user_batch, movie_batch)
	cost_l2 = tf.nn.l2_loss(tf.subtract(prediction, rating_batch))
	
	
	# cost_l2 = tf.reduce_mean(tf.pow(output - rating_batch, 2))
	# cost_reg = 0

	cost = tf.add(cost_l2, cost_reg)

	#default learning rate = 0.001
	optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		RMSEtr = []
		RMSEts = []
		for epoch in range(N_EPOCHS):
			stime = time.time()
			num_batch_loop = int(NUM_TR_ROW/BATCH_SIZE)
			np.random.shuffle(train_data)	
			errors = deque(maxlen=num_batch_loop)

			for i in range(num_batch_loop):
				_, c, pred_batch = sess.run([optimizer, cost, prediction], feed_dict = {user_batch: train_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE,0], movie_batch: train_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE,1], rating_batch: train_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE,2]})
				pred_batch = np.clip(pred_batch, 1.0, 5.0)
				errors.append(np.mean(np.power(pred_batch - train_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE,2], 2)))

			TR_epoch_loss = np.sqrt(np.mean(errors))
			RMSEtr.append(TR_epoch_loss)

			num_batch_loop = int(NUM_TS_ROW/TS_BATCH_SIZE)
			errors = deque(maxlen=num_batch_loop)

			for i in range(num_batch_loop):
				pred_batch = prediction.eval({user_batch: test_data[i*TS_BATCH_SIZE:(i+1)*TS_BATCH_SIZE,0], movie_batch: test_data[i*TS_BATCH_SIZE:(i+1)*TS_BATCH_SIZE,1], rating_batch: test_data[i*TS_BATCH_SIZE:(i+1)*TS_BATCH_SIZE,2]})
				pred_batch = np.clip(pred_batch, 1.0, 5.0)
				errors.append(np.mean(np.power(pred_batch - test_data[i*TS_BATCH_SIZE:(i+1)*TS_BATCH_SIZE,2], 2)))

			TS_epoch_loss = np.sqrt(np.mean(errors))
			RMSEts.append(TS_epoch_loss)
			ftime = time.time()
			remtime = (N_EPOCHS-epoch-1)*(ftime-stime)
			print("Epoch"+ str(epoch+1)+" completed out of "+str(N_EPOCHS)+"; Train loss:"+str(round(TR_epoch_loss,3))+"; Test loss:"+str(round(TS_epoch_loss,3)))
			printTime(remtime)

		print("Computing Final Test Loss...")

		bloss = 0
		for xx in range(num_batch_loop):
			pred_batch = prediction.eval({user_batch: test_data[xx*TS_BATCH_SIZE:(xx+1)*TS_BATCH_SIZE,0], movie_batch: test_data[xx*TS_BATCH_SIZE:(xx+1)*TS_BATCH_SIZE,1]})
			pred_batch = np.clip(pred_batch, 1.0, 5.0)
			bloss += np.mean(np.power(pred_batch - test_data[xx*TS_BATCH_SIZE:(xx+1)*TS_BATCH_SIZE,2], 2))
			if (xx+1)%50==0:
				per = float(xx+1)/(num_batch_loop)*100
				print(str(per)+"% Completed")
		test_loss = np.sqrt(bloss/num_batch_loop)
		print("Test Loss:"+str(round(test_loss,3)))
		
		RMSEtr[0]=RMSEts[0]	#this was done to ensure the scale matching in the plot (RMSEtr[0] starts from around 2.16 and would ruin the plot)
		plt.plot(RMSEtr, label='Training Set', color='b')
		plt.plot(RMSEts, label='Test Set', color='r')
		plt.legend()
		plt.ylabel('-----  RMSE  ---->')
		plt.xlabel('-----  Epoch  ---->')
		plt.title('RMSE vs Epoch (Biased Matrix Factorization)')
		plt.show()
		saver.save(sess, 'gen-model')
		print("Awesome !!")






col_names = ["user", "movie", "ratings", "timestamp"]
df = pd.read_csv('ml-1m/ratings.dat', sep='::', names=col_names, header=None,  engine='python')

df.drop('timestamp',axis=1, inplace=True)
data = np.array(df)

data[:,0:2] -= 1
data = np.asfarray(data)	

NUM_ROW = data.shape[0]
np.random.shuffle(data)
split = int(0.9*NUM_ROW)
train_data = data[:split]
test_data = data[split:-21]

NUM_TR_ROW = train_data.shape[0]
NUM_TS_ROW = test_data.shape[0]
print("Data preprocessing completed.")



print("Training starts here....")
train_nn(user_batch, movie_batch, rating_batch)
