import tensorflow as tf
import pandas as pd
import numpy as np

userId = 0
rated = np.array([111, 735, 36,515, 608])


sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('cap-model.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
user_batch = graph.get_tensor_by_name("user_batch:0")
movie_batch = graph.get_tensor_by_name("movie_batch:0")
rating_batch = graph.get_tensor_by_name("rating_batch:0")

movId = pd.read_csv('data/mov_hash.csv')['movId']
# count = np.array(pd.read_csv('data/count.csv'))
NUM_MOVIE = len(movId)

data = np.zeros((NUM_MOVIE-len(rated),3))
x=0
for j in range(NUM_MOVIE):
	if j not in rated:
		data[x,1]=j
		x+=1
data[:,0] = userId
BATCH_SIZE = 2000

num_loop = int(NUM_MOVIE/BATCH_SIZE)
for j in range(num_loop):
	feed_dict ={user_batch:data[j*BATCH_SIZE:(j+1)*BATCH_SIZE,0], movie_batch:data[j*BATCH_SIZE:(j+1)*BATCH_SIZE,1]}
	op_to_restore = graph.get_tensor_by_name("output:0")
	data[j*BATCH_SIZE:(j+1)*BATCH_SIZE,2] = sess.run(op_to_restore,feed_dict)
	# print data[j*BATCH_SIZE:(j+1)*BATCH_SIZE,2]	
feed_dict ={user_batch:data[(j+1)*BATCH_SIZE:,0], movie_batch:data[(j+1)*BATCH_SIZE:,1]}
op_to_restore = graph.get_tensor_by_name("output:0")
data[(j+1)*BATCH_SIZE:,2] = sess.run(op_to_restore,feed_dict)

for j in range(len(data)):
	data[j,1]=movId[int(data[j,1])]
data[:,2] = -data[:,2]
data = data[data[:,2].argsort()]
data[:,2] = -data[:,2]
data[:,2] = np.around(data[:,2]*2)/2
df = pd.DataFrame(data[:,1:],columns=['movieId', 'rating'])
df.to_csv(str(userId)+'.csv',index=False)

