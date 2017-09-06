import tensorflow as tf
import pandas as pd
import numpy as np
import time
from remtime import printTime

user_index = pd.read_csv('data/test_index.csv')['indexId']
userId = pd.read_csv('data/test.csv')['userId']
movId = pd.read_csv('data/mov_hash.csv')['movId']
train_index = pd.read_csv('data/train_index.csv')
score = np.array(pd.read_csv('data/score3.csv'))
genre = np.array(pd.read_csv('data/genres.csv'))
count = np.array(pd.read_csv('data/count.csv'))
year = np.array(pd.read_csv('data/releaseYr.csv'))
NUM_MOVIE = len(movId)
NUM_USER = len(userId)

print NUM_USER, NUM_MOVIE

print("Data preprocessing completed.")
print("Start here..")


sess=tf.Session()
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('cap-model.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
user_batch = graph.get_tensor_by_name("user_batch:0")
movie_batch = graph.get_tensor_by_name("movie_batch:0")

avg_rating = np.array(pd.read_csv('data/avg.csv')['avg_rating'])
user38 = np.zeros((5,2))
print NUM_MOVIE, len(avg_rating)
mov_avg = np.zeros((NUM_MOVIE,3))
mov_avg[:,0] = movId
mov_avg[:,1] = -avg_rating
mov_avg[:,1] = -year.reshape(NUM_MOVIE)
mov_avg = mov_avg[mov_avg[:,2].argsort()][0:50,0:2]
mov_avg = mov_avg[mov_avg[:,1].argsort()]
user38 = mov_avg[0:5]
user38[:,1] = -user38[:,1]
print user38

BATCH_SIZE = 2000
for i in range(NUM_USER):
	stime = time.time()
	movindex = np.array(train_index[train_index['user_index'] == user_index[i]]['mov_index'])
	# if i==38:
	# 	movindex=[]
	data = np.zeros((NUM_MOVIE-len(movindex),4))
	x=0
	for j in range(NUM_MOVIE):
		if j not in movindex:
			data[x,1]=j
			data[x,3]=year[j]
			x+=1
	# print(len(movindex))
	data[:,0] = user_index[i]

	data[:,3] = -data[:,3]
	data = data[data[:,3].argsort()]
	data[:,3] = -data[:,3]

	data = data[0:56]
		# print data[j*BATCH_SIZE:(j+1)*BATCH_SIZE,2]	
	feed_dict ={user_batch:data[:,0], movie_batch:data[:,1]}
	op_to_restore = graph.get_tensor_by_name("output:0")
	data[:,2] = sess.run(op_to_restore,feed_dict)
	
	data[:,3] = -data[:,3]
	data = data[data[:,3].argsort()]
	data[:,3] = -data[:,3]

	
	for j in range(len(data)):
		data[j,3]= count[int(data[j,1])]*np.power(data[j,2],3)
	data[:,3] = -data[:,3]
	data = data[data[:,3].argsort()]
	data[:,3] = -data[:,3]
	
	# for j in range(len(data)):
	# 	data[j,3]= np.sum(score[i]/np.sum(score[i])*genre[int(data[j,1])])
	# data[:,3] = -data[:,3]
	# data = data[data[:,3].argsort()]
	# data[:,3] = -data[:,3]
	top5 = data[0:5,0:3]


	top5[:,0] = userId[i]
	for j in range(0,5):
		top5[j,1]=movId[top5[j,1]]
	if i==38:
		top5[:,1:] = user38
	top5[:,2] = np.around(top5[:,2]*2)/2
	top5[:,2] = np.clip(top5[:,2], 3.5, 5.0)
	if i==0:
		recomm = top5
	else:
		recomm = np.vstack((recomm,top5))
	ftime = time.time()
	remtime = (ftime-stime)*(NUM_USER-i-1)
	
	printTime(remtime)

recomm = np.array(recomm)
recomm[:,0:2] = recomm[:,0:2].astype('int')
# recomm = pd.DataFrame(recomm,columns=['userId','movieId','rating'])
# cols = ['userId','movieId']
# recomm[cols] = recomm[cols].applymap(np.int64)
# recomm.to_csv('solution.csv',index=False)
print recomm[0:20]
np.savetxt('solution.csv', recomm,delimiter=",")
