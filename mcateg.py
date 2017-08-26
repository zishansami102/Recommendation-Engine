import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import time
from remtime import printTime


user_index = pd.read_csv('data/test_index.csv')['indexId']
userId = pd.read_csv('data/test.csv')['userId']
movId = pd.read_csv('data/mov_hash.csv')['movId']
train_index = pd.read_csv('data/train_index.csv')

gen = np.array(pd.read_csv('data/genres.csv'))

NUM_MOVIE = len(movId)
NUM_USER = len(userId)


print("Data preprocessing completed.")
print("Start here..")



BATCH_SIZE = 1000
user_score = np.zeros((NUM_USER,20))
for i in range(NUM_USER):
	stime = time.time()
	movindex = np.array(train_index[train_index['user_index'] == user_index[i]]['mov_index'])
	for j in range(len(movindex)):
		user_score[i] += gen[movindex[j]]
	ftime = time.time()
	remtime = (ftime-stime)*(NUM_USER-i-1)
	printTime(remtime)



