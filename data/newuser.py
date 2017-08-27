import tensorflow as tf
import pandas as pd
import numpy as np
import time
from remtime import printTime
from scipy.spatial.distance import cosine


movId = pd.read_csv('data/mov_hash.csv')['movId']


count = np.array(pd.read_csv('data/count.csv'))

NUM_MOVIE = len(movId)
userId = 0
data = np.array(([[111,	12827950],
						[735,	12827950]
						[36,	16297370]
						[515,	16297370]
						[608,	16297370]]))

print data