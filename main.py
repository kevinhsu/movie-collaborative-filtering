# -*- coding: UTF-8 -*-

from __future__ import division
import pandas as pd
from pandas import Series, DataFrame
import os
import numpy as np
from pprint import pprint
from time import time
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

start = time()

# Load rating data
current_path = os.getcwd()
ratings_path = current_path + '\\movie_lens\\data\\ratings.dat'
rnames = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_table(ratings_path, sep='::', header=None, names=rnames)

# Create pivot table
data = ratings.pivot(index='user_id',columns='movie_id',values='rating')

# Find the users paris with the most overlap ratings
# DataFrame is an object and it is not stored in memory
# Sacrifice I/O for saving memory
foo = DataFrame(np.empty((len(data.index),len(data.index)),dtype=int),index=data.index,columns=data.index)

"""
Non-parallelized Global Search
"""
# # The following loop can be parallelized
# ser_max = 0
# ser_id1 = 0
# ser_id2 = 0
# for i in foo.index:
#     for j in foo.columns:
#         if i > j:
#             # Count the number of the overlap movies of different users
#             # Use .ix[] to set rows
#             # Use [] to set columns
#             # Use .ix[,] to set rows and columns both
#             # foo is a symmetry matrix, so we only calculate half of it
#             if ser_max < data.ix[i][data.ix[j].notnull()].dropna().count():
#                 ser_max = data.ix[i][data.ix[j].notnull()].dropna().count()
#                 ser_id1 = i
#                 ser_id2 = j
#     print i

"""
Parallelized Global Search
"""
# Define a map function
def map_analysis(i):
    print i
    ser_max = 0
    ser_id1 = 0
    ser_id2 = 0
    for j in foo.columns:
        if i > j:
            if ser_max < data.ix[i][data.ix[j].notnull()].dropna().count():
                ser_max = data.ix[i][data.ix[j].notnull()].dropna().count()
                ser_id1 = i
                ser_id2 = j
    return [ser_max, ser_id1, ser_id2]

# Define a reduce function
def reduce_analysis(results):
    ser_max = 0
    ser_id1 = 0
    ser_id2 = 0
    for x_list in results:
        if ser_max < x_list[0]:
            ser_max = x_list[0]
            ser_id1 = x_list[1]
            ser_id2 = x_list[2]
    return ser_max, ser_id1, ser_id2

# Start pooling the process
pool = ThreadPool(4)
results = pool.map(map_analysis, [i for i in foo.index])
pool.close()
pool.join()

# Find the pairs id with max overlap
ser = reduce_analysis(results)
ser_max = ser[0]
ser_id1 = ser[1]
ser_id2 = ser[2]

"""
Non-parallelized Sampling Search
"""
# The following loop can be parallelized
ser_max = 0
ser_id1 = 0
ser_id2 = 0
num_of_samples = 2000
samples_index = np.random.permutation(foo.index)[:num_of_samples]
samples_columns = np.random.permutation(foo.columns)[:num_of_samples]
for i in samples_index:
    for j in samples_columns:
        # Count the number of the overlap movies of different users
        # Use .ix[] to set rows
        # Use [] to set columns
        # Use .ix[,] to set rows and columns both
        # foo is a symmetry matrix, so we only calculate half of it
        if i != j:
            if ser_max < data.ix[i][data.ix[j].notnull()].dropna().count():
                ser_max = data.ix[i][data.ix[j].notnull()].dropna().count()
                ser_id1 = i
                ser_id2 = j

# ser_id1 = 3618
# ser_id2 = 1181
# ser_max = 909

# Calculate the correlation of two users
data.ix[ser_id1].corr(data.ix[ser_id2])

# Find overlapped ratings of users pairs
test = data.reindex(index=[ser_id2,ser_id1],columns=data.ix[ser_id1][data.ix[ser_id2].notnull()].dropna().index)
test.ix[ser_id2].value_counts(sort=False).plot(kind='bar')
test.ix[ser_id1].value_counts(sort=False).plot(kind='bar')

# Sampling from the overlapped rated movies to calculate the correlation
periods_test = DataFrame(np.zeros((20,7)),columns=[int(ser_max/100),int(ser_max/50),int(ser_max/20),int(ser_max/10),int(ser_max/5),int(ser_max/2),ser_max])
for i in periods_test.index:   # Sampling 20 times
    for j in periods_test.columns:
         sample = test.reindex(columns=np.random.permutation(test.columns)[:j])
         periods_test.ix[i,j] = sample.iloc[0].corr(sample.iloc[1])  # ix is for label index, iloc is for int index
print periods_test[:5]
print periods_test.describe()

threshold = 0.1
temp_std = 0
# Take the threshold num which makes sampling correlation stable
for i, std in enumerate(periods_test.std()):
    if std < 0.1 and temp_std >= 0.1:
        mini_period = periods_test.columns[i]
        break
    temp_std = std

# Decide the value of min_periods. Set std 0.05 as threshold
# mini_period = 200
check_size = int(len(data.index) * 0.2)   # 20% dataset for testing
check = {}
check_data = data.copy() # Avoid the changes on original data
check_data = check_data.ix[check_data.count(axis=1) > mini_period]    # Filter users with few ratings. If there is no axis, the sum is the whole matrix
for user in np.random.permutation(check_data.index):
    movie = np.random.permutation(check_data.ix[user].dropna().index)[0]
    check[(user,movie)] = check_data.ix[user,movie]
    check_data.ix[user,movie] = np.nan
    check_size -= 1
    if not check_size:
        break

corr = check_data.T.corr(min_periods=mini_period)
corr_clean = corr.dropna(how='all') # del columns with all na
corr_clean = corr_clean.dropna(axis=1,how='all')    # del row with all na
check_ser = Series(check)  # 1000 real ratings

# New a series for storing predicted rating
result = Series(np.nan,index=check_ser.index)
for user,movie in result.index:#这个循环看着很乱，实际内容就是加权平均而已
        prediction = []
        if user in corr_clean.index:
            corr_set = corr_clean[user][corr_clean[user]>0.1].dropna()#仅限大于 0.1 的用户
        else:
            continue
        for other in corr_set.index:
            if  not np.isnan(data.ix[other,movie]) and other != user:#注意bool(np.nan)==True
                prediction.append((data.ix[other,movie],corr_set[other]))
        if prediction:
            result[(user,movie)] = sum([value*weight for value,weight in prediction])/sum([pair[1] for pair in prediction])
            #Why result has nan ?

result.dropna(inplace=True)
print result.corr(check_ser.reindex(result.index))  # Calculate the correlation of check_ser (training) and result (test)
(result-check_ser.reindex(result.index)).abs().describe()
end = time()

# =============================================================================
corr = data.T.corr(min_periods=mini_period)
corr_clean = corr.dropna(how='all')
corr_clean = corr_clean.dropna(axis=1,how='all')

lucky = np.random.permutation(corr_clean.index)[0]
gift = data.ix[lucky]
gift = gift[gift.isnull()]
corr_lucky = corr_clean[lucky].drop(lucky)#lucky 与其他用户的相关系数 Series，不包含 lucky 自身
corr_lucky = corr_lucky[corr_lucky>0.1].dropna()#筛选相关系数大于 0.1 的用户
for movie in gift.index:#遍历所有 lucky 没看过的电影
    prediction = []
    for other in corr_lucky.index:#遍历所有与 lucky 相关系数大于 0.1 的用户
        if not np.isnan(data.ix[other,movie]):
            prediction.append((data.ix[other,movie],corr_clean[lucky][other]))
    if prediction:
        gift[movie] = sum([value*weight for value,weight in prediction])/sum([pair[1] for pair in prediction])

gift.dropna().order(ascending=False)


