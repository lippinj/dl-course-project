import sys
import time

import numpy as np


# This script finds the mean rating of each movie in the training data,
# and calculates the training and validation error when each movie's
# rating is predicted to be simply its mean.

# The following file is created:
#  mean_ratings_[#].npy
#
# Run examples:
#  python compute_means.py 10k
#  python compute_means.py all

######################
# Parse command line #
######################

assert(len(sys.argv) == 2)
customers_str = sys.argv[1]

assert(customers_str in ('1k', '10k', '25k', '100k', 'all'))
num_customers = {
        '1k'  :   1000,
        '10k' :  10000,
        '25k' :  25000,
        '100k': 100000,
        'all' : 480189
    }[customers_str]
print('Number of customers: {:>8,}'.format(num_customers))

num_movies = 17770
print('Number of movies:    {:>8,}'.format(num_movies))

#####################################
# Read training and validation data #
#####################################

filename_train = 'train_{}.npy'.format(customers_str)
filename_validate = 'validate_{}.npy'.format(customers_str)

t0 = time.time()
train = np.load(filename_train)
num_train_points = train.shape[0]
t1 = time.time()
print('Read {:,} training points from {} in {:.1f} s.'.format(num_train_points, filename_train, t1 - t0))

validate = np.load(filename_validate)
num_validate_points = validate.shape[0]
t2 = time.time()
print('Read {:,} validation points from {} in {:.1f} s.'.format(num_validate_points, filename_validate, t2 - t1))

##########################
# Calculate mean ratings #
##########################

t0 = time.time()

sum_all = 0.0
count_all = 0.0
sums = np.zeros(num_movies)
counts = np.zeros(num_movies)

for i in range(train.shape[0]):
    mid = int(train[i,1])
    rating = train[i,2]
    sums[mid] += rating
    counts[mid] += 1.0
    sum_all += rating
    count_all += 1.0

mean_all = sum_all / count_all
for i in range(num_movies):
    if counts[i] == 0.0:
        counts[i] = 1.0
        sums[i] = mean_all

means = sums / counts

t1 = time.time()
print('Calculated means in {:.1f} s.'.format(t1 - t0))
print('Overall mean is {:.4f}'.format(mean_all))

filename_mr = 'mean_ratings_{}.npy'.format(customers_str)
np.save(filename_mr, means)
print('Saved means to {}'.format(filename_mr))

####################
# Calculate errors #
####################

for data, name in ((train, 'training'), (validate, 'validation')):
    t0 = time.time()

    mids = np.array(data[:,1], dtype=np.int)
    predicted = means[mids]
    ratings = data[:,2]
    errors = predicted - ratings
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)

    t1 = time.time()
    print('Calculated {} error in {:.1f} s.'.format(name, t1 - t0))
    print('mse: {:.4f}, rmse: {:.4f}'.format(mse, rmse))
