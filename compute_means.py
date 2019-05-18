import sys
import time

import numpy as np


# This script finds the mean rating of each movie in the training data,
# and calculates the training and validation error when each movie's
# rating is predicted to be simply its mean.

# The following file(s) are read:
#  train_[#].npy
#
# The following file(s) are created:
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
tally = np.zeros((num_movies, 5))

for i in range(num_train_points):
    mid = int(train[i,1])
    rating = int(train[i,2])
    tally[mid,rating-1] += 1.0

total = np.sum(tally, axis=0)
total /= np.sum(total)

for i in range(num_movies):
    if np.sum(tally[i,:]) == 0.0:
        tally[i,:] = total

tally /= np.sum(tally, axis=1)[:,None]
means = np.dot(tally, np.arange(1, 6))
mean_all = np.dot(total, np.arange(1, 6))

t1 = time.time()
print('Calculated means in {:.1f} s.'.format(t1 - t0))
print('Overall mean is {:.4f}'.format(mean_all))

filename_mr = 'mean_ratings_{}.npy'.format(customers_str)
np.save(filename_mr, (tally, total, means, mean_all))
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
