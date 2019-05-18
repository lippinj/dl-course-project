import sys
import time

import numpy as np


# This script splits data from data.npy into training and validation
# data sets, considering only customer IDs below the given limit.
#
# The following file(s) are read:
#  data.npy
#
# The following file(s) are created:
#  train_[#].npy
#  validate_[#].npy
#
# Run examples:
#  python preprocess.py 10k
#  python preprocess.py all

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

##################
# Read full data #
##################

t0 = time.time()
data0 = np.load('data.npy')
t1 = time.time()

print('Read {:,} training data points from data.npy in {:.1f} s.'.format(data0.shape[0], t1 - t0))

###########################
# Limit by customer count #
###########################

t0 = time.time()
I = data0[:,0] < num_customers
data = data0[I]
t1 = time.time()
    
print('Limited to {:,} customers for a total of {:,} data points in {:.1f} s.'.format(num_customers, data.shape[0], t1 - t0))

#########################################
# Split to training and validation sets #
#########################################

validate_fraction = 0.10

t0 = time.time()
train_count = int(data.shape[0] * (1.0 - validate_fraction))
np.random.shuffle(data)
train = data[:train_count,:]
validate = data[train_count:,:]
t1 = time.time()

print('Split to {:,} training and {:,} validation samples in {:.1f} s.'.format(train.shape[0], validate.shape[0], t1 - t0))

########
# Sort #
########

t0 = time.time()
train = train[np.argsort(train[:,0])]
t1 = time.time()
validate = validate[np.argsort(validate[:,0])]
t2 = time.time()

print('Sorted in {:.1f} s (train) and {:.1f} s (validate).'.format(t1 - t0, t2 - t1))

########
# Save #
########

filename_train = 'train_{}.npy'.format(customers_str)
filename_validate = 'validate_{}.npy'.format(customers_str)

while True:
    command = input('[s]ave or [q]uit? ')
    if command == 's':
        np.save(filename_train, train)
        print('Training data saved to {}'.format(filename_train))
        np.save(filename_validate, validate)
        print('Validation data saved to {}'.format(filename_validate))
        break

    elif command == 'q':
        print('Did not save data.')
        break

    else:
        print('Unrecognized command.')

