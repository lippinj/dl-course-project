import time
import numpy as np


# This script reads the data from the data/ folder into a .npy file,
# which is faster to load. The file will contain a matrix
# like:
#  [[customerID, movieID, rating]
#   [customerID, movieID, rating]
#                ...
#   [customerID, movieID, rating]]
# In addition, the customerIDs will be contiguous (which they aren't
# in the original data).
#
# The following file is created:
#  data.npy
#
# Run examples:
#  python preprocess.py
#  python preprocess.py

##################
# Read full data #
##################

def read_data_from(filename):
    """Read data points from the given file.
    
    Returns a np.array whose rows are data points:
      [[customer, movie, rating]
       [customer, movie, rating]
                   ...
       [customer, movie, rating]]
    """
    ret = []
    movie_id = None
    with open(filename, 'r') as f:
        for n, line in enumerate(f):
            line = line.rstrip()
            
            # Detect the start of data for a new movie
            if line[-1] == ':':
                movie_id = int(line[:-1]) - 1
                continue
            
            # Detect a new rating for the current movie
            assert(movie_id is not None)
            line = line.split(',')
        
            customer_id = int(line[0])
            rating = int(line[1])
            ret.append([customer_id, movie_id, rating])
            
    return np.array(ret)

datas = []
for i in (1, 2, 3, 4):
    t0 = time.time()
    filename = 'data/combined_data_{}.txt'.format(i)
    datas.append(read_data_from(filename))
    t1 = time.time()
    print('Read data file {}/4 ({}) in {:.1f} s.'.format(i, filename, t1 - t0))

t0 = time.time()
data0 = np.concatenate(datas)
t1 = time.time()

print('Combined to {:,} training data points in {:.1f} s.'.format(data0.shape[0], t1 - t0))

#####################################
# Make and construct customer index #
#####################################

def make_customer_index(data):
    """Returns a mapping from arbitrary customer_ids to contiguous ids."""
    S = set()
    for i in range(data.shape[0]):
        S.add(data[i][0])
    S = list(S)
    S.sort()
    
    I = {}
    for j, i in enumerate(S):
        I[i] = j
        
    return I

def apply_customer_index(data, I):
    """Maps customer_ids to contiguous ids in data (inplace)."""
    for i in range(data.shape[0]):
        data[i][0] = I[data[i][0]]

t0 = time.time()
I = make_customer_index(data0)
t1 = time.time()
print('Indexed {:,} customer IDs in {:.1f} s.'.format(len(I), t1 - t0))

apply_customer_index(data0, I)
t2 = time.time()
print('Applied index in {:.1f} s.'.format(t2 - t1))

########
# Save #
########

filename = 'data.npy'
np.save(filename, data0)
print('Saved to {}.'.format(filename))
