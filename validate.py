import sys
import time

import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# Calculates the validation error.
#
# The following file(s) are read:
#  mean_ratings_[#].npy
#  validate_[#].npy
#  naive_me_[#].pth
#  naive_ce_[#].pth
#  naive_pa_[#].pth
#
# Run examples:
#  python validate.py 10k
#  python validate.py all

######################
# Parse command line #
######################

SPACER = '=' * 78
print('================================ Validation =================================')

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

print(SPACER)

########################
# Read validation data #
########################

filename_data = 'validate_{}.npy'.format(customers_str)

t0 = time.time()
data         = np.load(filename_data)
customer_ids = torch.tensor(data[:,0], dtype=torch.long).view(-1)
movie_ids    = torch.tensor(data[:,1], dtype=torch.long).view(-1)
ratings      = torch.tensor(data[:,2], dtype=torch.float).view(-1)
num_points   = data.shape[0]
t1 = time.time()

del data

print('Read {:,} data points from {} in {:.1f} s.'.format(num_points, filename_data, t1 - t0))

#####################
# Read mean ratings #
#####################

t0 = time.time()

filename_mr = 'mean_ratings_{}.npy'.format(customers_str)
mean_ratings = torch.tensor(np.load(filename_mr), dtype=torch.float)
mean_ratings = mean_ratings[movie_ids]
mean_all = torch.mean(mean_ratings)

t1 = time.time()

print('Read means from {} in {:.1f} s.'.format(filename_mr, t1 - t0))
print('Overall mean is {:.4f}'.format(mean_all))


#######################
# Model specification #
#######################

filename_me = 'naive_me_{}.pth'.format(customers_str)
filename_ce = 'naive_ce_{}.pth'.format(customers_str)
filename_pa = 'naive_pa_{}.pth'.format(customers_str)

dim_customers = 20 # customer embedding dimensions
dim_movies = 20 # movie embedding dimensions

t0 = time.time()

movie_embedding = nn.Embedding(num_movies, dim_movies)
movie_embedding.load_state_dict(torch.load(filename_me))
movie_embedding.eval()

customer_embedding = nn.Embedding(num_customers, dim_customers)
customer_embedding.load_state_dict(torch.load(filename_ce))
customer_embedding.eval()

predict_appeal = nn.Sequential(
        nn.Linear(dim_customers + dim_movies, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.Tanh(),
        nn.Linear(100, 10),
        nn.Tanh(),
        nn.Linear(10, 1)
    )
predict_appeal.load_state_dict(torch.load(filename_pa))
predict_appeal.eval()

t1 = time.time()

print('Loaded models in {:.1f} s'.format(t1 - t0))

print(SPACER)

##############################
# Calculate validation error #
##############################

m = movie_embedding(movie_ids)
c = customer_embedding(customer_ids)
a = predict_appeal(torch.cat((c, m), dim=1)).view(num_points)
p = F.hardtanh(mean_ratings + a, 1.0, 5.0)

criterion = nn.MSELoss(reduction='mean')
mse = criterion(p, ratings).item()
rmse = np.sqrt(mse)

print('Validation MSE:  {:.4f}'.format(mse))
print('Validation RMSE: {:.4f}'.format(rmse))
