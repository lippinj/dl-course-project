import sys
import time

import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Train the models. Change the "Training execution" part to adjust
# number of epochs etc. Use "continue" to load a previously trained
# model and keep training it.
#
# The following file(s) are read:
#  train_[#].npy
#  tally_[#].npy
#
# The following file(s) are created:
#  results/train_naive_2_[#]_[#.####]_[#.####].png
#  results/train_naive_2_[#]_[#.####]_[#.####].losses.npy
#  naive_2_me_[#].pth
#  naive_2_ce_[#].pth
#  naive_2_pa_[#].pth
#
# Run examples:
#  python train_naive.py cpu 10k
#  python train_naive.py cuda all
#  python train_naive.py cuda all continue

######################
# Parse command line #
######################

SPACER = '=' * 78
print('======================= Training: Naive embeddings v2 ====================')

assert(len(sys.argv) in (3, 4))
device_str = sys.argv[1]
customers_str = sys.argv[2]

extra = None
if len(sys.argv) == 4:
    extra = sys.argv[3]
    assert(extra in ('continue',))

assert(device_str in ('cpu', 'cuda'))
device = torch.device(device_str)
print('Using device:        {:>8}'.format(device_str))

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

######################
# Read training data #
######################

filename_data = 'train_{}.npy'.format(customers_str)

t0 = time.time()
data       = np.load(filename_data)
num_points = data.shape[0]
t1 = time.time()

print('Read {:,} data points from {} in {:.1f} s.'.format(num_points, filename_data, t1 - t0))
print('Average ratings per customer: {:.2f}.'.format(float(num_points) / num_customers))

#####################
# Read mean ratings #
#####################

t0 = time.time()
filename_ta = 'tally_{}.npy'.format(customers_str)
tally = np.load(filename_ta)
tally = np.log(tally)
tally = tally[np.array(data[:,1], dtype=np.int)]
t1 = time.time()

print('Read tally from {} in {:.1f} s.'.format(filename_ta, t1 - t0))

#######################
# Model specification #
#######################

filename_me = 'naive_2_me_{}.pth'.format(customers_str)
filename_ce = 'naive_2_ce_{}.pth'.format(customers_str)
filename_pa = 'naive_2_pa_{}.pth'.format(customers_str)

dim_customers = 20 # customer embedding dimensions
dim_movies = 20 # movie embedding dimensions

movie_embedding = nn.Embedding(num_movies, dim_movies).to(device)
customer_embedding = nn.Embedding(num_customers, dim_customers).to(device)
predict_appeal = nn.Sequential(
        nn.Linear(dim_customers + dim_movies, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.Tanh(),
        nn.Linear(100, 20),
        nn.Tanh(),
        nn.Linear(20, 5)
    ).to(device)

if extra == 'continue':
    t0 = time.time()
    movie_embedding.load_state_dict(torch.load(filename_me))
    customer_embedding.load_state_dict(torch.load(filename_ce))
    predict_appeal.load_state_dict(torch.load(filename_pa))
    t1 = time.time()
    print('Loaded models in {:.1f} s'.format(t1 - t0))


print(SPACER)

#################
# Training spec #
#################

def train(num_epochs, lrs=[1e-1, 1e-1, 1e-1], batch_size=10000):
    L = []

    opts = []
    opts.append(optim.SGD(movie_embedding.parameters(), lr=lrs[0]))
    opts.append(optim.SGD(customer_embedding.parameters(), lr=lrs[1]))
    opts.append(optim.SGD(predict_appeal.parameters(), lr=lrs[2]))
    criterion = nn.MSELoss(reduction='mean')
    
    for i_epoch in range(num_epochs):
        # Epoch start.
        t0 = time.time()
        # I = torch.randperm(num_points, device=device)
        I = np.random.choice(num_points, num_points, replace=False)

        i = 0
        sum_square_error = 0.0
        n_batches = 0
        while i < num_points:
            j = min(i + batch_size, num_points)
            J = I[i:j]
            B = j - i

            customer_ids = torch.tensor(data[J,0], dtype=torch.long, device=device).view(B)
            movie_ids    = torch.tensor(data[J,1], dtype=torch.long, device=device).view(B)
            ratings      = torch.tensor(data[J,2], dtype=torch.float, device=device).view(B)
            logdists     = torch.tensor(tally[J], dtype=torch.float, device=device).view(B, 5)

            for opt in opts:
                opt.zero_grad()

            m = movie_embedding(movie_ids)
            c = customer_embedding(customer_ids)

            appeal = predict_appeal(torch.cat((c, m), dim=1)).view(B, 5)
            dist = F.softmax(logdists + appeal, dim=1)
            p = torch.mm(dist, torch.tensor([1., 2., 3., 4., 5.], device=device).view(5, 1)).view(B)

            loss = criterion(p, ratings)
            loss.backward()

            for opt in opts:
                opt.step()

            sum_square_error += B * loss.item()
        
            i = j
            n_batches += 1

        # Epoch end.
        t1 = time.time()
        mse = sum_square_error / num_points
        rmse = np.sqrt(mse)
        print('EPOCH {:>4} | {:>8,} b {:>12,} p | MSE: {:>6.4f} RMSE: {:>6.4f} | {:8.2f} s'.format(
            i_epoch + 1, n_batches, num_points, mse, rmse, t1 - t0))
        L.append(mse)

    return L

######################
# Training execution #
######################

L = []
L += train(25, [2e-1, 2e-1, 2e-1], 8192)

print(SPACER)

##################
# Saving results #
##################

start_rmse = np.sqrt(L[0])
end_rmse = np.sqrt(L[1])
tag = 'train_naive_2_{}_{:.5f}_{:.5f}'.format(customers_str, start_rmse, end_rmse)
filename_plot = 'results/{}.png'.format(tag)
filename_losses = 'results/{}.losses.npy'.format(tag)

np.save(filename_losses, np.array(L))
print('Losses saved to {}'.format(filename_losses))

plt.figure(figsize=(12, 8))
plt.plot(np.sqrt(L))
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.savefig(filename_plot)
print('Plot saved to {}'.format(filename_plot))

while True:
    command = input('[s]ave or [q]uit? ')
    if command == 's':
        torch.save(movie_embedding.state_dict(), filename_me)
        print('Movie embedding saved to {}'.format(filename_me))
        torch.save(customer_embedding.state_dict(), filename_ce)
        print('Customer embedding saved to {}'.format(filename_ce))
        torch.save(predict_appeal.state_dict(), filename_pa)
        print('Appeal predictor saved to {}'.format(filename_pa))
        break

    elif command == 'q':
        print('Did not save data.')
        break

    else:
        print('Unrecognized command.')

