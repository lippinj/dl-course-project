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


######################
# Parse command line #
######################

SPACER = '=' * 78
print('========================= Training: Naive embeddings ========================')

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
data          = np.load(filename_data)
customer_ids0 = torch.tensor(data[:,0], dtype=torch.long, device=device).view(-1)
movie_ids0    = torch.tensor(data[:,1], dtype=torch.long, device=device).view(-1)
ratings0      = torch.tensor(data[:,2], dtype=torch.float, device=device).view(-1)
num_points    = data.shape[0]
t1 = time.time()

del data

print('Read {:,} data points from {} in {:.1f} s.'.format(num_points, filename_data, t1 - t0))
print('Average ratings per customer: {:.2f}.'.format(float(num_points) / num_customers))

#####################
# Read mean ratings #
#####################

t0 = time.time()

filename_mr = 'mean_ratings_{}.npy'.format(customers_str)
mean_ratings0 = torch.tensor(np.load(filename_mr), dtype=torch.float, device=device)
mean_ratings0 = mean_ratings0[movie_ids0]
mean_all = torch.mean(mean_ratings0)

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

movie_embedding = nn.Embedding(num_movies, dim_movies).to(device)
customer_embedding = nn.Embedding(num_customers, dim_customers).to(device)
predict_appeal = nn.Sequential(
        nn.Linear(dim_customers + dim_movies, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.Tanh(),
        nn.Linear(100, 10),
        nn.Tanh(),
        nn.Linear(10, 1),
    ).to(device)

if extra == 'continue':
    t0 = time.time()
    movie_embedding.load_state_dict(torch.load(filename_me))
    customer_embedding .load_state_dict(torch.load(filename_ce))
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
        I = torch.randperm(num_points, device=device)

        i = 0
        sum_square_error = 0.0
        n_batches = 0
        while i < num_points:
            j = min(i + batch_size, num_points)
            J = I[i:j]
            B = j - i

            customer_ids = customer_ids0[J]
            movie_ids = movie_ids0[J]
            ratings = ratings0[J]
            mean_ratings = mean_ratings0[J]

            for opt in opts:
                opt.zero_grad()

            m = movie_embedding(movie_ids)
            c = customer_embedding(customer_ids)
            a = predict_appeal(torch.cat((c, m), dim=1)).view(B)
            p = F.hardtanh(mean_ratings + a, 1.0, 5.0)

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
L += train(100, [2e-1, 2e-1, 2e-1], 8192)

print(SPACER)

##################
# Saving results #
##################

start_rmse = np.sqrt(L[0])
end_rmse = np.sqrt(L[1])
tag = 'train_naive_{}_{:.5f}_{:.5f}'.format(customers_str, start_rmse, end_rmse)
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

filename_me = 'naive_me_{}.pth'.format(customers_str)
filename_ce = 'naive_ce_{}.pth'.format(customers_str)
filename_rp = 'naive_rp_{}.pth'.format(customers_str)

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

