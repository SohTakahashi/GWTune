#%%
import numpy as np
import matplotlib.pyplot as plt
import ot
import time
import pickle as pkl
import tqdm
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.init_matrix import InitMatrix

from tqdm import tqdm

#%%
def entropic_gromov_wasserstein(C1, C2, p, q, loss_fun, epsilon,T=None,
                                max_iter=1000, tol=1e-9, verbose=False, log=False):
    C1, C2, p, q = ot.utils.list_to_array(C1, C2, p, q)
    nx = ot.backend.get_backend(C1, C2, p, q)
    # add T as an input
    if T is None:
      T = nx.outer(p, q)
    constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun)
    cpt = 0
    err = 1
    if log:
        log = {'err': []}
    
    T_list = []
    tens_list = []
    while (err > tol and cpt < max_iter):
        Tprev = T
        # compute the gradient
        tens = ot.gromov.gwggrad(constC, hC1, hC2, T)
        T = ot.bregman.sinkhorn(p, q, tens, epsilon, method='sinkhorn')
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.norm(T - Tprev)
            if log:
                log['err'].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt += 1
        
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.norm(T - Tprev)

            if log:
                log["err"].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(cpt, err))
        
        if cpt <= 10:
            T_list.append(T)
            tens_list.append(tens)
        
    if log:
        log['gw_dist'] = ot.gromov.gwloss(constC, hC1, hC2, T)
        return T, log, T_list, tens_list
    else:
        return T, T_list, tens_list

def create_circle_data(n_points):
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = np.cos(t)
    y = np.sin(t)
    one_data = np.vstack((x, y)).T
    return one_data

def add_noise_to_one_point(points, noise_deg=0.0001, point_index=0, seed=None):
    """
    Adds Gaussian noise to all dimensions of a single point in the point cloud.
    This is a sanity check to observe the effect of noise on a single point.
    """
    if seed is not None:
        np.random.seed(seed)
    noise = np.zeros_like(points)
    
    if isinstance(point_index, int):
        noise[point_index, :] = np.random.normal(loc=0.0, scale=noise_deg, size=points.shape[1])
    elif isinstance(point_index, list):
        for index in point_index:
            noise[index, :] = np.random.normal(loc=0.0, scale=noise_deg, size=points.shape[1])
            
    return points + noise

def add_noise_to_one_dimension(points, noise_deg=0.0001, dimension=0, seed=None):
    """
    Adds Gaussian noise to only one dimension of each point in the point cloud.
    This is a sanity check to observe the effect of noise on a single dimension.
    """
    if seed is not None:
        np.random.seed(seed)
    noise = np.zeros_like(points)
    noise[:, dimension] = np.random.normal(loc=0.0, scale=noise_deg, size=points.shape[0])
    return points + noise
#%%

# Create two circles
n_points = 20
shape1 = create_circle_data(n_points)
shape1 = add_noise_to_one_point(shape1, noise_deg=0.1, point_index=[0, 1, 4, 6, 8, 11, 14, 18], seed=0)
shape2 = add_noise_to_one_dimension(shape1, noise_deg=0.1, dimension=0, seed=0)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(shape1[:, 0], shape1[:, 1])
plt.title("Shape 1")
plt.subplot(1, 2, 2)
plt.scatter(shape2[:, 0], shape2[:, 1])
plt.title("Shape 2")
plt.show()

#%%

# Compute the distance matrix
C1 = cdist(shape1, shape1)
C2 = cdist(shape2, shape2)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(C1)
plt.colorbar()
plt.title("C1")

plt.subplot(1, 2, 2)
plt.imshow(C2)
plt.colorbar()
plt.title("C2")

plt.show()
# %%
# Compute the entropic Gromov-Wasserstein distance
p = np.ones(n_points) / n_points
q = np.ones(n_points) / n_points
epsilon = 0.1

uniform_init_T = np.outer(p, q)

T, log, T_list, tens_list = entropic_gromov_wasserstein(
    C1, C2, p, q, "square_loss", epsilon, T=uniform_init_T, max_iter=200, tol=1e-9, verbose=False, log=True)

print("Gromov-Wasserstein distance:", log["gw_dist"])
# %%

# visualize T_list and tens_list
# T_list on the top row, tens_list on the bottom row
num_fig = len(T_list)
plt.figure(figsize=(20, 5))
plt.suptitle(f"Uniform, epsilon={epsilon}", fontsize=16)
for i in range(num_fig):
    plt.subplot(2, num_fig, i+1)
    plt.imshow(T_list[i])
    # plt.colorbar()
    plt.title(f"T_{i}")

    plt.subplot(2, num_fig, i+num_fig+1)
    plt.imshow(tens_list[i])
    # plt.colorbar()
    plt.title(f"Tens_{i}")
plt.show()
# %%

# random initialization
init_matrix = InitMatrix(n_points, n_points)

best_gwd = np.inf
for seed in tqdm(range(100)):
    random_init_T = init_matrix.make_initial_T("random", seed)
    # # visualize the random initial matrix
    # plt.figure(figsize=(5, 5))
    # plt.imshow(random_init_T)
    # plt.colorbar()
    # plt.title(f"Random init T with seed {seed}")
    # plt.show()
    
    T, log, T_list, tens_list = entropic_gromov_wasserstein(
        C1, C2, p, q, "square_loss", epsilon, T=random_init_T, max_iter=200, tol=1e-9, verbose=False, log=True)
    
    # print("Gromov-Wasserstein distance:", log["gw_dist"])
    gwd = log["gw_dist"]
    
    if gwd < best_gwd:
        best_gwd = gwd
        best_seed = seed
        best_T = T
        best_log = log
        best_T_list = T_list
        best_tens_list = tens_list

print("Best Gromov-Wasserstein distance:", best_gwd)
print("Best seed:", best_seed)
# %%
# visualize T_list and tens_list
num_fig = len(best_T_list)
plt.figure(figsize=(20, 5))
plt.suptitle(f"Random, epsilon={epsilon}", fontsize=16)
for i in range(num_fig):
    plt.subplot(2, num_fig, i+1)
    plt.imshow(best_T_list[i])
    # plt.colorbar()
    plt.title(f"T_{i}")

    plt.subplot(2, num_fig, i+num_fig+1)
    plt.imshow(best_tens_list[i])
    # plt.colorbar()
    plt.title(f"Tens_{i}")
plt.show()
# %%
diag_init_T = init_matrix.make_initial_T("diag")

T, log, T_list, tens_list = entropic_gromov_wasserstein(
    C1, C2, p, q, "square_loss", epsilon, T=diag_init_T, max_iter=200, tol=1e-9, verbose=False, log=True)

print("Gromov-Wasserstein distance:", log["gw_dist"])

# visualize T_list and tens_list
num_fig = len(T_list)
plt.figure(figsize=(20, 5))
plt.suptitle(f"Diagonal, epsilon={epsilon}", fontsize=16)
for i in range(num_fig):
    plt.subplot(2, num_fig, i+1)
    plt.imshow(T_list[i])
    # plt.colorbar()
    plt.title(f"T_{i}")

    plt.subplot(2, num_fig, i+num_fig+1)
    plt.imshow(tens_list[i])
    # plt.colorbar()
    plt.title(f"Tens_{i}")
plt.show()
# %%
