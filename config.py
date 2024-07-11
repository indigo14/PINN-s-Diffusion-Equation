import numpy as np

# Tuning Parameters
steps = 20000
lr = 1e-3
layers = np.array([2, 32, 32, 1])  # Hidden layers

# Domain bounds
x_min, x_max = -1, 1
t_min, t_max = 0, 1
total_points_x = 100
total_points_t = 200

# Training and testing sizes
Nu = 100  # Number of boundary condition points
Nf = 10000  # Number of collocation points

# Set seed for reproducibility
seed = 123
