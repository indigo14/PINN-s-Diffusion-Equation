# main.py

import torch
import torch.optim as optim
from model import FCN
from utils import generate_data, prepare_training_data, plot3D_Matrix, plot3D
from config import *

# Set random seed for reproducibility
torch.manual_seed(seed)

# Generate data
x, t, X, T, y_real, x_test, y_test, lb, ub = generate_data()

# Verify generated data shapes and values
print("Generated Data Shapes:")
print(f"x: {x.shape}, t: {t.shape}, X: {X.shape}, T: {T.shape}, y_real: {y_real.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}")
print("Generated Data Ranges:")
print(f"x: [{x.min().item()}, {x.max().item()}], t: [{t.min().item()}, {t.max().item()}]")
print(f"y_real: [{y_real.min().item()}, {y_real.max().item()}]")
print("Sample y_real Values:")
print(y_real[:5, :5])

# Prepare training data
X_train_Nu, Y_train_Nu, X_train_Nf = prepare_training_data(X, T, lb, ub)

# Transfer data to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_Nu = X_train_Nu.float().to(device)
Y_train_Nu = Y_train_Nu.float().to(device)
X_train_Nf = X_train_Nf.float().to(device)
f_hat = torch.zeros(X_train_Nf.shape[0], 1).to(device)
X_test = x_test.float().to(device)
Y_test = y_test.float().to(device)

# Create model
PINN = FCN(layers).to(device)# PINN is the neural network model defined using the class FCN. Initializes the neural network model and transfers it to the GPU
PINN.X_train_Nu = X_train_Nu
PINN.Y_train_Nu = Y_train_Nu
PINN.X_train_Nf = X_train_Nf
PINN.X_test = X_test
PINN.Y_test = Y_test

# Initialize optimizer
optimizer = optim.Adam(PINN.parameters(), lr=lr)
PINN.optimizer = optimizer  # Set the optimizer in the model

# Training loop
for epoch in range(steps):
    optimizer.step(PINN.closure)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {PINN.loss(X_train_Nu, Y_train_Nu, X_train_Nf).item()}")

# Verify Data Shapes and Model Predictions
with torch.no_grad():
    y1 = PINN(X_test).cpu()
    arr_x1 = X_test[:, 0].reshape((total_points_t, total_points_x)).T
    arr_T1 = X_test[:, 1].reshape((total_points_t, total_points_x)).T
    arr_y1 = y1.reshape((total_points_t, total_points_x)).T

    # Verify data shapes
    print("Shapes of Testing Data and Predictions:")
    print(f"arr_x1: {arr_x1.shape}, arr_T1: {arr_T1.shape}, arr_y1: {arr_y1.shape}")
    print("Value Range of Model Predictions:")
    print(f"arr_y1: [{arr_y1.min().item()}, {arr_y1.max().item()}]")
    print("Sample Model Predictions:")
    print(arr_y1[:5, :5])

    # Verify true solution data
    y_real = y_real.cpu()
    print("True Solution Shape and Values:")
    print(f"y_real: {y_real.shape}")
    print(f"y_real: [{y_real.min().item()}, {y_real.max().item()}]")
    print("Sample True Solution Values:")
    print(y_real[:5, :5])

    # Plot results
    plot3D_Matrix(arr_x1, arr_T1, arr_y1)
    plot3D_Matrix(X, T, y_real)

    # Plot individual components for inspection
    plot3D(x, t, y_real)
