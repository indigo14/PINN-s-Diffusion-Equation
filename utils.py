import torch
import numpy as np
from pyDOE import lhs
import matplotlib.pyplot as plt
from config import *

def plot3D(x, t, y):
    x_plot = x.squeeze(1)
    t_plot = t.squeeze(1)
    X, T = torch.meshgrid(x_plot, t_plot)
    Y = y.reshape(X.shape)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X.cpu().numpy(), T.cpu().numpy(), Y.cpu().numpy(), cmap='viridis')
    plt.show()

def plot3D_Matrix(x, t, y):
    X, T = x, t
    F_xt = y

    print("Shapes:", X.shape, T.shape, F_xt.shape)
    print("Value Range:", F_xt.min(), F_xt.max())

    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(T.cpu().numpy(), X.cpu().numpy(), F_xt.cpu().numpy(), cmap='viridis')
    fig.colorbar(cp)
    plt.xlabel('Time (t)')
    plt.ylabel('Space (x)')
    plt.title('Model Predictions')
    plt.show()

def generate_data():
    x = torch.linspace(x_min, x_max, total_points_x).view(-1, 1)
    t = torch.linspace(t_min, t_max, total_points_t).view(-1, 1)
    
    X, T = torch.meshgrid(x.squeeze(1), t.squeeze(1))
    
    def f_real(x, t):
        return torch.exp(-t) * torch.sin(np.pi * x)
    
    y_real = f_real(X, T)
    
    x_test = torch.hstack((X.transpose(1, 0).flatten()[:, None], T.transpose(1, 0).flatten()[:, None]))
    y_test = y_real.transpose(1, 0).flatten()[:, None]
    
    lb = x_test[0]
    ub = x_test[-1]
    
    return x, t, X, T, y_real, x_test, y_test, lb, ub

def prepare_training_data(X, T, lb, ub):
    left_X = torch.hstack((X[:, 0][:, None], T[:, 0][:, None]))  # First column
    left_Y = torch.sin(np.pi * left_X[:, 0]).unsqueeze(1)
    
    bottom_X = torch.hstack((X[0, :][:, None], T[0, :][:, None]))  # First row
    bottom_Y = torch.zeros(bottom_X.shape[0], 1)
    
    top_X = torch.hstack((X[-1, :][:, None], T[-1, :][:, None]))  # Last row
    top_Y = torch.zeros(top_X.shape[0], 1)
    
    X_train = torch.vstack([left_X, bottom_X, top_X])
    Y_train = torch.vstack([left_Y, bottom_Y, top_Y])
    
    idx = np.random.choice(X_train.shape[0], Nu, replace=False)
    X_train_Nu = X_train[idx, :]
    Y_train_Nu = Y_train[idx, :]
    
    X_train_Nf = lb + (ub - lb) * lhs(2, Nf)
    X_train_Nf = torch.vstack((X_train_Nf, X_train_Nu))
    
    return X_train_Nu, Y_train_Nu, X_train_Nf
