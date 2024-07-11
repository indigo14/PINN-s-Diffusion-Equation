import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

class FCN(nn.Module):
    """
    Superclass Initialization: Ensures the base class nn.Module is properly initialized.
    Activation Function: Uses Tanh to introduce non-linearity.
    Loss Function: Uses MSE loss to measure the error.
    Layer Initialization: Creates a sequence of linear layers based on the provided layers list.
    Optimizer Iteration Counter: Keeps track of the number of iterations for the optimizer.
    Xavier Initialization: Properly initializes weights and biases to aid in training convergence.
    """
    def __init__(self, layers):#Constructor method that initializes the neural network with the given layers.
        super().__init__()#necessary to properly initialize the base class part of the FCN objec
        self.activation = nn.Tanh()#introduces non-linearity into the model
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])#Initializes a list of linear layers using nn.ModuleList.
        self.iter = 0 #Initializes an iteration counter to be used by the optimizer.

        '''Initializes the weights of the linear layers using the Xavier normal initialization method. This method helps in maintaining the variance of weights throughout the network, which can help in faster convergence'''
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)#The gain parameter is set to 1.0, which is the recommended value for the Tanh activation function.
            nn.init.zeros_(self.linears[i].bias.data)#the biases of the i-th linear layer to zero
        self.optimizer = None  # Placeholder for optimizer
        self.X_train_Nu = None
        self.Y_train_Nu = None
        self.X_train_Nf = None
        self.X_test = None
        self.Y_test = None

        
        """
            The forward method defines the forward pass of the neural network, which is how input data passes through the network to produce an output.
            Input Handling: Converts input data to a PyTorch tensor if it is not already one and ensures it is of float type.
            Layer Processing: Passes the input through each layer, applying the linear transformation followed by the activation function, except for the final layer where only the linear transformation is applied.
            Output: Produces and returns the final output of the network.
        """   
      

    def forward(self, x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)#If x is not a tensor (likely a NumPy array), it converts x to a PyTorch tensor using torch.from_numpy
        a = x.float()
        for i in range(len(self.linears) - 2):#Loops through all but the last layer in the list of linear layers (self.linears).
            z = self.linears[i](a)# Applies the i-th linear layer to the input a, producing the output z.
            a = self.activation(z)# #Applies the activation function (Tanh) to the output z, producing the activated output a. This introduces non-linearity into the network.
        a = self.linears[-1](a)#Applies the last linear layer (output layer) to the activated output a. the activation function is not applied to the output layer, because the final output does not require non-linearity, especially in regression tasks.
        return a
        '''
        lossBC Method: Computes the mean squared error between the network's predictions and the target values for the boundary conditions.
        lossPDE Method:Computes the mean squared error for the PDE residual. This involves calculating the first and second derivatives of the network's output with respect to the input points and ensuring the network satisfies the PDE.
        loss Method:Combines the BC loss and the PDE loss to compute the total loss, which guides the training of the neural network to respect both the boundary conditions and the PDE.
        '''
    def lossBC(self, x_BC, y_BC):
        loss_BC = self.loss_function(self.forward(x_BC), y_BC)#self.forward(x_BC): Passes the boundary input points through the network to get predictions.
        return loss_BC

    def lossPDE(self, x_PDE):#The input data points for evaluating the PDE
        g = x_PDE.clone()#Clones the input points to avoid modifying the original data.
        g.requires_grad = True#Enable differentiation
        f = self.forward(g)#Passes the cloned input points through the network to get predictions.
        f_x_t = autograd.grad(f, g, torch.ones([g.shape[0], 1]).to(g.device), retain_graph=True, create_graph=True)[0]#Computes the first derivative of the network's output with respect to g
        f_xx_tt = autograd.grad(f_x_t, g, torch.ones(g.shape).to(g.device), create_graph=True)[0]#Computes the second derivative of the network's output with respect to g.
        f_t = f_x_t[:, [1]]#select the 2nd element for t (the first one is x) Selects the derivative with respect to time (t)
        f_xx = f_xx_tt[:, [0]]#select the 1st element for x (the second one is t) Selects the second derivative with respect to space (x)
        f = f_t - f_xx + torch.exp(-g[:, 1:]) * (torch.sin(np.pi * g[:, 0:1]) - np.pi ** 2 * torch.sin(np.pi * g[:, 0:1]))##The final f expression represents the residual of the PDE. It is the difference between the first derivative with respect to time and the second derivative with respect to space, plus the exact solution of the PDE.
        return self.loss_function(f, torch.zeros_like(f))# Computes the mean squared error between the residual and the target values (f_hat), which are typically zeros

    def loss(self, x_BC, y_BC, x_PDE):# The input data points at the boundary. y_BC: The target values at the boundary. x_PDE: The input data points for evaluating the PDE
        loss_bc = self.lossBC(x_BC, y_BC)#Computes the boundary condition loss
        loss_pde = self.lossPDE(x_PDE)#Computes the PDE loss.
        return loss_bc + loss_pde #Returns the total loss, which is the sum of the boundary condition loss and the PDE loss.

        '''optimizer.zero_grad(): Clears previous gradients to prevent accumulation.
    self.loss(...): Computes the total loss using the model's loss function, which includes boundary condition and PDE losses.
    loss.backward(): Performs backpropagation to compute gradients.
    self.iter += 1: Increments the iteration counter for tracking.
    Logging and Evaluation:
        Every 100 iterations, the method computes the loss on a test set and prints both training and testing errors to monitor the progress.
    return loss: Returns the loss value, which is necessary for optimizers like L-BFGS that use closure functions.
    '''

    def closure(self):
        if self.optimizer is None:
            raise ValueError("Optimizer not set for the model.")
        self.optimizer.zero_grad()#Clears old gradients from the previous step (if any). This is necessary to avoid accumulation of gradients from multiple backward passes.
        loss = self.loss(self.X_train_Nu, self.Y_train_Nu, self.X_train_Nf)#Computes the total loss by calling the loss method of the model, which combines the boundary condition loss (lossBC) and the PDE loss (lossPDE). The inputs are: X_train_Nu: The input data points at the boundary. Y_train_Nu: The target values at the boundary. X_train_Nf: The input data points for evaluating the PDE.
        loss.backward()#Computes the gradients of the loss with respect to the model parameters. These gradients are used by the optimizer to update the model parameters.
        self.iter += 1
        if self.iter % 100 == 0:#Every 100 iterations, perform additional actions such as logging the training and testing errors.
            loss2 = self.lossBC(self.X_test, self.Y_test)#Computes the boundary condition loss on the test dataset
            print("Training Error:", loss.detach().cpu().numpy(), "---Testing Error:", loss2.detach().cpu().numpy())#Detaches the loss tensor from the computation graph and moves it to the CPU, converting it to a NumPy array for printing.         
        return loss
