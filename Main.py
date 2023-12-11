# This file creates the networks and runs the training procedure

# Import the necessary packages
import numpy as np
import torch.nn as nn  
import torch.nn.functional as F
import torch.optim as optim
import torch
import datetime
import pickle
import tqdm
import os 

import Hyperparameters as hp
import Tasks
import Visualisation_helpers

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

# Get the hyperparameters
task_params = hp.get_task_params()
network_params = hp.get_network_params()
training_params = hp.get_training_params()

# Unload the task parameters
T_stim = task_params["T_stim"] # Stimulus duration
T_comp = task_params["T_comp"] # Computation duration
T_plan = task_params["T_plan"] # Planning duration
T_resp = task_params["T_resp"] # Response duration
dt_inv = task_params["dt_inv"] # Number of time steps per time unit
circular_input_dim = task_params["circular_input_dim"] # Number of input units around the circle
go_input_dim = task_params["go_input_dim"] # Number of input units for go cue
task_input_dim = task_params["task_input_dim"] # Number of input units for task
output_dim = task_params["output_dim"] # Number of output units
kappa = task_params["kappa"] # Concentration parameter for the von Mises distribution
task_list = task_params["task_list"] # List of tasks

# Compute additional values from the task parameters
T_tot = T_stim + T_comp + T_plan + T_resp # Total trial duration
input_dim = circular_input_dim + go_input_dim + task_input_dim # Total input dimension
dt = 1/dt_inv # Time step
T_steps = dt_inv*T_tot # Number of time steps per 

# Unload the network parameters
N = network_params["N"] # Number of recurrent units

# Unload the training parameters
l1_reg = training_params["l1_reg"] # l1 regularization parameter
lr = training_params["lr"] # Learning rate
response_weight = training_params["response_weight"] # Weight of response loss
steps = training_params["steps"] # Number of training steps

# Set up the network architecture
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.Layers = nn.ModuleDict({
            'input': nn.Linear(input_dim, N, bias = False, device=DEVICE),
            'recurrent': nn.Linear(N, N, bias = True, device=DEVICE),
            'output': nn.Linear(N, output_dim, bias = True, device=DEVICE)
        })

    def forward_one_step(self, x, h):
        # This method takes in the input x of size (batch_size, input_dim) for one time step
        # and the activities h of size (batch_size, N) from the previous time step.
        # The method returns the activies at the current time step, h_new of size (batch_size, N). 
        # and the output at the current time step, y of size (batch_size, output_dim).

        # Compute the activities at the current time step
        h_new = (1 - dt)*h + dt*F.relu(self.Layers['input'](x) + self.Layers['recurrent'](h))
        # Compute the output at the current time step
        y = self.Layers['output'](h_new) 
        return y, h_new
    
    def forward(self, X):
        # This method takes in a time series of inputs X of size (batch_size, T_steps, input_dim)
        # and returns the corresponding time series of outputs Y of size (batch_size, T_steps, output_dim)

        # Initialize the activities h of size (batch_size, N) to zero
        h = torch.zeros(X.shape[0], N, device=DEVICE)
        # Initialize the outputs Y of size (batch_size, T_steps, output_dim) to zero
        Y = torch.zeros(X.shape[0], X.shape[1], output_dim, device=DEVICE)
        # Loop over the time steps
        for t in range(X.shape[1]):
            # Compute the output and activities at the current time step
            y, h = self.forward_one_step(X[:,t,:], h)
            # Store the output at the current time step
            Y[:,t,:] = y
        return Y

def train_network(model, optimiser, loss_function):
    # This method trains the network
    # It takes in the network model, the optimiser, and the loss function
    # It returns the loss curve

    # Initialize the loss curve
    losses = []
    # Loop over the training steps
    for step in tqdm.tqdm(range(steps)):
        # Train the network for one episode
        loss = train_one_episode(model, optimiser, loss_function)
        # Store the loss
        losses.append(loss.detach().numpy())
    return losses

def train_one_episode(model, optimiser, loss_function):
    # This method trains the network for one episode (one batch of trials)
    # It takes in the network model, the optimiser, and the loss function
    # It returns the loss for the episode

    # Set the gradients to zero
    optimiser.zero_grad()

    # Generate the input sequence and target output for the task
    # Train on a batch consisting of each subtask
    subtask_batch = int(128/len(task_list))
    X_list = []
    Y_tar_list = []
    for task in task_list:
        X, Y_tar = Tasks.task(task, batch_size = subtask_batch)
        X_list.append(X)
        Y_tar_list.append(Y_tar)
    
    X = torch.cat(X_list, dim=0) 
    X = X.to(DEVICE)
    Y_tar = torch.cat(Y_tar_list, dim=0)
    Y_tar = Y_tar.to(DEVICE)
    # Compute the output of the network
    Y = model.forward(X)

    # Compute the loss
    loss_pre = loss_function(Y[:,0:(T_stim+T_comp+T_plan)*dt_inv,:], Y_tar[:,0:(T_stim+T_comp+T_plan)*dt_inv,:])
    loss_response = loss_function(Y[:,(T_stim+T_comp+T_plan)*dt_inv:,:], Y_tar[:,(T_stim+T_comp+T_plan)*dt_inv:,:])
    loss = loss_pre + response_weight*loss_response
    # Compute the l1 penalty
    l1_penalty = model.Layers['recurrent'].weight.abs().sum() + model.Layers['output'].weight.abs().sum() + model.Layers["input"].weight.abs().sum()
    # Compute the total loss
    total_loss = loss + l1_reg*l1_penalty
    # Perform gradient descent
    total_loss.backward()
    # clip the gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    # Perform one step of gradient descent
    optimiser.step()
    return loss

def save_training_run(model, losses):
    # This method saves: 
    # (1) Hyperparameters
    # (2) Network weights 
    # (3) Training loss curves and visualisation
    # (4) Visualisations of performance on each trained subtask
    # These are saved to a new directory within the directory "Training_runs"
    # The name of the directory is the current date and time
    
    # Create the directory if it doesn't exist
    directory = "Training_runs/" + str(datetime.datetime.now().strftime("%m%d_%H%M"))
    if not os.path.exists(directory):
        os.makedirs(directory)

    # The hyperparameters are pickled 
    with open(directory + "/task_params.pkl", "wb") as f:
        pickle.dump(task_params, f)
    with open(directory + "/network_params.pkl", "wb") as f:
        pickle.dump(network_params, f)
    with open(directory + "/training_params.pkl", "wb") as f:
        pickle.dump(training_params, f)

    # The model weights are saved using pytorch's save method
    torch.save(model.state_dict(), directory + "/model_weights.pt")

    # The loss curve is saved as (1) a numpy array and (2) a plot (3) a smoothed plot
    np.save(directory + "/losses.npy", np.array(losses))
    fig, ax = Visualisation_helpers.plot_loss_curve(losses)
    fig.savefig(directory + "/loss_curve.pdf", bbox_inches='tight')
    fig, ax = Visualisation_helpers.plot_smoothed_loss_curve(losses)
    fig.savefig(directory + "/smoothed_loss_curve.pdf", bbox_inches='tight')

    # The performance on each subtask is saved as a plot
    for task in task_list:
        if task[2] == 0:
            fig, ax = Visualisation_helpers.discrepancy_visualisation(task, model)
            fig.savefig(directory + "/task_" + str(task) + ".pdf", bbox_inches='tight')
        elif task[2] == 1:
            fig, ax = Visualisation_helpers.cosine_visualisation(task, model)
            fig.savefig(directory + "/task_" + str(task) + ".pdf", bbox_inches='tight')


if __name__ == "__main__":
    # Create the network
    model = Net()
    # Move the network to the GPU if available
    model.to(DEVICE)
    # Create the optimiser
    optimiser = optim.Adam(model.parameters(), lr=lr)
    # Create the loss function
    loss_function = nn.MSELoss()

    # Train the network
    losses = train_network(model, optimiser, loss_function)

    # Save the training run
    save_training_run(model, losses)

