# This file contains helper functions for visualising the results of experiments

# Import the necessary packages
import numpy as np
import matplotlib.pyplot as plt

import Hyperparameters as hp
import Tasks

# Get the hyperparameters
task_params = hp.get_task_params()

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

T_tot = T_stim + T_comp + T_plan + T_resp # Total trial duration
dt = 1/dt_inv # Time step

def plot_loss_curve(losses):
    # This function plots the loss curve

    # First ensure that the losses are in the appropriate format
    losses = np.array(losses)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(losses)
    ax.set_title("Loss curve")
    ax.set_xlabel("Episode number")
    ax.set_ylabel("Loss")
    ax.set_xlim([0,len(losses)])
    ax.set_ylim([0,max(losses)])

    return fig, ax

def plot_smoothed_loss_curve(losses, window_size = 100):
    # This function plots the smoothed loss curve

    # First ensure that the losses are in the appropriate format
    losses = np.array(losses)

    # Compute the smoothed loss curve
    smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode = 'valid')

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(smoothed_losses)
    ax.set_title("Smoothed loss curve")
    ax.set_xlabel("Episode number")
    ax.set_ylabel("Loss")
    ax.set_xlim([0,len(losses)-100])
    ax.set_ylim([0,max(losses)])

    return fig, ax

def plot_sample_trial(task):
    X, Y_tar = Tasks.task(task_description=task, batch_size = 1)
    
    fig, axs = plt.subplots(1, 2, figsize=(20,5))

    axs[0].imshow(X[0,:,:circular_input_dim].T, aspect = 'auto')
    axs[0].set_xlabel('Time step')
    axs[0].set_ylabel('Input unit')
    axs[0].set_title('Input sequence')

    axs[1].plot(Y_tar[0,:,:])
    axs[1].set_xlabel('Time step')
    axs[1].set_ylim([-1.1,1.1])
    axs[1].set_ylabel('Output unit')
    axs[1].set_title('Target output sequence')

    return fig, axs

def discrepancy_visualisation(task, model):
    assert task[2] == 0, "This function only works for tasks with discrepancy responses."
    X, Y_tar = Tasks.task(task, batch_size = 250)
    # Compute the output of the network
    Y = model.forward(X)
    # Compute the average response during the response phase
    Y_avg = Y[:,(T_stim+T_comp+T_resp)*dt_inv:,:].mean(dim = 1)
    # Compute the target response during the response phase
    Y_tar = Y_tar[:,(T_stim+T_comp+T_resp)*dt_inv:,:].mean(dim = 1)
    # Plot the scatter plot
    fig, ax = plt.subplots(figsize = (8,8))
    ax.scatter(Y_avg.detach().numpy(), Y_tar.detach().numpy())
    ax.set_xlabel('Average response', fontsize = 18)
    ax.set_ylabel('Target response', fontsize = 18)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    # Add in the diagonal line
    ax.plot([-1,1],[-1,1], 'r')

    return fig, ax
    
def cosine_visualisation(task, model):
    assert task[2] == 1, "This function only works for tasks with cosine responses."
    batch_size = 8
    X, Y_tar = Tasks.task(task, batch_size = batch_size)
    # Compute the output of the network
    Y = model.forward(X)

    # Plot samples of the target and actual output of the network over the planning and response period
    t = np.linspace((T_stim+T_comp), T_tot, (T_resp+T_plan)*dt_inv)
    fig, axs = plt.subplots(batch_size, 1, figsize=(10,5*batch_size))
    for k in range(batch_size):
        axs[k].plot(t, Y_tar[k,(T_stim+T_comp)*dt_inv:,:].detach().numpy(), label = 'Target')
        axs[k].plot(t, Y[k,(T_stim+T_comp)*dt_inv:,:].detach().numpy(), label = 'Actual')
        axs[k].set_xlabel('Time')
        axs[k].set_ylabel('Output')
        axs[k].legend()
        axs[k].set_xlim([(T_stim+T_comp), T_tot])
        axs[k].set_ylim([-1.1,1.1])
        # Add in the vertical lines at T_plan
        axs[k].plot([T_plan,T_plan],[-1.1,1.1], 'k--')

    return fig, axs