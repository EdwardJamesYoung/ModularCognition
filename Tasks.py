import numpy as np
import torch

import Hyperparameters as hp

# Get the task parameters 
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

# Compute additional values from the task parameters
T_tot = T_stim + T_comp + T_plan + T_resp # Total trial duration
input_dim = circular_input_dim + go_input_dim + task_input_dim # Total input dimension
dt = 1/dt_inv # Time step
T_steps = dt_inv*T_tot # Number of time steps per 
sample_orientations = torch.linspace(0, 2*np.pi, circular_input_dim + 1)[:-1]


# This task function takes in:
# (1) A sub-task descriptor (a tuple of length 3)
# (2) The batch size (number of trials to be simulated) 
# This function returns:
# (1) The input sequence to the network, which has size (batch_size, T_steps, input_dim) 
# (2) The target output of the network, which has size (batch_size, T_steps, output_dim)

def task(task_description, batch_size = 64):
    subtask1, subtask2, subtask3 = task_description
    # Define the input sequence
    X = torch.zeros(batch_size, T_steps, input_dim)
    # The go cue comes on at the start of the response period
    # The go cue is the input dimension after the circular input dimensions
    X[:,(T_stim+T_comp+T_plan)*dt_inv:,circular_input_dim] = 1
    # The task cue is the input dimension after the circular input dimensions and the go cue dimensions
    # The task cue comes on during the computation period
    X[:,T_stim*dt_inv:(T_stim + T_comp)*dt_inv,circular_input_dim+go_input_dim+subtask2] = 1

    # Define the target output sequence
    Y_tar = torch.zeros(batch_size, T_steps, output_dim)

    # First sample (i) the two stimuli (ii) the target direction (iii) the novel stimulus direction
    theta1 = torch.zeros(batch_size, 1)
    theta2 = torch.zeros(batch_size, 1)
    theta_tar = torch.zeros(batch_size, 1)
    theta_nov = torch.zeros(batch_size, 1)
    A1 = 0
    A2 = 0
    # Sampling depends on the second subtask (cognition)
    if subtask2 == 0:
        # This cognition subtask is get_max.
        # In this case, the two amplitudes are sampled from A_1 ~ U[0.4, 1] and A_2 ~ U[0.2, 0.8*A_1]
        A1 = torch.rand(batch_size, 1)*(1-0.4) + 0.4
        A2 = torch.rand(batch_size, 1)*(0.8*A1 - 0.2) + 0.2
        # The angles are sampled such that the minimum difference between them is \pi/4
        # This is done by first sampling \theta_1 uniformly from [0, 2\pi]
        # and then sampling \theta_2 uniformly from [\theta_1 + \pi/4, \theta_1 + 7\pi/4]
        theta1 = torch.rand(batch_size, 1)*2*np.pi
        theta2 = theta1 + torch.rand(batch_size, 1)*(3*np.pi/2) + np.pi/4 
        theta2 = torch.remainder(theta2, 2*np.pi)

        # Set target output direction - in this case, the direction with the larger amplitude
        theta_tar = theta1 
    elif subtask2 == 1:
        # The cognition subtask is weighted_average.
        # In this case, the two amplitudes are sampled uniformly from [0.2,1]
        A1 = torch.rand(batch_size, 1)*0.8 + 0.2
        A2 = torch.rand(batch_size, 1)*0.8 + 0.2
        # This is done by first sampling \theta_1 uniformly from [0, 2\pi]
        # and then sampling \theta_2 uniformly from [\theta_1 + \pi/4, \theta_1 + 7\pi/4]
        theta1 = torch.rand(batch_size, 1)*2*np.pi
        theta2 = theta1 + torch.rand(batch_size, 1)*(3*np.pi/2) + np.pi/4
        theta2 = torch.remainder(theta2, 2*np.pi)

        # Set the target output direction - in this case, the weighted average of the two directions
        real_part = A1 * torch.cos(theta1) + A2 * torch.cos(theta2)
        imag_part = A1 * torch.sin(theta1) + A2 * torch.sin(theta2)

        # Compute the argument
        theta_tar = torch.atan2(imag_part, real_part)
    elif subtask2 == 2:
        # The cognition subtask is clustering
        # In this case, the two amplitudes are sampled uniformly from [0.2,1]
        A1 = torch.rand(batch_size, 1)*0.8 + 0.2
        A2 = torch.rand(batch_size, 1)*0.8 + 0.2
        # This is done by first sampling \theta_1 uniformly from [0, 2\pi]
        # and then sampling \theta_2 uniformly from [\theta_1 + \pi/4, \theta_1 + 7\pi/4]
        theta1 = torch.rand(batch_size, 1)*2*np.pi
        theta2 = theta1 + torch.rand(batch_size, 1)*(3*np.pi/2) + np.pi/4
        theta2 = torch.remainder(theta2, 2*np.pi)

        # Set the target output - in this case, 0 if the directions are within \pi/2 of each other and 1 otherwise
        # Compute the absolute difference between the angles
        diff = torch.abs(theta1 - theta2)
        # Adjust for the circular nature of the angles
        diff = torch.min(diff, 2*np.pi - diff)

        # Create a tensor that contains 1s when the angles are within pi/4 of each other, and 0 otherwise
        theta_tar = torch.zeros(batch_size, 1)
        theta_tar[diff < np.pi/4] = 1
    elif subtask2 == 3:
        # The cognition subtask is thresholding
        # In this case, the two amplitudes are sampled uniformly from [0.2,1]
        A1 = torch.rand(batch_size, 1)*0.8 + 0.2
        A2 = torch.rand(batch_size, 1)*0.8 + 0.2
        # This is done by first sampling \theta_1 uniformly from [0, 2\pi]
        # and then sampling \theta_2 uniformly from [\theta_1 + \pi/4, \theta_1 + 7\pi/4]
        theta1 = torch.rand(batch_size, 1)*2*np.pi
        theta2 = theta1 + torch.rand(batch_size, 1)*(3*np.pi/2) + np.pi/4

        # Set the target output - in this case, 0 if the sum of the amplitude is less than 1.2 and 1 otherwise
        theta_tar = torch.zeros(batch_size, 1)
        theta_tar[A1 + A2 > 1.2] = 1

    
    # Define the input sequence for the first subtask (stimulus)
    if subtask1 == 0:
        # This stimulus subtask is integral load in.
        # In this case, the input sequence is the sum of two oscillating von Mises functions
        
        # Sample the phases of the oscillations
        phi1 = torch.rand(batch_size, 1)*2*np.pi
        phi2 = torch.rand(batch_size, 1)*2*np.pi

        # Sample the angular frequencies of the oscillations.
        # There are between 1 and 4 oscillations during the stimulus period.
        # This enforces that the angular frequencies are between 1 and 4 times (2\pi/T_stim) 
        omega1 = (torch.rand(batch_size, 1)*(4-1) + 1)*2*np.pi/T_stim
        omega2 = (torch.rand(batch_size, 1)*(4-1) + 1)*2*np.pi/T_stim
 
        # Construct the two bumps
        bump1 = A1[:,:,None]*torch.exp(torch.distributions.von_mises.VonMises(0, kappa).log_prob(sample_orientations[None,:]-theta1[:,None]))
        bump2 = A2[:,:,None]*torch.exp(torch.distributions.von_mises.VonMises(0, kappa).log_prob(sample_orientations[None,:]-theta2[:,None]))
        
        # Construct the temporal modulations
        t = torch.linspace(0, T_stim, T_stim*dt_inv + 1)[1:]
        t1 = omega1*t + phi1
        t2 = omega2*t + phi2
        t1 = t1[:,:,None]
        t2 = t2[:,:,None]
        modulation1 = torch.cos(t1)
        modulation2 = torch.cos(t2)

        # Construct the input sequence
        X[:,0:T_stim*dt_inv,0:circular_input_dim] = modulation1*bump1 + modulation2*bump2
    elif subtask1 == 1:
        # The stimulus task is the sequential load in.
        # In this case the two stimuli appear one after the other
        # Each stimulus occurs for half of the stimulus period
        
        # We randomise which stimulus appears first
        first_stim = torch.randint(0,2,(batch_size,1))
        # Iterate through the batch
        for epi in range(batch_size):
            if first_stim[epi] == 0:
                # Show (A1, theta1) first and then (A2, theta2)
                X[epi,0:int(T_stim*dt_inv/2),0:circular_input_dim] = A1[epi]*torch.exp(torch.distributions.von_mises.VonMises(0, kappa).log_prob(sample_orientations[None,:]-theta1[epi,None]))
                X[epi,int(T_stim*dt_inv/2):T_stim*dt_inv,0:circular_input_dim] = A2[epi]*torch.exp(torch.distributions.von_mises.VonMises(0, kappa).log_prob(sample_orientations[None,:]-theta2[epi,None]))
            else:
                # Show (A2, theta2) first and then (A1, theta1)
                X[epi,0:int(T_stim*dt_inv/2),0:circular_input_dim] = A2[epi]*torch.exp(torch.distributions.von_mises.VonMises(0, kappa).log_prob(sample_orientations[None,:]-theta2[epi,None]))
                X[epi,int(T_stim*dt_inv/2):T_stim*dt_inv,0:circular_input_dim] = A1[epi]*torch.exp(torch.distributions.von_mises.VonMises(0, kappa).log_prob(sample_orientations[None,:]-theta1[epi,None]))

    # Finally, set the correct output for the response phase
    if subtask3 == 0:
        # This response subtask is a discrepency estimation task. 
        # In this case, the output is the difference between the target and the novel stimulus directions
        # The output of the network is in [-1, 1], so we scale the difference to be in [-1, 1]
        
        # Set novel stimulus direction - in this case, uniformly sampled from [\theta_1 - 7\pi/8, \theta_1 + 7\pi/8]
        theta_nov = torch.rand(batch_size, 1)*(7*np.pi/4) + theta1 - 7*np.pi/8
        # Wrap the angles to be between 0 and 2\pi
        theta_nov = torch.remainder(theta_nov, 2*np.pi)
        # Set the input for the response and planning phases to be a von Mises bump centred at the novel stimulus direction
        X[:,(T_stim+T_comp)*dt_inv:,0:circular_input_dim] = torch.exp(torch.distributions.von_mises.VonMises(0, kappa).log_prob(sample_orientations[None,:]-theta_nov[:,None]))

        diff = torch.remainder(theta_tar - theta_nov, 2*np.pi)
        diff[diff > np.pi] = diff[diff > np.pi] - 2*np.pi
        Y_tar[:,(T_stim+T_comp+T_plan)*dt_inv:,0] = diff/np.pi
    elif subtask3 == 1:
        # This response subtask is a timing response task
        # A bump moves around the circle at a constant speed theta(t) 
        # The network must respond by creating a cosine output cos(theta(t) - theta_tar)

        # The angular velocity is sampled uniformly from [1,2] times (2\pi/T_resp)
        omega_nov = (torch.rand(batch_size)*(2-1) + 1)*2*np.pi/T_resp # omega has shape (batch_size, 1)
        # Randomly vary the sign of omega
        omega_nov = omega_nov*torch.sign(torch.rand(batch_size) - 0.5)
        # The initial angle is sampled uniformly from [0, 2\pi]
        phi_nov = torch.rand(batch_size)*2*np.pi # phi has shape (batch_size, 1)
        # The angular position is given by theta(t) = omega*t + phi
        t = torch.linspace(0, T_plan + T_resp, (T_plan+T_resp)*dt_inv + 1)[1:] # t has shape ((T_plan+T_resp)*dt_inv,)
        theta_nov = omega_nov[:,None]*t[None,:] + phi_nov[:,None] # theta_nov has shape (batch_size, (T_plan+T_resp)*dt_inv)
        # Wrap the angles to be between 0 and 2\pi
        theta_nov = torch.remainder(theta_nov, 2*np.pi)
        # Set the input for the response and planning phases to be a von Mises bump centred at the novel stimulus direction
        X[:,(T_stim+T_comp)*dt_inv:,0:circular_input_dim] = torch.exp(torch.distributions.von_mises.VonMises(0, kappa).log_prob(sample_orientations[None,None,:]-theta_nov[:,:,None]))
        # Set the target output to be the cosine of the difference between the novel stimulus direction and the target direction
        # First runcate theta_nov to be only over the response period
        theta_nov = theta_nov[:,-T_resp*dt_inv:] # theta_nov has shape (batch_size, T_resp*dt_inv)
        Y_tar[:,(T_stim+T_comp+T_plan)*dt_inv:,0] = torch.cos(theta_nov - theta_tar)

    return X, Y_tar