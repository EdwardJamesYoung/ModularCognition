# This file creates dictionaries of hyperparameters for different training runs. 

# At some point this will be able to generate sets of hyperparameters 
# based on command line arguments. For now, it just returns a fixed
# set of hyperparameters.

# Hyperparameters are grouped into three dictionaries:
# (1) Task parameters, which describe the structure of the task
# (2) Network parameters, which describe the structure of the network
# (3) Training parameters, which describe the training procedure


# Define hyperparameters for the task
task_params = {
    "T_stim": 8, # Stimulus duration
    "T_comp": 6, # Computation duration
    "T_plan": 2, # Planning duration
    "T_resp": 8, # Response duration
    "dt_inv": 4, # Number of time steps per time unit
    "circular_input_dim": 36, # Number of input units around the circle
    "go_input_dim": 1, # Number of input units for go cue
    "task_input_dim": 2, # Number of input units for task
    "output_dim": 1, # Number of output units
    "kappa": 1, # Concentration parameter for the von Mises distribution
    "task_list": [(0,0,0), (1,0,0), (0,0,1), (1,0,1), (0,1,0), (1,1,0), (0,1,1), (1,1,1)] # List of tasks
}

# Define hyperparameters for the network
network_params = {
    "N": 256 # Number of recurrent units
}

# Define hyperparameters for training
training_params = {
    "lr": 0.001, # Learning rate
    "l1_reg": 1e-6, # l1 regularization parameter
    "response_weight": 100, # Weight of response loss
    "steps": 1000 # Number of training steps
}


def get_task_params():
    # This method returns the task parameters
    return task_params

def get_network_params():
    # This method returns the network parameters
    return network_params

def get_training_params():
    # This method returns the training parameters
    return training_params




