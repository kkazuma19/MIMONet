# %% [markdown]
# ## Test Case III: Heat Exchanger (2D Plane)

# %%
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import random
import matplotlib.pyplot as plt

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.append(src_path)
    
from utils import MIMONetDataset, DeepONetDataset, ChannelScaler
from mimonet_drop import MIMONet_Drop

# %%
# check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(seed)


# %%
# set working directory
working_dir = "/projects/bcnx/kazumak2/MIMONet/HeatExchanger"
data_dir = os.path.join(working_dir, "data")

# %% [markdown]
# ## load datasets

# %% [markdown]
# ### Load sharing parameters/dataset

# %%
# trunk dataset
trunk_input = np.load(os.path.join(data_dir, "share/trunk.npz"))['trunk']

# %%
# for post-processing
trunk_input_orig = trunk_input.copy()

# %%
# min-max scaling [-1, 1]
trunk_input[:, 0] = 2 * (trunk_input[:, 0] - np.min(trunk_input[:, 0])) / (np.max(trunk_input[:, 0]) - np.min(trunk_input[:, 0])) - 1
trunk_input[:, 1] = 2 * (trunk_input[:, 1] - np.min(trunk_input[:, 1])) / (np.max(trunk_input[:, 1]) - np.min(trunk_input[:, 1])) - 1

# %%
trunk_input.shape

# %% [markdown]
# ### Training data

# %%
# branch input dataset
branch = np.load(os.path.join(data_dir, "branch.npz"))

branch1 = branch['branch1']
branch2 = branch['branch2']

print("Branch1 shape:", branch1.shape)
print("Branch2 shape:", branch2.shape)

# split the dataset into training, cal, and testing sets
train_size = int(0.7 * len(branch1))
cal_size = int(0.1 * len(branch1))
test_size = len(branch1) - train_size - cal_size

#train_size = int(0.8 * len(branch1))
#test_size = len(branch1) - train_size
train_branch1, cal_branch1, test_branch1 = branch1[:train_size], branch1[train_size:train_size+cal_size], branch1[train_size+cal_size:]
train_branch2, cal_branch2, test_branch2 = branch2[:train_size], branch2[train_size:train_size+cal_size], branch2[train_size+cal_size:]

# %%
# create a dictionary for the output channel names
# 0: turb-kinetic-energy
# 1: pressure
# 2: temperature
# 3: z-velocity
# 4: y-velocity
# 5: x-velocity
# 6: velocity-magnitude

dict_channel = {
    0: 'turb-kinetic-energy',
    1: 'pressure',
    2: 'temperature',
    3: 'z-velocity',
    4: 'y-velocity',
    5: 'x-velocity',
    6: 'velocity-magnitude'
}

# select the output channel
target_channel = [1, 3, 4, 5, 6]

# print the selected output channel names
# target_label is used to store the names of the selected output channels for further processing (e.g., plotting)
print("Selected output channels:")
target_label = []
for channel in target_channel:
    print(dict_channel[channel])
    target_label.append(dict_channel[channel])    

# %%
target_channel

# %%
# target dataset
target = np.load(os.path.join(data_dir, "target.npy"))

print("Target shape:", target.shape)

## extract the output channels
## select the desired channels using the list (target_channel)
target = target[..., target_channel]


# split the target dataset into training and testing sets
train_target = target[:train_size]
cal_target = target[train_size:train_size+cal_size]
test_target = target[train_size+cal_size:]
#
print("Train target shape:", train_target.shape)
print("Cal target shape:", cal_target.shape)
print("Test target shape:", test_target.shape)

# %%
# (# train samples, 2) 
# get the mean and standard deviation of each channel
mean_branch1 = np.mean(train_branch1, axis=0)
std_branch1 = np.std(train_branch1, axis=0)

print("Mean of branch1:", mean_branch1)
print("Std of branch1:", std_branch1)

# (# train samples, 100)
mean_branch2 = np.mean(train_branch2)
std_branch2 = np.std(train_branch2)

print("Mean of branch2:", mean_branch2)
print("Std of branch2:", std_branch2)

# %%
# normalize the branch data using the mean and std
train_branch_1 = (train_branch1 - mean_branch1) / std_branch1
cal_branch_1 = (cal_branch1 - mean_branch1) / std_branch1
test_branch_1 = (test_branch1 - mean_branch1) / std_branch1
train_branch_2 = (train_branch2 - mean_branch2) / std_branch2
cal_branch_2 = (cal_branch2 - mean_branch2) / std_branch2
test_branch_2 = (test_branch2 - mean_branch2) / std_branch2

# print the shapes of the normalized data
print("Shape of normalized train_branch1:", train_branch_1.shape)
print("Shape of normalized cal_branch1:", cal_branch_1.shape)
print("Shape of normalized test_branch1:", test_branch_1.shape)
print("Shape of normalized train_branch2:", train_branch_2.shape)
print("Shape of normalized cal_branch2:", cal_branch_2.shape)
print("Shape of normalized test_branch2:", test_branch_2.shape)

# %% [markdown]
# ### Scaling the target data

# %%
# scaling the target data
'''  
note: reverse the scaling for the target data
train_target = scaler.inverse_transform(train_target_scaled)
test_target = scaler.inverse_transform(test_target_scaled)
'''
scaler = ChannelScaler(method='minmax', feature_range=(-1, 1))
scaler.fit(train_target)
train_target_scaled = scaler.transform(train_target)
cal_target_scaled = scaler.transform(cal_target)
test_target_scaled = scaler.transform(test_target)

print("Shape of scaled train_target:", train_target_scaled.shape)
print("Shape of scaled cal_target:", cal_target_scaled.shape)
print("Shape of scaled test_target:", test_target_scaled.shape)

# %% [markdown]
# ## Torch Dataset and DataLoader

# %%
# test dataset and dataloader
test_dataset = MIMONetDataset(
    [test_branch_1, test_branch_2],  # branch_data_list
    trunk_input,                     # trunk_data
    test_target_scaled               # target_data
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,  # set to 1 for testing
    shuffle=False,
    num_workers=0
)

# %%
train_dataset = MIMONetDataset(
    [train_branch_1, train_branch_2],  # branch_data_list
    trunk_input,                       # trunk_data
    train_target_scaled                # target_data
)

# %% [markdown]
# ## MIMONet Model

# %%
# Architecture parameters
dim = 256
branch_input_dim1 = 2
branch_input_dim2 = 100
trunk_input_dim = 2

# Define the model arguments for orig_MIMONet
model_args = {
    'branch_arch_list': [
        [branch_input_dim1, 512, 512, 512, dim],
        [branch_input_dim2, 512, 512, 512, dim]
    ],
    'trunk_arch': [trunk_input_dim, 256, 256, 256, dim],
    'num_outputs': target.shape[-1] -1,  # number of output channels
    'activation_fn': nn.ReLU,
    'merge_type': 'mul',
    'dropout_p': 0.1,  # dropout probability
}

# %%
from training import train_model

# %%
# scheduler parameters
scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau
scheduler_args={'mode': 'min', 'factor': 0.5, 'patience': 10,}

# %%
def velocity_magnitude_loss(pred, target):
    # pred: (B, N, 3) → u, v, w
    # target: (B, N, 4) → u, v, w, |u|

    pred_u = pred[..., 1]
    pred_v = pred[..., 2]
    pred_w = pred[..., 3]

    pred_mag = torch.sqrt(pred_u**2 + pred_v**2 + pred_w**2 + 1e-8)  # avoid sqrt(0)
    true_mag = target[..., 4]

    return torch.mean((pred_mag - true_mag)**2)

class MSEWithVelocityMagnitudeLoss(nn.Module):
    def __init__(self, lambda_mag=0.1):
        super().__init__()
        self.lambda_mag = lambda_mag
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # pred: [B, N, 5] → [p, w, v, u, |u|]
        # target: [B, N, 5] → same order

        loss_data = self.mse(pred[..., :4], target[..., :4])  # match p, w, v, u
        loss_mag = velocity_magnitude_loss(pred, target)

        return loss_data + self.lambda_mag * loss_mag


# %%
criterion = MSEWithVelocityMagnitudeLoss(lambda_mag=0.0001)
#criterion = MSEWithVelocityMagnitudeLoss(lambda_mag=0)
# %%
def run():
    train_model(
        model_fn=MIMONet_Drop,
        model_args=model_args,
        optimizer_fn=torch.optim.Adam,
        optimizer_args={'lr': 1e-3, 'weight_decay': 1e-6},
        scheduler_fn=scheduler_fn,
        scheduler_args=scheduler_args,
        dataset=train_dataset,
        device=device,
        num_epochs=100,
        batch_size=4,
        #criterion=nn.MSELoss(),
        criterion=criterion,
        patience=500,
        k_fold=None,
        multi_gpu=False,
        working_dir="/projects/bcnx/kazumak2/MIMONet/HeatExchanger/",
    )


run()

print("Training completed.")




