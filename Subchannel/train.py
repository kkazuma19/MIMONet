# %%
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import random
import matplotlib.pyplot as plt
#from scripts.training import train_model

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.append(src_path)

# importing custom modules
from utils import MIMONetDataset, DeepONetDataset, ChannelScaler
from mimonet import MIMONet

from training import train_model

# %%
# check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# %%
# set working directory
working_dir = "/projects/bcnx/kazumak2/MIMONet/Subchannel/"
data_dir = os.path.join(working_dir, "data")

# %% [markdown]
# ## load datasets

# %% [markdown]
# ### Load sharing parameters/dataset

# %%
# trunk dataset
trunk_input = np.load(os.path.join(data_dir, "share/trunk_input.npz"))['trunk']

# %% [markdown]
# ### Training data

# %%
# training data
train_branch = np.load(os.path.join(data_dir, "training/train_branch_input.npz"))
train_branch_1 = train_branch['func_params']
train_branch_2 = train_branch['stat_params']

# [samples, channel, gridpoints]
train_target = np.load(os.path.join(data_dir, "training/train_target.npz"))['target']
# convert to [samples, gridpoints, channel]
train_target = np.moveaxis(train_target, 1, 2)

print("train_branch_1 shape:", train_branch_1.shape)
print("train_branch_2 shape:", train_branch_2.shape)
print("train_target shape:", train_target.shape)

# %%
# scaling the functional input data using predefined mean and std
f_mean = np.load(os.path.join(data_dir, "share/func_mean_std_params.npz"))['mean']
f_std = np.load(os.path.join(data_dir, "share/func_mean_std_params.npz"))['std']

train_branch_1 = (train_branch_1 - f_mean) / f_std

# scaling the static input data using predefined mean and std
s_mean = np.load(os.path.join(data_dir, "share/stat_mean_std_params.npz"))['mean']
s_std = np.load(os.path.join(data_dir, "share/stat_mean_std_params.npz"))['std']

for i in range(s_mean.shape[0]):
    train_branch_2[:, i] = (train_branch_2[:, i] - s_mean[i]) / s_std[i]

# %% [markdown]
# ### Test data

# %%
test_branch = np.load(os.path.join(data_dir, "test/test_branch_input.npz"))
test_branch_1 = test_branch['func_params']
test_branch_2 = test_branch['stat_params']

test_target = np.load(os.path.join(data_dir, "test/test_target.npz"))['target']
test_target = np.moveaxis(test_target, 1, 2)

print("test_branch_1 shape:", test_branch_1.shape)
print("test_branch_2 shape:", test_branch_2.shape)
print("test_target shape:", test_target.shape)

# scaling the functional input data using predefined mean and std
test_branch_1 = (test_branch_1 - f_mean) / f_std
# scaling the static input data using predefined mean and std
for i in range(s_mean.shape[0]):
    test_branch_2[:, i] = (test_branch_2[:, i] - s_mean[i]) / s_std[i]

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
test_target_scaled = scaler.transform(test_target)


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
branch_input_dim1 = 100
branch_input_dim2 = 2
trunk_input_dim = 2

# Define MIONet instance (no Fourier, no final linear)
model = MIMONet(
    branch_arch_list=[
        [branch_input_dim1, 512, 512, 512, dim],
        [branch_input_dim2, 512, 512, 512, dim]
    ],
    trunk_arch=[trunk_input_dim, 256, 256, dim],
    num_outputs=3, 
    activation_fn=nn.ReLU,
    merge_type='mul'  # or 'sum'
)

model = model.to(device)

# Print parameter count
num_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {num_params:,}")

# %%
#from scripts.training import train_model

# %%
train_model(
    model=model,
    dataset=train_dataset,
    device='cuda',
    num_epochs=2000,
    batch_size=8,
    lr=1e-3,
    patience=20,
    multi_gpu=False,
    working_dir=""
)

print("Training completed.")


