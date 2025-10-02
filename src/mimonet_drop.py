import torch
import torch.nn as nn

class FCN(nn.Module):
    """ 
    Fully Connected Neural Network (FCNN) class with optional dropout.
    """
    def __init__(self, architecture, activation_fn=nn.ReLU, dropout_p=0.0):
        super(FCN, self).__init__()
        layers = []
        for i in range(len(architecture) - 1):
            in_dim, out_dim = architecture[i], architecture[i+1]

            # Final layer: no activation, no dropout
            if i == len(architecture) - 2:
                layers.append(nn.Linear(in_dim, out_dim, bias=True))
            else:
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(activation_fn())
                if dropout_p > 0:
                    layers.append(nn.Dropout(p=dropout_p))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class MIMONet_Drop(nn.Module):
    def __init__(self, branch_arch_list, trunk_arch, num_outputs=1,
                 activation_fn=nn.ReLU, merge_type='mul', dropout_p=0.0):
        super(MIMONet_Drop, self).__init__()

        self.merge_type = merge_type
        self.num_outputs = num_outputs
        self.hidden_dim = branch_arch_list[0][-1]

        # Branch networks with dropout
        self.branch_nets = nn.ModuleList([
            FCN(arch, activation_fn, dropout_p=dropout_p) for arch in branch_arch_list
        ])

        # Trunk network with dropout
        trunk_arch[-1] = self.hidden_dim * self.num_outputs
        self.trunk_net = FCN(trunk_arch, activation_fn, dropout_p=dropout_p)

        self.bias = nn.Parameter(torch.zeros(1, 1, num_outputs))

