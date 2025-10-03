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
        
    def forward(self, branch_inputs, trunk_input):
        """
        Forward pass.

        Args:
            branch_inputs (list[Tensor]): Each tensor is (batch_size, input_dim)
            trunk_input (Tensor): (batch_size, num_trunk_points, input_dim)

        Returns:
            Tensor: (batch_size, num_trunk_points, num_outputs)
        """
        # Branch processing: each output is (batch, hidden_dim)
        branch_outputs = [net(inp) for net, inp in zip(self.branch_nets, branch_inputs)]

        # Merge branches
        if self.merge_type == 'sum':
            combined_branch = sum(branch_outputs)
        elif self.merge_type == 'mul':
            combined_branch = branch_outputs[0]
            for b in branch_outputs[1:]:
                combined_branch = combined_branch * b
        else:
            raise ValueError(f"Unsupported merge type: {self.merge_type}")

        # Trunk processing
        trunk_out = self.trunk_net(trunk_input)  # (batch, num_trunk_points, hidden_dim * num_outputs)
        B, P = trunk_out.shape[0], trunk_out.shape[1]
        trunk_out = trunk_out.view(B, P, self.hidden_dim, self.num_outputs)  # (B, P, I, O)

        # Einsum: (B, I) × (B, P, I, O) → (B, P, O)
        output = torch.einsum('bi,bpio->bpo', combined_branch, trunk_out)
        return output + self.bias  # final shape: (B, P, O)

