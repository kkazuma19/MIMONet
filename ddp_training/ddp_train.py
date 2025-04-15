import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np

# 1. Dummy Dataset
class SyntheticDataset(Dataset):
    def __init__(self, n_samples=10000):
        x = np.linspace(-2, 2, n_samples).reshape(-1, 1)
        y = np.sin(3 * x) + 0.3 * np.random.randn(*x.shape)
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 2. Simple NN Model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# 3. Training function for each process
def train(rank, world_size):
    print(f"[Rank {rank}] Starting process...")

    # Setup
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Data
    dataset = SyntheticDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # Model
    model = SimpleNet().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.01)

    # Training
    for epoch in range(20):
        ddp_model.train()
        sampler.set_epoch(epoch)
        running_loss = 0.0
        for x, y in dataloader:
            x = x.to(rank)
            y = y.to(rank)

            optimizer.zero_grad()
            outputs = ddp_model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[Rank {rank}] Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.4f}")

    dist.destroy_process_group()

# 4. Main script
def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    train(rank, world_size)

if __name__ == '__main__':
    main()
