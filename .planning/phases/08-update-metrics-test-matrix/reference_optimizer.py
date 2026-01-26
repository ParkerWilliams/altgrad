"""Reference implementation: MasterOptim with grid-based (rung) updates.

This is the user-provided sample code that demonstrates successful FP8 training
with explicit grid/rung-based optimization. Key patterns to incorporate:

1. Master weights in FP32, model weights in FP8
2. Build explicit grid from FP8 representable values
3. Stochastic rounding: floor(v_rungs + rand)
4. Rung clipping: clamp(v_rungs, -10, 10) to avoid NaN cliffs
5. Flip counting: (p.data != old_data).sum()

Source: User-provided CIFAR10 example (2026-01-26)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float8_e4m3fn

# --- 1. OPTIMIZERS ---

class MasterOptim:
    def __init__(self, params, val, momentum=0.9, weight_decay=1e-4, mode='euclidean'):
        self.params = list(params)
        self.val, self.momentum, self.wd = val, momentum, weight_decay
        self.mode = mode
        self.master_p = [p.detach().clone().float().to(DEVICE) for p in self.params]
        self.velocity = [torch.zeros_like(p).float().to(DEVICE) for p in self.master_p]

        # Build grid
        raw_bits = torch.arange(-128, 128, dtype=torch.int8)
        all_floats = raw_bits.view(DTYPE).to(torch.float32)
        clean_grid = all_floats[~torch.isnan(all_floats) & ~torch.isinf(all_floats)]
        self.grid = torch.sort(torch.unique(clean_grid))[0].to(DEVICE)

    @torch.no_grad()
    def step(self, current_scale=None):
        flips = 0
        scale = current_scale if current_scale is not None else self.val
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            old_data = p.data.clone()
            grad = p.grad.to(torch.float32)
            if self.wd != 0: grad.add_(self.master_p[i], alpha=self.wd)

            self.velocity[i] = self.momentum * self.velocity[i] + grad

            if self.mode == 'euclidean':
                self.master_p[i].sub_(scale * self.velocity[i])
                p.data.copy_(self.master_p[i].to(DTYPE))
            else:
                indices = torch.searchsorted(self.grid, self.master_p[i].contiguous())
                v_rungs = self.velocity[i] * scale
                # CLIPPING: Prevent jumps of more than 10 rungs to avoid NaN/Inf cliffs
                v_rungs = torch.clamp(v_rungs, -10, 10)
                v_rounded = torch.floor(v_rungs + torch.rand_like(v_rungs)).to(torch.int32)
                new_indices = torch.clamp(indices - v_rounded, 0, len(self.grid) - 1)
                new_floats = self.grid[new_indices.long()].view(p.shape)
                self.master_p[i].copy_(new_floats)
                p.data.copy_(new_floats.to(DTYPE))

            flips += (p.data != old_data).sum().item()
        return flips

# --- 2. MODEL ARCHITECTURE ---

class CIFARNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Cast weights/biases to float32 for stable math in forward pass
        x = F.conv2d(x, self.conv1.weight.to(torch.float32), self.conv1.bias.to(torch.float32), padding=1)
        x = self.pool(F.relu(x))
        x = F.conv2d(x, self.conv2.weight.to(torch.float32), self.conv2.bias.to(torch.float32), padding=1)
        x = self.pool(F.relu(x))
        x = x.view(-1, 64 * 8 * 8)
        x = F.linear(x, self.fc1.weight.to(torch.float32), self.fc1.bias.to(torch.float32))
        x = F.relu(x)
        return F.linear(x, self.fc2.weight.to(torch.float32), self.fc2.bias.to(torch.float32))

# --- 3. COMPARISON EXECUTION ---

def run_cifar_comparison(epochs=5):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    train_loader = DataLoader(datasets.CIFAR10('./data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.CIFAR10('./data', train=False, transform=transform), batch_size=512)

    for mode, val in [('euclidean', 0.05), ('uniform', 6.0)]:
        print(f"\n>>> Running: {mode.upper()} (LR/Scale: {val})")
        model = CIFARNet().to(DEVICE)
        for p in model.parameters(): p.data = p.data.to(DTYPE)
        optimizer = MasterOptim(model.parameters(), val, mode=mode)

        for epoch in range(epochs):
            curr_scale = val * 0.5 * (1 + math.cos(math.pi * epoch / epochs)) if mode == 'uniform' else val
            model.train()
            epoch_flips, epoch_loss = 0, 0
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                model.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                epoch_flips += optimizer.step(curr_scale)
                epoch_loss += loss.item()

            model.eval()
            correct = sum((model(x.to(DEVICE)).argmax(1) == y.to(DEVICE)).sum().item() for x, y in test_loader)
            print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f} | Acc: {correct/10000:.2%} | Flips: {epoch_flips:,}")

if __name__ == "__main__":
    run_cifar_comparison(epochs=5)
