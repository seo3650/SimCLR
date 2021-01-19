import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MLP(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )
    
    def forward(self, x):
        out = self.layers(x)
        return out


class SimCLR(nn.Module):
    def __init__(self, model_g):
        super(SimCLR, self).__init__()
        self.model_f = models.resnet18(pretrained=False)
        self.model_g = model_g
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2).float()
        out = self.model_g(self.model_f(x))
        return out
class NT_XentLoss:
    def __init__(self, batch_size, temperature, device):
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.mask = self._make_mask(batch_size)

    def _make_mask(self, batch_size):
        "Make mask for correlated samples"
        mask = torch.ones((2*batch_size, 2*batch_size), dtype=bool).to(self.device)
        mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size+i] = 0
            mask[batch_size+i, i] = 0
    
    def __call__(self, zi, zj):
        criterion = nn.CrossEntropyLoss()
        # Calculate sim
        z = torch.cat((zi, zj))
        cos = nn.CosineSimilarity(dim=2)
        sim = cos(z.unsqueeze(0), z.unsqueeze(1)) # Shape: 2N * 2N

        # Calculate loss
        sim = sim / self.temperature
        sim_ij = torch.diag(sim, self.batch_size)
        sim_ji = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_ij, sim_ji)).reshape(2*self.batch_size, 1)
        negative_samples = sim[self.mask].reshape(2*self.batch_size, -1)

        target = torch.zeros(2*self.batch_size).to(self.device).long()
        input = torch.cat((positive_samples, negative_samples), dim=1)
        loss = criterion(input, target)
        return loss

def make_model(input_dim, hid_dim, out_dim):
    model = SimCLR(
        MLP(input_dim, hid_dim, out_dim)
    )
    return model