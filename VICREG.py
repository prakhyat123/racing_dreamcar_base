import torch
from torch import nn
import torch.nn.functional as F

class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # What is this??
        self.num_features = int(args.mlp.split("-")[-1])
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        self.embedding = 1000
        self.projector = Projector(args, self.embedding)
        self.hparams_variance_loss_epsilon = 0.0001
        self.gamma_var = 1

    def forward(self, x, y):
        x = self.projector(self.resnet(x))
        y = self.projector(self.resnet(y))
        #In Varinace Loss
        invarinace_loss = F.mse_loss(x, y)
        # Variance Loss
        std_x = torch.sqrt(x.var(dim=0) + self.hparams_variance_loss_epsilon)
        std_y = torch.sqrt(y.var(dim=0) + self.hparams_variance_loss_epsilon)
        loss_var_x = torch.mean(F.relu(self.gamma_var - std_x))
        loss_var_y = torch.mean(F.relu(self.gamma_var - std_y))
        variance_loss = loss_var_x + loss_var_y
        #Covariance Loss
        N, D = x.shape
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        cov_x = ((x.T @ x) / (N-1)).square() # D * D
        cov_y = ((y.T @ y) / (N-1)).square() # D * D
        loss_covariance_x = (cov_x.sum() - cov_x.diagonal().sum()) / D
        loss_covariance_y = (cov_y.sum() - cov_y.diagonal().sum()) / D
        loss_cov = loss_covariance_x + loss_covariance_y

        total_loss = loss_cov + 25*(variance_loss + invarinace_loss)

        return total_loss, invarinace_loss, variance_loss, loss_cov



def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers=[]
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f)-2):
        layers.append(nn.Linear(f[i],f[i+1]))
        # BatchNorm 1d??
        layers.append(nn.BatchNorm1d(f[i+1]))
        layers.append(nn.LeakyReLU(0.01))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)




