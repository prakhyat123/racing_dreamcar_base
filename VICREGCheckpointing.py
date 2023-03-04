import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # What is this??
        self.num_features = int(args.mlp.split("-")[-1])
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        #self.resnet2 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.embedding = 1000
        self.projector = Projector(args, self.embedding)

    def forward(self, inp):
        x = self.resnet(inp)
        x = self.projector(x)
        return output



def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers=[]
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f)-2):
        layers.append(nn.Linear(f[i],f[i+1]))
        # BatchNorm 1d??
        layers.append(nn.BatchNorm1d(f[i+1]))
        if args.activation == "gelu":
            layers.append(nn.GELU())
        elif args.activation == "relu":
            layers.append(nn.ReLU(True))
        elif args.activation == "lrelu":
            layers.append(nn.LeakyReLU(0.01))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)




