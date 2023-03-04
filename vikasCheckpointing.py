import argparse
import string
import augmentations as aug
import torch
from torch import nn
import torchvision.datasets as datasets
from pathlib import Path
from VICREGCheckpointing import VICReg
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
#from torch.utils.tensorboard import SummaryWriter
#from torchlars import LARS
import bitsandbytes as bnb
from torch.utils.checkpoint import checkpoint_sequential
import torch.nn.functional as F


hparams_variance_loss_epsilon = 0.0001
gamma_var = 1

def loss_function(x,y):
    #In Varinace Loss
    invarinace_loss = F.mse_loss(x, y)
    # Variance Loss
    std_x = torch.sqrt(x.var(dim=0) + hparams_variance_loss_epsilon)
    std_y = torch.sqrt(y.var(dim=0) + hparams_variance_loss_epsilon)
    loss_var_x = torch.mean(F.relu(gamma_var - std_x))
    loss_var_y = torch.mean(F.relu(gamma_var - std_y))
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
    total_loss = loss_cov + (25 * variance_loss) + (25*invarinace_loss)
    return total_loss

def main(args):
    print(args)

    #writer = SummaryWriter()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #transforms = aug.TrainTransformOriColorJitter()
    transforms = aug.TrainTransformOriColorJitterPerspective()
    dataset = datasets.ImageFolder(args.data_dir, transforms)

    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle = True
        )

    # num_epochs=1
    learning_rate=0.15

    model = VICReg(args).to(device)
    if args.loadStoredModel == 0:
        print("Loading Model")
        model.load_state_dict(torch.load(args.data_dir / "model.pth"))
    #Defining the optimizer and lr scheduler:
    optimizer = bnb.optim.AdamW8bit(model.parameters(),lr=learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate, momentum=0.9)
    #optimizer = LARS(optimizer=optimizer, eps=1e-8, trust_coef=0.001)

    #scaler = torch.cuda.amp.GradScaler()
    best_epoch_loss = 1e10
    log_Step=0
    for epoch in range(args.epochs):
        running_loss = 0.0
        for step, ((x,y), _) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            #Applying gradient checkpointing
            segments = 2

            # get the modules in the model. These modules should be in the order
            # the model should be executed
            modules = [module for k, module in model._modules.items()]
            # now call the checkpoint API and get the output
            output1 = checkpoint_sequential(modules, segments, x)
            output2 = checkpoint_sequential(modules, segments, y)
            total_loss = loss_function(output1,output2)

            #total_loss, invarinace_loss, variance_loss, loss_cov = model.forward(x,y)
            if not torch.isnan(total_loss):
                total_loss.backward()
                optimizer.step()
            if step%50==0:
                print(total_loss.item())
                log_Step=log_Step+1
            running_loss += total_loss.item()
        if(running_loss<best_epoch_loss):
            best_epoch_loss = running_loss
            print("[Info] Found New Best Model With Loss: ", best_epoch_loss)
            torch.save(model.state_dict(), args.data_dir / "model.pth")
            #torch.save(model.resnet.state_dict(), args.data_dir / "resnet18.pth")
        print('Epoch Loss: {:.4f}'.format(running_loss))

    #writer.flush()
    #writer.close()
    print("Ending Program")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Representation Learning for DonkeyCar")
    parser.add_argument("--data-dir", type=Path, required=True, help='Path to the input Images')
    parser.add_argument("--mlp", default="4096-1024",
                        help='Size and number of layers of the MLP expander head')
    parser.add_argument("--epochs", type=int, default=100,
                        help='Number of epochs')
    parser.add_argument("--loadStoredModel", type=int, default=-1,
                        help='Load Previous model')
    parser.add_argument("--batch-size", type=int, default=80,
                        help='Model Batch Size')
    parser.add_argument("--activation", default="gelu",
                        help='Activation Function')
    args = parser.parse_args()
    main(args)





