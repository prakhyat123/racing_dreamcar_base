import argparse
import augmentations as aug
import torch
from torch import nn
import torchvision.datasets as datasets
from pathlib import Path
from VICREG import VICReg
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

def main(args):
    print(args)

    writer = SummaryWriter()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transforms = aug.TrainTransform()
    dataset = datasets.ImageFolder(args.data_dir, transforms)

    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=512,
            shuffle = True
        )

    # num_epochs=1
    learning_rate=0.01

    model = VICReg(args).to(device)

    #Defining the optimizer and lr scheduler:
    optimizer=optim.AdamW(model.parameters(),lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.75)

    scaler = torch.cuda.amp.GradScaler()
    best_epoch_loss = 2e15
    for epoch in range(args.epochs):
        running_loss = 0.0
        for step, ((x,y), _) in enumerate(loader):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast():
                total_loss, invarinace_loss, variance_loss, loss_cov = model.forward(x,y)
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step_loss = total_loss.item() * x.size(0)
            if step%50==0:
                print(total_loss)
            writer.add_scalar("Total Loss/train", step_loss, step)
            writer.add_scalar("invarinace_loss/train", invarinace_loss, step)
            writer.add_scalar("variance_loss", variance_loss, step)
            writer.add_scalar("loss_cov", loss_cov, step)
            running_loss += step_loss
        if(running_loss<best_epoch_loss):
            best_epoch_loss = running_loss
            print("[Info] Found New Best Model With Loss: ", best_epoch_loss)
            torch.save(model.resnet.state_dict(), args.data_dir / "model.pth")
            torch.save(model.resnet.state_dict(), args.data_dir / "resnet18.pth")
        print('Epoch Loss: {:.4f}'.format(running_loss))
        scheduler.step()

    writer.flush()
    writer.close()
    torch.save(model.state_dict(), args.data_dir)
    print("Model Saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Representation Learning for DonkeyCar")
    parser.add_argument("--data-dir", type=Path, required=True, help='Path to the input Images')
    parser.add_argument("--mlp", default="2048-1024",
                        help='Size and number of layers of the MLP expander head')
    parser.add_argument("--epochs", type=int, default=10,
                        help='Number of epochs')
    args = parser.parse_args()
    main(args)





