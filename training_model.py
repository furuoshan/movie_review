import os
import torch
from unet import UNet
from dataloader import load_transformer_dataset
import torchvision
from torch.optim import Adam
import logging
from torch import nn
import torch.nn.functional as F
from test_model import accuracy

def sinusoidalnumber(number):
    return torch.randn(number, 1, image_size, image_size)

def get_loss(x, label, embedding_t):
    return F.mse_loss(x, embedding_t[label])

def load_checkpoint(model, optimizer, device, filepath="./checkpoint.pth"):
    checkpoint = torch.load(filepath, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]
    t = checkpoint["t"]
    return epoch, t


def save_checkpoint(epoch, model, optimizer, loss, t, filepath="./checkpoint.pth"):
    tempt_filepath = "./checkpoint_tempt.pth"
    torch.save({"epoch": epoch,
                "model": model.state_dict(),
               "optimizer": optimizer.state_dict(),
                "t": t},
               tempt_filepath)
    os.replace(tempt_filepath, filepath)
    with open("loss.txt", "a", encoding="utf-8") as f:
        f.write(f"epoch: {epoch}, loss: {loss}\n")

T = 10
image_size = 32
batch_size = 128
file_path = "./checkpoint.pth"

if __name__=="__main__":
    train_loader, test_loader = load_transformer_dataset(image_size, batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = UNet()
    model.to(device)
    model.train()
    optimizer = Adam(model.parameters(), lr=1e-4)
    if os.path.isfile(file_path):
        epoch, embedding_t = load_checkpoint(model, optimizer, device, file_path)
    else:
        epoch = 0
        embedding_t = sinusoidalnumber(T)
    embedding_t = embedding_t.to(device)
    try:
        while True:
            print(f"epoch {epoch+1} is training\n")
            loss_aver = 0
            aver = 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(dtype=torch.long, device=device)
                optimizer.zero_grad()
                x = model(images)
                loss = get_loss(x, labels, embedding_t)
                loss.backward()
                optimizer.step()
                loss_aver+=loss.item()
                aver+=1
            epoch += 1
            save_checkpoint(epoch, model, optimizer, loss_aver/aver, embedding_t, file_path)
            accu = accuracy(epoch, model, test_loader, embedding_t)
            print(f"loss: {loss_aver/aver} accuracy: {accu*100}%\n")
    except KeyboardInterrupt:
        pass
