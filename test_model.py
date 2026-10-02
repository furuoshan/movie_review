from unet import UNet
from dataloader import load_transformer_dataset, show_image
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

image_size = 32
batch_size = 128
file_path = "./checkpoint.pth"

def check_num(x, t):
    loss = []
    for i in range(10):
        loss.append(F.mse_loss(x, t[i]).item())
    return loss.index(min(loss))

def load_checkpoint(model, optimizer, device, filepath="./checkpoint.pth"):
    checkpoint = torch.load(filepath, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]
    t = checkpoint["t"]
    return epoch, t

def accuracy(epoch, model, test_loader, t):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.eval()
    correct = 0
    sum = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            x = model(images)
            for idx, i in enumerate(labels):
                sum += 1
                if check_num(x[idx], t) == i: correct += 1
##                else:
##                    plt.title(f"True: {i}, Predicted: {check_num(x[idx], t)}")
##                    show_image(images[idx])
    with open("./accuracy.txt", "a") as f:
        f.write(f"epoch: {epoch} accuracy: {correct/sum*100}%\n")
    model.train()
    return correct/sum
if __name__ == "__main__":
    pass