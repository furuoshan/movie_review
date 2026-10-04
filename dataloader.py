from torch.utils.data import DataLoader
import torch
from datasets import load_dataset

def load_checkpoint(model, optimizer, device, filepath="./checkpoint.pth"):
    checkpoint = torch.load(filepath, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.vocab = checkpoint["vocab"]
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]
    return epoch

def load_transformer_dataset(batch_size) -> DataLoader:
    train_dataset = load_dataset(".\\data", split="train")
    test_dataset = load_dataset(".\\data", split="test")
    # unsupervised_dataset = load_dataset(".\\data", split="unsupervised")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    # unsupervised_loader = DataLoader(unsupervised_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, test_loader, None

def show_image(image):
    pass



if __name__ == "__main__":
##    dataloader = load_transformer_dataset(image_size=28, batch_size=64)
##    show_image(dataloader)
##   print(next(iter(dataloader))[0].shape)
    pass



