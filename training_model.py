import os
import torch
from unet import ReviewClassifier
from dataloader import load_transformer_dataset, load_checkpoint
import torchvision
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
from test_model import Accuracy

def get_loss(x, y):
    y = y.to(x.device)
    return F.cross_entropy(x, y)

def save_checkpoint(epoch, model, optimizer, loss, vocab, filepath="./checkpoint.pth"):
    tempt_filepath = "./checkpoint_tempt.pth"
    torch.save({"epoch": epoch,
                "model": model.state_dict(),
               "optimizer": optimizer.state_dict(),
                "vocab": vocab},
               tempt_filepath)
    os.replace(tempt_filepath, filepath)
    with open("loss.txt", "a", encoding="utf-8") as f:
        f.write(f"epoch: {epoch}, loss: {loss}\n")

max_length = 128
batch_size = 128
embedding_dim = 512
file_path = "./checkpoint.pth"
lr = 5e-5
warm_up_steps = 2000

if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, test_loader, unsupervised_loader = load_transformer_dataset(batch_size)

    model = ReviewClassifier(embedding_dim, max_length, train_loader).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    try:
        if os.path.exists(file_path): epoch = load_checkpoint(model, optimizer, device, file_path)
        else: epoch = 0
        while True:
            print(f"epoch {epoch+1} is training\n")
            model.train()
            loss_sum = 0
            num = 0
            step = epoch*len(train_loader)
            for batch_idx, data in enumerate(train_loader):
                optimizer.param_groups[0]["lr"] = 1 / embedding_dim * min((step + 1) ** (-0.5),
                                                            (step + 1) * (warm_up_steps ** (-1.5)))
                x = [text.lower().split() for text in data["text"]]
                for text in x:
                    while len(text) > max_length:
                        text.pop()
                    while len(text) < max_length:
                        text.append("<pad>")

                x = model(x)
                loss = get_loss(x, data["label"])
                loss_sum += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                num += 1
                step += 1
            epoch += 1
            save_checkpoint(epoch, model, optimizer, loss_sum / num, model.vocab, file_path)
            accuracy = Accuracy(epoch, model, test_loader, max_length)
            print(f"loss: {loss_sum / num} accuracy:{accuracy}\n")
            if epoch >= 50: break
    except KeyboardInterrupt:
        pass
