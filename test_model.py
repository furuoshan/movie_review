import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

def Accuracy(epoch, model, test_loader, max_length):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.eval()

    correct = 0
    sum = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            x = [text.lower().split() for text in data["text"]]
            for text in x:
                while len(text) > max_length:
                    text.pop()
                while len(text) < max_length:
                    text.append("<pad>")
            x = model(x)
            for id, vec in enumerate(x):
                if vec[0]>vec[1] and data["label"][id] == 0: correct += 1
                elif vec[1]>=vec[0] and data["label"][id] == 1: correct += 1
                sum += 1
        model.train()
    with open("./accuracy.txt", "a") as f:
        f.write(f"epoch: {epoch} accuracy: {correct/sum*100}%\n")
    return correct/sum

if __name__ == "__main__":
    pass