from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
import torch
import numpy as np
import matplotlib.pyplot as plt


def load_transformer_dataset(image_size=32,batch_size=128) -> DataLoader:
    transform_train = transforms.Compose([transforms.RandomRotation(degrees=15),
                                          transforms.Resize((image_size,image_size),
                                          interpolation=InterpolationMode.BILINEAR), transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.Resize((image_size, image_size),
                                         interpolation=InterpolationMode.BILINEAR),
                                         transforms.ToTensor()])

    train_dataset = datasets.MNIST(
        root='./data',  # 数据存储路径
        train=True,  # 加载训练集
        download=False,  # 如果数据不存在则自动下载
        transform=transform_train,
    )
    test_dataset = datasets.MNIST(
        root='./data',  # 数据存储路径
        train=False,  # 加载测试集
        download=False,  # 如果数据不存在则自动下载
        transform=transform_test,
    )
    return DataLoader(train_dataset, batch_size=batch_size, drop_last=True), DataLoader(test_dataset, batch_size=batch_size, drop_last=False)

def show_image(image):
    image = image.to("cpu")
    reverse_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda t: t * 255.0),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(mode="L"),
        ]
    )
    plt.imshow(reverse_transforms(image), cmap="gray")
    plt.show()



if __name__ == "__main__":
##    dataloader = load_transformer_dataset(image_size=28, batch_size=64)
##    show_image(dataloader)
##   print(next(iter(dataloader))[0].shape)
    pass



