from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms

def load_datasets():
    transform = transforms.ToTensor()
    # ex: data/FashionMNIST, set root='./data'
    train = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
    test = datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)
    train_dataloader = DataLoader(train, batch_size=5, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=5, shuffle=True)
    return train_dataloader, test_dataloader





