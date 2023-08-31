# Huggingface datasets
from datasets import load_dataset

### COCO-AB
dataset = load_dataset("coallaoh/COCO-AB")

### coco_captions 
dataset = load_dataset("conceptual_captions")



# Pytorch Datasets
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

### FashionMNIST
training_data = datasets.FashionMNIST(
    root="FashionMNIST_data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="FashionMNIST_data",
    train=False,
    download=True,
    transform=ToTensor()
)

### Flickr30k
data = datasets.Flickr30k(
    root="Flickr30k_data",
    ann_file="Flickr30k_data/annotations.json",
    transform=ToTensor()
)

