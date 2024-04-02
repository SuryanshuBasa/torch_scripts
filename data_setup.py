"""
Contains fucntionality to setup data for image classifications
using PyTorch Datasets and DataLoaders
"""
from torchvision import datasets ,transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

NUM_WORKERS = os.cpu_count() // 2

def get_dataloaders(
  train_dir : str,
  test_dir : str,
  train_transforms : transforms,
  test_transforms : transforms,
  num_workers : int = NUM_WORKERS,
  batch_size : int = 32
):
    """Creates train and test dataloaders in PyTorch.
    
    Takes in paths for train and test directories and converts the directories
    in standard image classification format into PyTorch DataLoaders
    
    Args:
    train_dir: path/to/train_dir
    test_dir: path/to/test_dir
    train_transforms: torch transforms that must be applied to the train_images
    test_transforms: torch transforms that must be applied to the test_images
    num_workers: no of cpu cores
    batch_size: defines batch size to batch images
    
    Returns:
    A tuple of (train_dataloader , test_dataloader , class_names) where class names
    is a list containing class names of the data
    Example usage:
    train_dataloader , test_dataloader , class_names = get_dataloaders(
      train_sir = path,
      test_dat = path,
      transforms = transforms.Compose([]) object
      batch_size = 32
    )
    """
    print("[INFO] Setting DataLoader...")
    # Use ImageFolder to convert into torchvison.datasets
    train_data = datasets.ImageFolder(root = train_dir,
                            transform = train_transforms)
    
    test_data = datasets.ImageFolder(root = test_dir,
                          transform = test_transforms)
    # Convert torchvision.datasets to DataLoader
    train_dataloader = DataLoader(dataset = train_data,
    batch_size = batch_size,
    shuffle = True,
    # num_workers = NUM_WORKERS,
    pin_memory = True
    )
    test_dataloader = DataLoader(dataset = test_data,
    batch_size = batch_size,
    shuffle = False,
    # num_workers = NUM_WORKERS,
    pin_memory = True
    )
    print("[INFO] Done")
    
    
    return train_dataloader , test_dataloader , train_data.classes
