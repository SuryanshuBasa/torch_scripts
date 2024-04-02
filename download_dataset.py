"""
Script to get dataset already in Standard Image Classification format
from an URL
"""
import os
from pathlib import Path
import zipfile
import requests
from typing import Tuple
import argparse

def download_dataset(url: str , dir_name: str = "dataset") -> Tuple[str,str]:
    """Gets dataset in Standard Image Classification Format.
    from the url and is structed in the way /data/dirname/...
    and returns path pointing to the train and test dir
    
    Args:
        url: url to the dataset
        dir_name: name of the directory inside data
    Returns:
        A Tuple of 2 strings which are pointer to the train and test dirs in
        the dataset
    Example:
        train_dir, test_dir = downlaod_dataset("your_URL_here" , "name_of_dataset")
    """
    # Create a directory
    dir = Path("data")
    image_path = dir / dir_name
    if not image_path.is_dir():
        print("[INFO] Creating directory...")
        image_path.mkdir(exist_ok = True , parents = True)
        print("[INFO] Downloading Dataset...")
        with open(image_path / "dataset.zip" , "wb") as f:
            request = requests.get(url)
            f.write(request.content)
        print("[INFO] Finished downloading... Unzipping")
        with zipfile.ZipFile(image_path / "dataset.zip" , "r") as zipref:
            zipref.extractall(image_path)
        print("[INFO] Unzipping complete...")
    
        os.remove(image_path / "dataset.zip")
    else:
        print("Directory already exits, skipping download")
    
    return image_path/"train" , image_path/"test"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downloads dataset from the URL or directly downloads food101 dataset')
    parser.add_argument("--url" , help = "URL to download dataset at the target_dir" , type = str , default = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    args = parser.parse_args()
    download_dataset(url = args.url)