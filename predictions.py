"""
Download image from internet and performs inference
"""
import torch
from torch import nn
from PIL import Image
import requests
import torchvision
from typing import List
from timeit import default_timer as timer
import matplotlib.pyplot as plt

def prediction_on_image(url : str , model: nn.Module ,
                        model: str,
                        class_names: List[str] ,
                        transform: torchvision.transforms,
                        device: torch.device) -> float:
    """Function to download and image and predict on it.

    Taking in an image url from the image and takes in a PyTorch
    model and performs inference on it and plots the image along with its
    prediction as title and returns time taken for
    inference.

    Args:
        url: url of the image to download
        model: PyTorch model
        class_names: List that contains the class names 
        transform: torchvison transform to resize and convert image to tensor
        device: GPU if available else cpu
    Returns:
        Time(float) taken for inference
    Example:
        time_taken = prediction_on_image(url = "path of image from internet",
        model = your_model ,
        class_names = class_names , 
        transform = some_transform,
        device = your_device)     
    """        
    print(f"[INFO] Downloading image")
    with open("image.jpg", "wb") as f:
        request = requests.get(url)
        f.write(request.content)
    with Image.open("image.jpg") as img:
        image = transform(img)
        image = image.unsqueeze(dim = 0)
        start = timer()
        model.eval()
        with torch.inference_mode():
            y_logits = model(image.to(device))
            y_preds = torch.softmax(y_logits,dim = 1).argmax(dim = 1)
        end = timer()
        plt.imshow(img)
        plt.title(f"Prediction: {class_names[y_preds]}")
        plt.axis(False)
    return end-start
    
if __name__ == '__main__':
    print("Main")
