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

def prediction_on_image(url : str ,
                         model: nn.Module ,
                        class_names: List[str] ,
                        device: torch.device,
                        transform: torchvision.transforms = None) -> float:
    """Function to download and image and predict on it.

    Taking in an image url from the image and takes in a PyTorch
    model and performs inference on it and plots the image along with its
    prediction as title and returns time taken for
    inference.

    Args:
        url: url of the image to download
        model: PyTorch model
        class_names: List that contains the class names 
        transform: torchvison transform to resize and convert image to tensor (defualt img transform is applied)
        device: GPU if available else cpu
    Returns:
        Time(float) taken for inference
    Example:
        time_taken = prediction_on_image(url = "path of image from internet",
        model = your_model ,
        class_names = class_names , 
        transform = some_transform, (optional)
        device = your_device)     
    """
    # Setting transforms if none
    if transform is None:
      img_transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
      ])
    else:
      img_transform = transform
           
    print(f"[INFO] Downloading image")
    # Download image
    with open("image.jpg", "wb") as f:
        request = requests.get(url)
        f.write(request.content)
    # Open img to perform operations
    with Image.open("image.jpg") as img:
        # Transform image
        image = transform(img)
        # Add batch dim
        image = image.unsqueeze(dim = 0)
        start = timer()
        # Get predictions
        model.eval()
        with torch.inference_mode():
            y_logits = model(image.to(device))
            y_preds = torch.softmax(y_logits,dim = 1).argmax(dim = 1)
            # Get confidence 
            conf = torch.softmax(y_logits,dim = 1).max()
        end = timer()
        plt.imshow(img)
        plt.title(f"Prediction: {class_names[y_preds]} | Conf: {conf:.4f}")
        plt.axis(False)
    return end-start
    
# if __name__ == '__main__':
#     print("Main")
