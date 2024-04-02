"""
Utility fucnction to plot results from a
PyTorch model
"""
from typing import Dict , List , Tuple
import matplotlib.pyplot as plt

def plot_loss_curve(results: Dict[str,List[float]],
                   model_name: str = "model"):
    """Utility function to plot metric
    
    Function to plot metrics like train test loss and
    accuracy of a PyTorch model.
    
    Args:
         results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
        model_name: name of the model for thr title
    """
    epochs = range(len(results["train_loss"]))
    plt.figure(figsize=(10,7))
    
    # Plot losses
    plt.subplot(1,2,1)
    plt.plot(epochs ,results["train_loss"], label = "train_loss")
    plt.plot(epochs ,results["test_loss"], label = "test_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.title("Loss Curves")
    plt.legend()

    # Plot accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs ,results["train_acc"], label = "train_acc")
    plt.plot(epochs ,results["test_acc"], label = "test_acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()

    plt.suptitle(f"{model_name}")

# plot_loss_curve(results)

            
