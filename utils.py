"""
Function to save trained model
"""

from pathlib import Path
import torch
def save_model(
     model: torch.nn.Module,
               target_dir: str,
               model_name: str) -> Path:
    """Saves a PyTorch model to a target director
    
    Args:
        model : PyTorch model to save
        target_dir: Path to save the model
        model_name: Name of the model
    Returns:
        path_to_model as pathlib.Path  
    Example:
    save_model = save_model(model= model , target_dir = your/path/model.pth , model_name = "name")
    """
    # Create target dir
    dir = Path(target_dir)
    dir.mkdir(exist_ok = True, parents = True)
    
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name must end with '.pt' or .'pth' "
    model_name_path = dir / model_name
    
    
    # Saving model
    print(f"[INFO] Saving model to: {model_name_path}")
    torch.save(obj = model.state_dict(),
               f = model_name_path)
    return model_name_path
    
# if __name__ == '__main__':
#     name = save_model("data" , "model.pth")
#     print(name , type(name))
    
