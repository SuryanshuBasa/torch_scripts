"""Functions to automate train and testing of the model
in PyTorch
"""
from typing import Tuple , Dict , List
from tqdm.auto import tqdm
import torch
from torch import nn

def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               optimizer : torch.optim.Optimizer,
               loss_fn : nn.Module,
               device: torch.device
               ) -> Tuple[float,float]:
  """Training loop to train a model for one epoch.

  Funtion that takes in a PyTorch model, puts it in train mode and
  performs training steps like forward pass, loss calculation,
  backpropagation and gradient descent

  Args:
    model : PyTorch model that is to be trained
    dataloader: torch.utils.DataLoader to load batches of images
    optimizer: optimizer for gradient descent
    loss_fn: loss function for loss calculation
    device: device to run the model
  Returns:
    A tuple of [train loss and train accuracy] that can be tracked in the
    wrapper train function.
    [1.0234 , 12.55]
  Example:
    train_loss, train_acc = train_step(model,dataloader,optimizer,loss_fn,device)
  """

  # Initialise loss and accuracy
  train_loss , train_acc = 0 , 0
  # Put model to train mode
  model.train()

  for X,y in dataloader:
    # Put data to device
    X, y = X.to(device), y.to(device)
    # Forward pass
    y_logits = model(X)
    y_preds = torch.softmax(y_logits , dim = 1).argmax(dim=1)
    # Calculate Loss
    loss = loss_fn(y_logits, y)
    # Zero grad
    optimizer.zero_grad()
    # Backpropagation
    loss.backward()
    # Gradient Descent
    optimizer.step()
    # Accumulate Loss and Acc
    train_loss += loss.item()
    train_acc += (y_preds == y).sum().item()/len(y_logits)
  # Normalize Loss and Acc
  train_loss /= len(dataloader)
  train_acc /= len(dataloader)
  return train_loss , train_acc

def test_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn : nn.Module,
               device: torch.device
               ) -> Tuple[float,float]:
  """Testing loop to test a model for the epoch.

  Funtion that takes in a PyTorch model, puts it in eval mode and
  turns on inference_mode() to perform test operation and gathers
  loss and accuracy.

  Args:
    model : PyTorch model that is to be trained
    dataloader: torch.utils.DataLoader to load batches of images
    loss_fn: loss function for loss calculation
    device: device to run the model
  Returns:
    A tuple of [test loss and test accuracy] that can be tracked in the
    wrapper test function.
    [1.0234 , 12.55]
  Example:
    test_loss , test_acc = test_step(model,dataloader,loss_fn,device)
  """
  # Initialise loss and accuracy
  test_loss , test_acc = 0 , 0
  # Put model to eval mode
  model.eval()
  # On inference context manager
  with torch.inference_mode():
    for X,y in dataloader:
      # Put data to device
      X, y = X.to(device), y.to(device)
      # Forward pass
      y_logits = model(X)
      y_preds = torch.softmax(y_logits , dim = 1).argmax(dim=1)
      # Calculate Loss
      loss = loss_fn(y_logits, y)
      # Accumulate Loss and Acc
      test_loss += loss.item()
      test_acc += (y_preds == y).sum().item()/len(y_logits)
    # Normalize Loss and Acc
  test_loss /= len(dataloader)
  test_acc /= len(dataloader)
  return test_loss , test_acc

def train(model: nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: nn.Module,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          epochs: int
          ) -> Dict[str,List[float]]:
  """Trains and Test PyTorch model

  Uses the train_step() and test_step() fucntions to train the model
  for a given number of epochs and prints out loss and accuracy.

  Args:
    model : PyTorch model that is to be trained
    train_dataloader: torch.utils.DataLoader to load batches of train images
    test_dataloader: torch.utils.DataLoader to load batches of test images
    loss_fn: loss function for loss calculation
    optimizer: optimizer for gradient descent
    device: device to run the model
    epochs: no of interations to train model
  Returns:
    A dict which contains train test loss and accuracy for every epoch
    in the format:
    {
      train loss: [...],
      train acc: [...],
      test loss: [...],
      test acc: [...]
    }
  Example:
    results = {
      "train_loss": [1.245,1.134,1.092],
      "train_acc": [12.12,13.03,16.02],
      "test_loss": [1.245,1.13,1.092],
      "test_acc": [12.12,13.03,16.02],
    }
  """
  results = {
      "train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }

  for epoch in tqdm(range(epochs)):
    # Run train step
    train_loss , train_acc  = train_step(model = model,
                                         dataloader = train_dataloader,
                                         loss_fn = loss_fn,
                                         optimizer = optimizer,
                                         device = device)
    # Run test step
    test_loss , test_acc  = test_step(model = model,
                                         dataloader = test_dataloader,
                                         loss_fn = loss_fn,
                                         device = device)
    # Append respective values
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

    # Print values
    print(f"\nEpochs: {epoch+1} | "
          f"Train_loss: {train_loss: .4f} | "
          f"Train_acc: {train_acc: .2f} | "
          f"Test_loss: {test_loss: .4f} | "
          f"Test_acc: {test_acc: .2f}"
          )
  return results
