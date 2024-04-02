import argparse
import torch
from torch import nn
from torchvision import transforms
import data_setup, engine, download_dataset, model, utils
from timeit import default_timer as timer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url" , help="URL for thr dataset to download" , default = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip")
    parser.add_argument("--epochs", help="No of epochs to train the model" , type = int)
    parser.add_argument("--batch_size" , help="Set the batch size number" ,type = int, default = 32)
    parser.add_argument("--learning_rate" , help="Set the learning rate if the optimizer" ,type = float, default = 0.001)
    parser.add_argument("--hidden_units" , help="Set the hidden_units in the model" , default = 10)
    
    args = parser.parse_args()
    
    # Setup Hyperparametes and device agnostic code
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device is set to {DEVICE}")
    URL = args.url
    EPOCHS = int(args.epochs)
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    HIDDEN_UNITS = args.hidden_units
    
    # Download dataset
    train_dir , test_dir = download_dataset.download_dataset(url = URL , dir_name = "food_dataset")
    # print(train_dir)
    # Setup data tranforms
    data_transform = transforms.Compose([
        transforms.Resize(size=(64,64)),
        transforms.ToTensor()
    ])
    # Get train test dataloaders
    train_dataloader , test_dataloader , class_names = data_setup.get_dataloaders(train_dir = train_dir,
                                                                                 test_dir = test_dir,
                                                                                  train_transforms= data_transform,
                                                                                  test_transforms = data_transform,
                                                                                  batch_size = BATCH_SIZE)
    print("[INFO] Getting model...")
    # Instantiate model
    model_0 = model.TinyVGG(input_shape = 3,
                         output_shape = len(class_names),
                         hidden_units = HIDDEN_UNITS).to(DEVICE)
    print("[INFO] Done")
    
    # Setup loss func and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = model_0.parameters(),
                                lr = LEARNING_RATE)
    print("[INFO] Starting training...")
    start = timer()
    # Train model
    results = engine.train(model = model_0,
                          train_dataloader = train_dataloader,
                          test_dataloader = test_dataloader,
                          loss_fn = loss_fn,
                          optimizer = optimizer,
                          epochs = EPOCHS,
                          device = DEVICE)
    end = timer()
    print(f"[INFO] Training took {end-start:.4f} seconds to complete")
    # Save model
    model_name = "food_dataset_model.pt"
    path = "model"
    utils.save_model(model = model_0 ,
                    target_dir = path,
                    model_name = model_name)    
if __name__ == '__main__':
    main()
