import argparse
import model
import torch
import torchvision

def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_0 = model.TinyVGG(input_shape = 3,
                         output_shape = 3,
                         hidden_units = 10).to(device)
    model_0.load_state_dict(torch.load('model/food_dataset_model.pt'))
    return model_0
    
def get_pred(model_0,url):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load image
    img = torchvision.io.read_image(url).type(torch.float32).unsqueeze(dim=0)/255.
    transform = torchvision.transforms.Resize(size= (64,64))
    img = transform(img)
    print(img.shape)
    model_0.eval()
    with torch.inference_mode():
        y_pred = torch.softmax(model_0(img.to(device)),dim = 1).argmax(dim = 1)
        return y_pred
        
def main():
    dict = {0 : "pizza",
           1: "steak",
           2: "sushi"}
    parser = argparse.ArgumentParser()
    parser.add_argument("--url" , help="URL for image to download")
    args = parser.parse_args()
    model_0 = load_model()
    print(f"Prediction is; {dict[get_pred(model_0 , args.url).item()]}")
    
if __name__ == "__main__":
    main()
