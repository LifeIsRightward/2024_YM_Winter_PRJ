import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer2 = torch.nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(7 * 7 * 64, 625)
        self.fc2 = nn.Linear(625, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

device = torch.device("mps")
model = CNN().to(device)

model_path = 'CNN_MNIST'
model.load_state_dict(torch.load(model_path))

model.eval()
with torch.no_grad():
    image_path = './test.png'
    image = Image.open(image_path).convert('L') 
    transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    input_image = transform(image).unsqueeze(0).to(device)

    output = model(input_image)
    _, predicted = torch.max(output.data, 1)

    f = open("CNN_MNIST.out", 'w')
    f.write(str(predicted.item()))
    f.close()
    print(f"Predicted label: {predicted.item()}")