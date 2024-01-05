import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

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

batch_size = 64

train_data = datasets.MNIST('./data', 
                            train=True, 
                            download=True, 
                            transform=transforms.ToTensor())
test_data = datasets.MNIST('./data', 
                           train=False, 
                           download=True, 
                           transform=transforms.ToTensor())

# train_data = datasets.FashionMNIST('./data', 
#                             train=True, 
#                             download=True, 
#                             transform=transforms.ToTensor())
# test_data = datasets.FashionMNIST('./data', 
#                            train=False, 
#                            download=True, 
#                            transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_data, 
                                           batch_size=batch_size, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, 
                                          batch_size=batch_size, 
                                          shuffle=True)

# print("학습 데이터 수: ", len(train_loader.dataset))
# print("테스트 데이터 수: ", len(test_loader.dataset))
# print("데이터 하나의 크기: ", train_loader.dataset[0][0].shape)
# print("첫번째 데이터의 정답: ", train_loader.dataset[0][1])

# plt.imshow(train_loader.dataset[0][0][0])
# plt.show()

# Window
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mac
# device = torch.device("mps")

model = CNN().to(device)

lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()

epochs = 100

for epoch in range(epochs):
    # 학습 코드
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, total_loss / len(train_loader)))
    
model.eval()
test_loss = 0
correct = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += loss_func(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
accuracy = correct / len(test_loader.dataset) * 100

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset), accuracy))
