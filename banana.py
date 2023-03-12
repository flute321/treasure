import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torchvision
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam

transform_test = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])])

transform_crossval = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])])

transform_train = transforms.Compose([transforms.Resize(255),
                                       transforms.RandomRotation(30),
                                       transforms.RandomGrayscale(0.2),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(0.25),
                                       transforms.RandomPerspective(),
                                       transforms.RandomVerticalFlip(0.15),
                                       transforms.ColorJitter(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])])

train_data = datasets.ImageFolder('./banana/train', transform=transform_train)
test_data = datasets.ImageFolder('./banana/test', transform=transform_test)
crossval_data = datasets.ImageFolder('./banana/val', transform=transform_crossval)

batch_size = 20
train_load = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=True)
test_load = torch.utils.data.DataLoader(test_data, batch_size=batch_size,shuffle=True)
crossval_load = torch.utils.data.DataLoader(crossval_data,batch_size=batch_size, shuffle=True)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=4, padding=1)
        self.pool1 = nn.MaxPool2d(3,3)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=2, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(12)
        self.pool2 = nn.MaxPool2d(3,3)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(3,3)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=1, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24*4*4, 4)

    def forward(self, input):
        output = F.relu(self.pool1(self.conv1(input)))      
        output = F.relu(self.bn1(self.conv2(output)))     
        output = self.pool2(output)                        
        output = F.relu(self.pool3(self.conv4(output)))     
        output = F.relu(self.bn2(self.conv5(output)))     
        output = output.view(-1, 24*4*4)
        output = self.fc1(output)

        return output

model = Network()
classes = ('sigatoka', 'pestalotiopsis', 'healthy', 'cordana')

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.001)

path = "./myFirstModel.pth"
torch.save(model.state_dict(), path)

def testAccuracy(load):
    
    model.eval()
    accuracy = 0.0
    total = 0
    
    with torch.no_grad():
        for data in load:
            images, labels = data
            outputs = model(images)
            x , predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    return (100 * accuracy / total)


def train(num_epochs, load):
    
    best_accuracy = 0.0
    device = torch.device("cpu")
    model.to(device)

    for epoch in range(num_epochs):  
        running_loss = 0.0

        for i, (images, labels) in enumerate((load), 0):
            
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 10 == 0:    
                print('epoch: %d number: %5d loss: %.3f' %
                      (epoch + 1, i + 1, running_loss/10))
                running_loss = 0.0

        accuracy = testAccuracy(load)
        print('For epoch', epoch+1,'the accuracy is %d %%' % (accuracy))
        
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), path)
            best_accuracy = accuracy


def testBatch(load):
    images, labels = next(iter(load))

    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] 
                               for j in range(batch_size/2.0)))
  
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] 
                              for j in range(batch_size/2.0)))

def classAccuracy():
    class_correct = list(0. for i in range(4))
    class_total = list(0. for i in range(4))
    with torch.no_grad():
        for data in test_load:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(4):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))



train(30,train_load)
print('Finished Training')
print('Accuracy:', testAccuracy(test_load))

model.load_state_dict(torch.load(path))
testBatch(crossval_load)
train(5,crossval_load)
print('Finished Cross Validation')
print('Accuracy:', testAccuracy(test_load))

model1 = Network()
path = "myFirstModel.pth"
model1.load_state_dict(torch.load(path))

testBatch(test_load)
classAccuracy()

