import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from spikingjelly.clock_driven import functional, surrogate, neuron

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # normalization for test set
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    class SpikingCNN(nn.Module):
        def __init__(self):
            super(SpikingCNN, self).__init__()
            # Convolutional layers
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.lif1 = neuron.LIFNode(surrogate_function=surrogate.ATan())
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.lif2 = neuron.LIFNode(surrogate_function=surrogate.ATan())
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.lif3 = neuron.LIFNode(surrogate_function=surrogate.ATan())
            self.pool = nn.MaxPool2d(2, 2)
            
            # Fully connected layers
            self.fc1 = nn.Linear(128 * 4 * 4, 512)
            self.lif4 = neuron.LIFNode(surrogate_function=surrogate.ATan())
            self.fc2 = nn.Linear(512, 10)
            self.lif5 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        
        def forward(self, x):
            # Apply convolutional layers with LIF activation
            x = self.lif1(self.conv1(x))
            x = self.pool(x)
            x = self.lif2(self.conv2(x))
            x = self.pool(x)
            x = self.lif3(self.conv3(x))
            x = self.pool(x)
            
            # Flatten the tensor
            x = x.view(x.size(0), -1)
            
            # Apply fully connected layers with LIF activation
            x = self.lif4(self.fc1(x))
            x = self.lif5(self.fc2(x))
            return x

    model = SpikingCNN().to(device)

    def train(model, trainloader, criterion, optimizer, epoch):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # reset the membrane potentials after each batch
            functional.reset_net(model)
            
            running_loss += loss.item()
            if batch_idx % 100 == 99:
                print(f'[Epoch: {epoch + 1}, Batch: {batch_idx + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    def evaluate(model, testloader, criterion):
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # reset the membrane potentials after each batch
                functional.reset_net(model)
        
        print(f'Test Loss: {test_loss / len(testloader):.3f}, Accuracy: {100. * correct / total:.3f}%')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(10):
        train(model, trainloader, criterion, optimizer, epoch)
        evaluate(model, testloader, criterion)

if __name__ == '__main__':
    main()