
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from norse.torch import LIFCell
from norse.torch.functional.lif import LIFParameters

# Hyperparameters
batch_size = 64
epochs = 10
learning_rate = 1e-3
# timesteps_list = [2, 4, 6]
# tau_m_list = [5.0, 10.0, 20.0]
timesteps_list = [12]
tau_m_list = [1.0, 2.5, 5.0, 10.0]

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the spiking neural network
class SpikingNN(nn.Module):
    def __init__(self, tau_m):
        super(SpikingNN, self).__init__()
        self.tau_m = tau_m
        self.lif_params = LIFParameters(tau_syn_inv=1 / tau_m, tau_mem_inv=1 / tau_m)

        # Input size matches flattened CIFAR-10 images
        self.fc1 = nn.Linear(3 * 32 * 32, 256)
        self.lif1 = LIFCell(p=self.lif_params)
        self.fc2 = nn.Linear(256, 10)  # Output layer for 10 classes

    def forward(self, x, timesteps):
        batch_size = x.size(0)

        x = x.reshape(batch_size, 3 * 32 * 32)

        # Pass through the first fully connected layer (fc1) once
        x = torch.relu(self.fc1(x))
        # print(f"Shape after fc1: {x.shape}")  # Debug

        # Initialize LIF state
        dummy_input = torch.zeros(batch_size, 256, device=x.device)
        state1 = self.lif1.initial_state(dummy_input)

        # Process through the LIF cell across timesteps
        for t in range(timesteps):
            spk1, state1 = self.lif1(x, state1)
            # print(f"Shape after lif1 at timestep {t}: {spk1.shape}")  # Debug

        # Pass the output of LIFCell to the final layer
        out = self.fc2(spk1)
        # print(f"Shape before fc2: {spk1.shape}")  # Debug
        return out




# Training function
def train(model, device, train_loader, optimizer, timesteps):
    model.train()
    criterion = nn.CrossEntropyLoss()
    i = 0
    print(len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(i)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, timesteps)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # i += 1

# Testing function
def test(model, device, test_loader, timesteps):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, timesteps)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(test_loader.dataset)

# Main script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = []

print(device)

for timesteps in timesteps_list:
    # i = 0
    for tau_m in tau_m_list:
        # print(i)
        print(f"Training with T={timesteps} and tau_m={tau_m}...")
        model = SpikingNN(tau_m).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            train(model, device, train_loader, optimizer, timesteps)
            accuracy = test(model, device, test_loader, timesteps)
            print(f"Epoch {epoch+1}, Accuracy: {accuracy:.4f}")
        # i += 1

        results.append((timesteps, tau_m, accuracy))

print("Final Results:")
for result in results:
    print(f"T={result[0]}, tau_m={result[1]}, Accuracy={result[2]:.4f}")
