# HUMAN_ACTIVITY_RCCOGNITION_USING_SMARTPHONE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VideoFolder
from torchvision.models.video import r3d_18

# Define the transform for preprocessing the video frames
preprocess = transforms.Compose([
    transforms.Resize((128, 171)),  # Resize frames
    transforms.RandomCrop((112, 112)),  # Crop frames
    transforms.ToTensor(),  # Convert frames to tensor
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])  # Normalize frames
])

# Load the dataset
dataset = VideoFolder('path/to/dataset', transform=preprocess)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Define the 3D CNN model
model = r3d_18(pretrained=True)
model.fc = nn.Linear(512, len(dataset.classes))  # Adjust the output size for the number of action classes

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0
    for videos, labels in dataloader:
        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(videos)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

# Save the trained model
torch.save(model.state_dict(), 'path/to/save/model.pth')

