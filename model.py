import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class SymbolDetector(nn.Module):
    def __init__(self, num_classes=4):
        super(SymbolDetector, self).__init__()
        conv_layers = []
        # 3 input channels, 32 output channels, 3x3 kernel, stride 1, padding 1
        # 3 is for RGB channels

        # 3 x 256 x 256 -> 32 x 256 x 256
        conv_layers.append(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1))
        # 32 x 256 x 256 -> 32 x 256 x 256
        # Do not add BatchNorm2d here, it performs poorly
        # conv_layers.append(nn.BatchNorm2d(32))
        # conv_layers.append(nn.GroupNorm(8, 32))
        # 32 x 256 x 256 -> 32 x 256 x 256
        conv_layers.append(nn.ReLU())
        # 32 x 256 x 256 -> 32 x 128 x 128
        conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        # 32 x 128 x 128 -> 64 x 128 x 128
        conv_layers.append(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1))
        # 64 x 128 x 128 -> 64 x 128 x 128
        # conv_layers.append(nn.BatchNorm2d(64))
        # conv_layers.append(nn.GroupNorm(8, 64))
        # 64 x 128 x 128 -> 64 x 128 x 128
        conv_layers.append(nn.ReLU())
        # 64 x 128 x 128 -> 64 x 64 x 64
        conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        # 64 x 64 x 64 -> 128 x 64 x 64
        conv_layers.append(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        # 128 x 64 x 64 -> 128 x 64 x 64
        # 128 x 64 x 64 -> 128 x 64 x 64
        conv_layers.append(nn.ReLU())
        # 128 x 64 x 64 -> 128 x 32 x 32
        conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))


        # Bottle neck layer
        # 128 x 32 x 32 -> 8 x 32 x 32
        conv_layers.append(nn.Conv2d(128, 16, kernel_size=3, stride=1, padding=1))
        # 16 x 32 x 32 -> 8 x 32 x 32
        conv_layers.append(nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0))


        # 8 x 32 x 32 -> 1 x (8*32*32)
        conv_layers.append(nn.Flatten())

        self.conv_layers = nn.Sequential(*conv_layers)

        fc1_layers = []
        # 8*32*32 -> 1024
        fc1_layers.append(nn.Linear(8*32*32, 1024))
        fc1_layers.append(nn.ReLU())
        # 1024 -> 512
        fc1_layers.append(nn.Linear(1024, 512))
        fc1_layers.append(nn.ReLU())
        # 512 -> 256
        fc1_layers.append(nn.Linear(512, 256)) 
        fc1_layers.append(nn.ReLU())
        # 256 -> 1
        fc1_layers.append(nn.Linear(256, 1))
        fc1_layers.append(nn.Sigmoid()) # 1 is for symbol presence

        self.fc1 = nn.Sequential(*fc1_layers)

        fc2_layers = []
        # 8*32*32 -> 1024
        fc2_layers.append(nn.Linear(8*32*32, 1024))
        fc2_layers.append(nn.ReLU())
        # 1024 -> 512
        fc2_layers.append(nn.Linear(1024, 512))
        fc2_layers.append(nn.ReLU())
        # 512 -> 256
        fc2_layers.append(nn.Linear(512, 256)) 
        fc2_layers.append(nn.ReLU())
        # 256 -> 4
        fc2_layers.append(nn.Linear(256, 4)) # 4 is for bounding box

        self.fc2 = nn.Sequential(*fc2_layers)

    def forward(self, x):
        x = x.to('cuda')
        for layer in self.conv_layers:
            x = layer(x)
        presence = x
        for layer in self.fc1:
            presence = layer(presence)
        bbox = x
        for layer in self.fc2:
            bbox = layer(bbox)
        return presence, bbox
    
    def loss(self, presence_pred, bbox_pred, presence_labels, bbox_labels, bbox_mask):
        beta = 0.8
        presence_loss = nn.BCELoss()(presence_pred, presence_labels) # value between 0 and 1
        bbox_loss = nn.SmoothL1Loss()(bbox_pred*bbox_mask, bbox_labels*bbox_mask)

        return presence_loss + beta * bbox_loss



def training_model(model, data_loader, val_loader, optimizer, num_epochs=10, device='cuda'):
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        model.train()
        running_loss = 0.0
        for images, presence_labels, bbox_labels in tqdm(data_loader):
            images = images.to(device)
            presence_labels = presence_labels.to(device)
            bbox_labels = bbox_labels.to(device)

            # Forward pass
            presence_pred, bbox_pred = model(images)

            bbox_mask = presence_labels.expand_as(bbox_labels)
            loss = model.loss(presence_pred, bbox_pred, presence_labels, bbox_labels, bbox_mask)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(data_loader.dataset)
        train_losses.append(epoch_train_loss)
        print(f'Epoch {epoch+1}/{num_epochs} Loss: {epoch_train_loss:.4f}')


        model.eval()
        running_val_loss = 0.0
    

        with torch.no_grad():
            for images, presence_labels, bbox_labels in val_loader:
                images = images.to(device)
                presence_labels = presence_labels.to(device)
                bbox_labels = bbox_labels.to(device)


                # Forward pass
                presence_pred, bbox_pred = model(images)

                bbox_mask = presence_labels.expand_as(bbox_labels)
                loss = model.loss(presence_pred, bbox_pred, presence_labels, bbox_labels, bbox_mask)
                
                # Accumulate loss
                running_val_loss += loss.item() * images.size(0)  # Loss per batch multiplied by batch size
        
        # Calculate average validation loss for the epoch
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f"Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")

    return model, train_losses, val_losses 



    