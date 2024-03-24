
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import ast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import ast
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import csv
import os

class TextDetectionModel(nn.Module):
    def __init__(self):
        super(TextDetectionModel, self).__init__()

        
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
       
        self.conv2 = nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(6, 10, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Couches entièrement connectées pour  la régression
        self.fc1 = nn.Linear(156250, 512)
        self.relu3 = nn.ReLU()
        self.fc3 = nn.Linear(512, 4)  # 4 valeurs pour les coordonnées (x, y, width, height)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)


        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Aplatir les caractéristiques pour les couches entièrement connectées
        x = x.view(x.size(0), -1)

        # Propagation avant à travers les couches entièrement connectées
        x = self.fc1(x)
       

        
       
        regression_output = self.fc3(x)

        return  regression_output
       
       

model = TextDetectionModel()


class CustomDataset(Dataset):
    def __init__(self, csv_file, image_folder):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transforms.Compose([
            transforms.Resize((500, 500)),
            transforms.ToTensor(),
        ])
   
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.iloc[idx, 0]
        image_path = os.path.join(self.image_folder, image_name)
       
       
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        coordinates = eval(self.data.iloc[idx, 1])  # Convertir la chaîne en tuple
        coordinates = [float(coord) for coord in coordinates]

        min_value = min(coordinates)
        max_value = max(coordinates)
        coordinates = [(coord - min_value) / (max_value - min_value) for coord in coordinates]


        sample = {
            'image': image,
            'coordinates': coordinates,
            
        }

        return sample

# Chemin vers  fichier CSV
csv_path = '/content/drive/MyDrive/Classroom/csv_file/coordinates.csv'
images_path = '/content/drive/MyDrive/Classroom/mes_images'


custom_dataset = CustomDataset(csv_path, images_path)


dataloader = DataLoader(custom_dataset, batch_size=128, shuffle=True)


optimizer = optim.Adam(model.parameters(), lr=0.0001)

#-------------------------------------------------
checkpoint_path = '/content/drive/MyDrive/Classroom/train_ckeck/model.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    # Ajoutez d'autres éléments que vous avez sauvegardés

    print(f"Chargement du point de contrôle à l'époque {epoch} avec une perte de {loss}")
else:
    print("Aucun point de contrôle trouvé, l'entraînement commencera depuis le début.")

#-------------------------------------------------







train_losses = []
iteration = 0
for epoch in range(10):
    model.train()

    for batch in dataloader:
       
        images = batch['image']
        coordinates = batch['coordinates']
        """ print(f'Type of images: {type(images)}')
        print(batch)
        print(type(batch))
        print("images = ", type(images))
        print("coordinates = ", type(coordinates))"""

        optimizer.zero_grad()

        
        regression_output = model(images)

        # Calcul de la perte
        #print(type(regression_output))
        #print(regression_output)
       
        coordinates = torch.stack([coord.float() for coord in coordinates], dim=1)
        #print(coordinates.shape)
        #print(coordinates)
        loss =  nn.MSELoss()(regression_output, coordinates)

        # Rétropropagation et mise à jour des poids
        loss.backward()
        optimizer.step()
        train_losses.append(loss)
       

    print(f'Epoch {epoch+1}/{5}, Iteration {iteration}, Loss: {loss.item()}')

    iteration += 1

checkpoint_path = '/content/drive/MyDrive/Classroom/train_ckeck/model.pth'
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
  
}, checkpoint_path)
plt.plot([loss.item() for loss in train_losses], label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()
drive.mount('/content/drive')