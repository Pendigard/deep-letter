import torch
import torch.nn as nn
from torchvision.models import vgg16
import common_function as common_function
from torchinfo import summary
from tqdm import tqdm

class EMNIST_VGG(nn.Module):
    def __init__(self, num_classes=62):
        super(EMNIST_VGG, self).__init__()
        self.conv_VGG16_8 = vgg16(pretrained=True).features[:5]
        print(summary(self.conv_VGG16_8, input_size=(3, 224, 224))) # Affiche le résumé du modèle
        for param in self.conv_VGG16_8.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(64*112*112, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.conv_VGG16_8(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = EMNIST_VGG()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.000025, momentum=0.9)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        for batch, (X, y) in tqdm(enumerate(train_loader, 0)): # tqdm permet d'afficher une barre de progression
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            print('[%d, %5d] loss: %.3f' %(epoch + 1, batch + 1, loss.item()))
            if batch % 10 == 0:
                common_function.save_model(model, "models/model_character_vgg.pth")

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in tqdm(test_loader):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            print('Accuracy of the network on the test images: %d %%' %(100 * correct / total))
    print('Accuracy of the network on the test images: %d %%' %(100 * correct / total))

train_loader, test_loader = common_function.get_emnist_char_loader(transform=common_function.transformVGG)
model = common_function.load_model(model, "models/model_character_vgg.pth")
for param in model.conv_VGG16_8.parameters():
    param.requires_grad = False
test(model, test_loader)
#train(model, criterion, optimizer, epochs=1)
#common_function.save_model(model, "models/model_character_vgg.pth")

"""
code présent dans common_function.py

class Convert_Black_White_RGB(object):
    def __init__(self):
        pass
    def __call__(self, img):
        return img.convert('RGB')

transformVGG = transforms.Compose([
    Convert_Black_White_RGB(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_emnist_sets(split='byclass', transform=transformBW):
    #Fonction qui retourne les datasets de emnist
    #split : split du dataset (par défaut : byclass tout les caractères)
    emnist_dataset_train = datasets.EMNIST('dataset/', train=True, download=True, transform=transform, split=split)
    emnist_dataset_test = datasets.EMNIST('dataset/', train=False, download=True, transform=transform, split=split)
    return emnist_dataset_train, emnist_dataset_test



def get_emnist_char_loader(split='byclass', transform=transformBW):
    #Fonction qui retourne les dataloader des datasets de emnist
    #split : split du dataset (par défaut : byclass tout les caractères)
    emnist_dataset_train, emnist_dataset_test = get_emnist_sets(split,transform)
    train_loader = DataLoader(emnist_dataset_train ,batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(emnist_dataset_test ,batch_size=batch_size_test, shuffle=True)
    return train_loader, test_loader

"""