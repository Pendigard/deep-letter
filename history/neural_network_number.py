import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image, ImageOps
import numpy as np



batch_size_train = 64
batch_size_test = 1000

transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])

train_loader = torch.utils.data.DataLoader(
  datasets.MNIST('dataset/', train=True, download=True,
                             transform=transform),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  datasets.MNIST('dataset/', train=False, download=True,
                             transform=transform),
  batch_size=batch_size_test, shuffle=True)


# Change l'appareil en fonction de la disponibilité de chaque carte graphique
device = 'cpu'

(
    'cuda' 
    if torch.cuda.is_available() 
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding="same"),  # 1 input channel, 16 output channels, kernel_size=3x3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling layer
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16*14*14, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)


loss_fn = nn.CrossEntropyLoss()

def learning(nbr_epoch, learning_rate=0.1):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in tqdm(range(nbr_epoch)):
        model.train() # Indique que le modèle est en mode apprentissage
        for batch, (X, y) in tqdm(enumerate(train_loader)):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} : Loss {loss.item()}")

def test_dataset():
    test_loss, correct = 0, 0
    with torch.no_grad(): # Empêche le calcul du gradient
        for X, y in test_loader: # X = image, y = label
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= len(test_loader.dataset)
    correct /= len(test_loader.dataset)
    print(f"Test Error : \n Accuracy : {(100*correct):>0.1f}%, Avg loss : {test_loss:>8f} \n")


def predict(img):
    model.eval()
    with torch.no_grad(): # Empêche le calcul du gradient
        img = img.to(device)
        pred = model(img)
        return pred.argmax(1)

def save_model():
    torch.save(model.state_dict(), 'history/model.pth')
    print("Model saved")

def load_model():
    model.load_state_dict(torch.load('history/model.pth'))
    print("Model loaded")

def load_image(path):
    img = Image.open(path).convert('L')
    # Resize
    img = img.resize((28,28))
    img = transform(img)
    return img

def load_image_invert(path):
    img = Image.open(path).convert('L')
    # Resize
    img = img.resize((28,28))
    img = ImageOps.invert(img)
    img = img.point(lambda x: 0 if x<128 else 255, '1')
    #img.show()
    img = transform(img)
    return img

def load_test_images():
    images = []
    labels = []
    images.append(load_image("img_test/8B.png"))
    labels.append(8)
    images.append(load_image("img_test/7E.png"))
    labels.append(7)
    images.append(load_image("img_test/7C.png"))
    labels.append(7)
    images.append(load_image("img_test/7D.png"))
    labels.append(7)
    images.append(load_image("img_test/7_bien.jpg"))
    labels.append(7)
    images.append(load_image_invert("img_test/7H.jpg"))
    labels.append(7)
    images.append(load_image("img_test/3S.png"))
    labels.append(3)
    images.append(load_image("img_test/3.jpg"))
    labels.append(3)
    images.append(load_image("img_test/4.jpg"))
    labels.append(4)
    images.append(load_image("img_test/4C.png"))
    labels.append(4)
    images.append(load_image("img_test/9.png"))
    labels.append(9)
    images.append(load_image("img_test/5B.png"))
    labels.append(5)
    for i in range(10):
        images.append(load_image(f"img_test/{i}b.png"))
        labels.append(i)
    return images, labels

def test_images():
    images, labels = load_test_images()
    nbr_good_pred = 0
    for i in range(len(images)):
        img_tensor = torch.tensor(images[i])
        img_tensor = torch.reshape(img_tensor,(1,1,28,28))
        pred = predict(img_tensor)
        print(f"Image {i} : Prediction : {pred.item()}, Label : {labels[i]}")
        if pred.item() == labels[i]:
            nbr_good_pred += 1
    print(f"Accuracy : {(nbr_good_pred/len(images))*100:>0.1f}%")
   

def show_random_image():
    img = test_loader.dataset[0][0]
    img = img*0.3081 + 0.1307
    img = img.reshape(28,28)
    img = img.numpy()
    img = np.array(img*255, dtype=np.uint8)
    img = Image.fromarray(img)
    #img.show()

def test_number():
    img = load_image("img_test/1284.png")
    predictions = []
    for i in range(0,img.shape[2]-28,2):
        crop_img = img[:,:,i:i+28]
        img_tensor = torch.tensor(crop_img)
        img_tensor = torch.reshape(img_tensor,(1,1,28,28))
        pred = predict(img_tensor)
        predictions.append(pred.item())
    print(predictions)


def main(learn=True, learning_rate=0.1, nbr_epoch=10, load=False, save=False):
    if load:
        load_model()
    if learn:
        learning(nbr_epoch,learning_rate)
    test_dataset()
    test_images()
    #test_number()
    if save:
        save_model()

#show_random_image()
main(learn=False, learning_rate=0.1, nbr_epoch=2, load=True, save=False)
#show_random_image()
