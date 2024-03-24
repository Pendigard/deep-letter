import common_function as common_function
import numpy as np
import torch
import random
from torch import nn, optim, zeros, tensor, cat, Tensor, save, load
from torchvision import transforms

from PIL import Image

class RNNWord(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNWord, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input):
        h0 = zeros(self.num_layers * 2, input.size(0), self.hidden_size).to(input.device)
        c0 = zeros(self.num_layers * 2, input.size(0), self.hidden_size).to(input.device)

        out, _ = self.lstm(input, (h0, c0))

        out = out[:, -1, :]

        out = self.fc(out) 

        return out

input_size = 1
hidden_size = 128
num_layers = 2
num_classes = 2

model = RNNWord(input_size, hidden_size, num_layers, num_classes)

optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 200.0]))
transform = transforms.Compose([ transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

def adapt_image_column(img,x):
    """
    Fonction qui applique les transformations nécessaires à une colonne d'une image
    @param img image à transformer
    @param x abscisse de la colonne à transformer
    @return colonne de l'image transformée
    """
    img = img.convert('L')
    img = img.crop((x, 0, x+1, img.height))
    img = img.resize((1,28))
    img = common_function.transform_image_black_white(img)
    img = transform(img)
    return img


def convert_image_to_tensor(img):
    """
    Fonction qui convertit une image en un tenseur de colonnes
    @param img image à convertir
    @return tenseur des colonnes de l'image
    """
    columns = []
    for i in range(img.width):
        columns.append(adapt_image_column(img,i))
    torch_sequence = Tensor(len(columns),28,1)
    cat(columns, out=torch_sequence)
    return torch_sequence

def convert_label_to_tensor(label, size):
    """
    Fonction qui convertit une liste de labels en un tenseur
    @param label liste de labels
    @param size taille du tenseur
    @return tenseur de la liste de labels
    """
    result = []
    for i in range(size):
        if i in label:
            result.append(1)
        else:
            result.append(0)
    result = tensor(result)
    return result



def load_data(b):
    """
    Fonction qui charge les données d'apprentissage et de test
    @return liste de tenseurs d'images transformées et liste de tenseurs de labels
    """
    labels = []
    datas = []
    infos = []
    x = 1 + random.randint(0, 11520)
    file = open("dataset/IAM/ascii/words.txt", "r")
    for _ in range(x):
        target = next(file, None) 
        while target is not None:
            next_line = next(file, None)

            if next_line is not None and next_line.startswith(target.split(" ")[0]):
                target = next_line
            else:
                break

    for _ in range(b):
        line = next(file, None)
        if line is not None:
            if line[0] == "#":
                continue
            line = line.split(" ")
            dir = line[0].split("-")
            img = Image.open("dataset/IAM/lines/" + dir[0] + "/" + dir[0]+"-"+dir[1] + "/" + dir[0]+"-"+dir[1]+"-"+dir[2]+".png")
            torch_sequence = convert_image_to_tensor(img)
            datas.append(torch_sequence)
            
            label = []

            if int(line[3]) > 0 and int(line[5]) > 0:
                for i in range(int(line[3]), int(line[3])+int(line[5]), 1):
                    label.append(i)
                    
                infos.append(label)
                label = convert_label_to_tensor(label, img.width)
            labels.append(label)
    
    return datas, labels, infos

def learning(nbr_epoch=10, learning_rate=0.1):
    """
    Fonction qui lance l'apprentissage
    @param nbr_epoch nombre d'epoch d'apprentissage
    @param learning_rate taux d'apprentissage
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(nbr_epoch):
        for i in range(len(datas)):
            data = datas[i].to("cpu")
            label = labels[i].to("cpu")
            optimizer.zero_grad()
            output = model(data)
            ##print(i)
            ##print("out : " + str(output))
            ##print("lb : ".join(map(str, label)))
            loss = criterion(output, label)
            
            l1_regularization = 0.05
            l1_loss = 0
            for param in model.parameters():
                l1_loss += torch.abs(param).sum()
            total_loss = loss + l1_regularization * l1_loss

            total_loss.backward()
            optimizer.step()
            print("Epoch : ", epoch, " Image : ", i, " Loss :" , loss.item(), " Word : ", 1 in output.argmax(1).numpy())

def predict_from_path(path):
    """
    Fonction appliquant le modèle à une image
    @param img image à tester
    @return un tenseur du nombre de colonne de l'image de 1 et 0. 1 : word, 0 : pas de word
    """
    img = Image.open(path)
    img_tensor = convert_image_to_tensor(img)
    model.eval()
    output = model(img_tensor)
    return output.argmax(1)

def predict_from_image(img):
    """
    Fonction appliquant le modèle à une image
    @param img image à tester
    @return un tenseur du nombre de colonne de l'image de 1 et 0. 1 : word, 0 : pas de word
    """
    img_tensor = convert_image_to_tensor(img)
    model.eval()
    output = model(img_tensor)
    return output.argmax(1)

def get_images_from_pred(pred, path=None, img=None):
    """
    Retourne une liste d'images de lettres à partir d'une prédiction du modèle
    @param pred prédiction du modèle
    @param path chemin de l'image à découper
    @return liste d'images découpées
    """
    images = []
    split_coord = []
    if path is not None:
        img = Image.open(path)
    pred = pred.numpy()
    word = (0,0) 
    in_word = False 
    for i in range(len(pred)):
        if in_word: 
            if pred[i] == 1: 
                word = (word[0], i)
            else: 
                in_word = False
                split_coord.append(int(word[0] + (word[1] - word[0])/2)) 
        else: 
            if pred[i] == 1: 
                in_word = True
                word = (i, i)
    if len(split_coord) > 0:
        if split_coord[0] != 0:
            split_coord.insert(0, 0)
        if split_coord[-1] != img.width:
            split_coord.append(img.width)
        for i in range(len(split_coord)-1):
            if split_coord[i+1] - split_coord[i] > img.height*0.25: # On vérifie que l'image n'est pas trop petite
                images.append(img.crop((split_coord[i], 0, split_coord[i+1], img.height)))
            else: # Si l'image est trop petite on la fusionne à la suivante
                split_coord[i+1] = split_coord[i]
    else: # Si on a pas trouvé de word on retourne l'image entière
        images.append(img)
    return images

def center_letter(img, word_width, word_height):
    """
    Fonction qui centre une lettre dans une image
    img : image à centrer
    Retourne l'image centrée
    """
    box = img.getbbox()
    if box is None:
        return img
    img = img.crop(box)

    width = img.width
    height = img.height

    side_size = int(max(width, height) * 1.5)
    new_image = Image.new("L", (side_size, side_size), 0)
    new_image.paste(img, (int((side_size - width)/2), int((side_size - height)/2))) 
    
    #new_image.show()
    return new_image

def save_model(path='models/word_detector.pth'):
    save(model.state_dict(), path)
    print("Model saved")

def load_model(path='models/word_detector.pth'):
    print("Loading model...")
    model.load_state_dict(load(path))
    print("Model loaded")

def cut_image_based_on_predictions(img, predictions):
    """
    Coupe l'image en plusieurs petites images en utilisant les groupes de 1 dans la prédiction.
    @param img: Image à découper
    @param predictions: Tenseur de prédictions
    @return: Liste d'images découpées
    """

    predictions_array = predictions.numpy()
    print("lb : ".join(map(str, predictions_array)))

    word_indices = np.where((predictions_array[:-1] == 0) & (predictions_array[1:] == 1))[0]
    print(word_indices)

    images = []
    start_idx = 0

    for end_idx in word_indices:
        cropped_img = img.crop((start_idx, 0, end_idx, img.height))
        
        images.append(cropped_img)

        start_idx = end_idx

    images.append(img.crop((start_idx, 0, img.width, img.height)))

    return images

"""
path_to_image = "dataset/IAM/lines/a01/a01-000u/a01-000u-00.png"
#load_model()

datas, labels, infos = load_data(20)
learning(1,0.1)
#save_model()

img = Image.open(path_to_image)
predictions = predict_from_image(img)
cropped_images = cut_image_based_on_predictions(img, predictions)

for i, cropped_img in enumerate(cropped_images):
    if cropped_img.width > 0 and cropped_img.height > 0:
        cropped_img.show()
    else:
        print(f"Image {i} has invalid size.")
"""



