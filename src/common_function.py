from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import load, save
import numpy as np
from PIL import Image

batch_size_train = 64
batch_size_test = 128

class Add_Noise(object):
    def __init__(self, noise_factor=0.2):
        self.noise_factor = noise_factor

    def __call__(self, img):
        self.noise_factor = np.random.uniform(0, self.noise_factor)
        img_array = np.array(img)
        noise = np.random.normal(loc=0, scale=self.noise_factor, size=img_array.shape)
        noisy_img = img_array + noise
        noisy_img = np.clip(noisy_img, 0, 255)
        noisy_img = Image.fromarray(noisy_img)
        return noisy_img

class Convert_Black_White(object):
    def __init__(self):
        pass
    def __call__(self, img):
        return transform_image_black_white(img)
    
class Convert_resize_center(object):
    def __init__(self):
        pass
    def __call__(self, img):
        return center_letter(img)
    
class Convert_Black_White_RGB(object):
    def __init__(self):
        pass
    def __call__(self, img):
        return img.convert('RGB')
    

transformBW=transforms.Compose([
                               Convert_Black_White(),
                               #Add_Noise(0.1),
                               #transforms.RandomRotation(10),
                               Convert_resize_center(),
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.5,), (0.5,))
                             ])

transformVGG = transforms.Compose([
    Convert_Black_White_RGB(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    
def transform_image_black_white(img):
    img = img.convert('L')
    # On obtient les pixels des coins de l'image
    top_left = img.getpixel((0,0))
    top_right = img.getpixel((img.size[0]-1,0))
    bottom_left = img.getpixel((0,img.size[1]-1))
    bottom_right = img.getpixel((img.size[0]-1,img.size[1]-1))
    # Calcul de la couleur moyenne des coins
    mean_color = (top_left + top_right + bottom_left + bottom_right) / 4
    if mean_color < 128:
        img = img.point(lambda x: 0 if x < 128 else 255, '1')
    else:
        img = img.point(lambda x: 255 if x < 128 else 0, '1')
    
    return img

def center_letter(img):
    """
    Fonction qui centre une lettre dans une image
    img : image à centrer
    Retourne l'image centrée
    """
    box = img.getbbox()
    if box is None:
        return img
    
    #ajoute un contour noir à l'image pour centrer la lettre
    img = img.crop(box)

    width = img.width
    height = img.height

    side_size = int(max(width, height) * 1.5)
    new_image = Image.new("L", (side_size, side_size), 0) # On crée une image noire de taille side_size x side_size
    new_image.paste(img, (int((side_size - width)/2), int((side_size - height)/2))) # On colle l'image au centre de l'image noire

    new_image = new_image.resize((28,28))
    
    #new_image.show()
    return new_image

def delete_background(img):
    """
    Fonction qui supprime le fond noir d'une image
    img : image à modifier
    Retourne l'image sans le fond noir
    """
    img = img.convert('RGBA')
    datas = img.getdata()
    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    return img

def get_emnist_sets(split='byclass', transform=transformBW):
    """
    Fonction qui retourne les datasets de emnist
    split : split du dataset (par défaut : byclass tout les caractères)
    """
    emnist_dataset_train = datasets.EMNIST('dataset/', train=True, download=True, transform=transform, split=split)
    emnist_dataset_test = datasets.EMNIST('dataset/', train=False, download=True, transform=transform, split=split)
    return emnist_dataset_train, emnist_dataset_test



def get_emnist_char_loader(split='byclass', transform=transformBW):
    """
    Fonction qui retourne les dataloader des datasets de emnist
    split : split du dataset (par défaut : byclass tout les caractères)
    """
    emnist_dataset_train, emnist_dataset_test = get_emnist_sets(split,transform)
    train_loader = DataLoader(emnist_dataset_train ,batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(emnist_dataset_test ,batch_size=batch_size_test, shuffle=True)
    return train_loader, test_loader

def get_picture_right(img):
    """
    Fonction qui retourne l'image dans le bon sens et transposée symétriquement par rapport à l'axe vertical
    Par défaut dans le set emnist les images tournée à 90° et transposée symétriquement par rapport à l'axe vertical
    img : image à transformer
    return : image dans le bon sens et transposée symétriquement par rapport à l'axe vertical
    """
    img = img.rotate(-90)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def save_model(model, path):
    """
    Fonction qui sauvegarde un modèle
    model : modèle à sauvegarder
    path : chemin de sauvegarde
    """
    save(model.state_dict(), path)

def load_model(model, path):
    """
    Fonction qui charge un modèle
    model : modèle à charger
    path : chemin de chargement
    return : modèle chargé
    """
    model.load_state_dict(load(path))
    return model

