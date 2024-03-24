
from torch import nn, where, tensor, reshape, cuda, randn
from PIL import Image
import src.neural_network as neural_network
import numpy as np
from random import randint
import matplotlib.pyplot as plt
import src.common_function as common_function
from torchinfo import summary


OKGREEN_COLOR = '\033[92m' # Vert pour les bonnes prédictions dans le test
FAIL_COLOR = '\033[91m' # Rouge pour les mauvaises prédictions dans le test


def load_image_tensor(path=None, img=None):
    """
    Fonction qui charge une image et la transforme en tensor
    path : chemin de l'image
    Retourne le tensor de l'image
    """    
    if path != None:
        img = Image.open(path).convert('L')
    
    img = common_function.transform_image_black_white(img)
    # Resize
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = img.rotate(90)
    img = img.resize((28,28))
    img = common_function.transformBW(img)
    img = reshape(img,(1,1,28,28))
    return img
    

def load_test_images():
    """
    Fonction qui charge les images de test et les labels associés
    Retourne une liste d'images et une liste de labels
    """
    images = []
    labels = []    
    images.append(load_image_tensor("img_test/3S.png"))
    labels.append('3')
    images.append(load_image_tensor("img_test/8B.png"))
    labels.append('8')
    images.append(load_image_tensor("img_test/7E.png"))
    labels.append('7')
    images.append(load_image_tensor("img_test/7C.png"))
    labels.append('7')
    images.append(load_image_tensor("img_test/7D.png"))
    labels.append('7')
    images.append(load_image_tensor("img_test/7_bien.jpg"))
    labels.append('7')
    #images.append(load_image_invert("img_test/7H.jpg"))
    #labels.append(7)
    images.append(load_image_tensor("img_test/3S.png"))
    labels.append('3')
    images.append(load_image_tensor("img_test/3.jpg"))
    labels.append('3')
    images.append(load_image_tensor("img_test/4.jpg"))
    labels.append('4')
    images.append(load_image_tensor("img_test/4C.png"))
    labels.append('4')
    images.append(load_image_tensor("img_test/9.png"))
    labels.append('9')
    images.append(load_image_tensor("img_test/5B.png"))
    labels.append('5')
    images.append(load_image_tensor("img_test/5B.png"))
    labels.append('5')
    for i in range(10):
        images.append(load_image_tensor(f"img_test/{i}b.png"))
        labels.append(str(i))
    images.append(load_image_tensor("img_test/A.png"))
    labels.append('A')
    images.append(load_image_tensor("img_test/A2.png"))
    labels.append('A')
    images.append(load_image_tensor("img_test/W.png"))
    labels.append('W')
    images.append(load_image_tensor("img_test/m.jpg"))
    labels.append('m')
    images.append(load_image_tensor("img_test/v.png"))
    labels.append('V')

    return images, labels

def label_to_char(label):
    """
    Fonction qui transforme un label en caractère
    label : label à transformer
    Retourne le caractère associé au label
    """
    if label >= 0 and label <= 9:
        return chr(label + ord('0'))
    elif label >= 10 and label <= 35:
        return chr(label + ord('A') - 10)
    elif label >= 36 and label <= 61:
        return chr(label + ord('a') - 36)
    else:
        return "?"
    
def tensor_label_to_char_list(tensor_label):
    """
    Fonction qui transforme un tensor de label en tensor de caractères
    tensor_label : tensor de label à transformer
    Retourne le tensor de caractères associé au tensor de label
    """
    tensor_char = []
    for label in tensor_label:
        tensor_char.append(label_to_char(label))
    return tensor_char

def pred_to_char_dict(pred,limit=None):
    """
    Fonction qui transforme un tensor de label en dictionnaire de caractères
    tensor_label : tensor de label à transformer
    Retourne le dictionnaire de caractères associé à la probabilité de prédiction
    """
    if limit != None:
        values, indices = pred.flatten().topk(limit)
    else:
        values, indices = pred.flatten().topk(pred.flatten().shape[0])
    dict_char = {}
    for i in range(len(indices)):
        dict_char[label_to_char(indices[i].item())] = values[i].item()
    return dict_char

def test_images(model):
    """
    Procédure qui teste les images de test externes au dataset
    model : modèle à tester
    """
    images, labels = load_test_images()
    nbr_good_pred = 0
    for i in range(len(images)):
        pred = model.predict(images[i])
        highest_prob = pred.argmax(1)
        result = FAIL_COLOR
        if label_to_char(highest_prob.item()) == labels[i]:
            result = OKGREEN_COLOR
            nbr_good_pred += 1
        else:
            print(f"{FAIL_COLOR} {tensor_label_to_char_list(pred.flatten().topk(3).indices)}") # Affiche les 3 plus grandes probabilités de prédiction
        result += f"Image {i} : Prediction : {label_to_char(highest_prob.item())}, Label : {labels[i]}"
        print(result)
    print(f"{OKGREEN_COLOR}Accuracy : {(nbr_good_pred/len(images))*100:>0.1f}%")

def init_nn_id_char_type(name="byclass"):
    """
    Fonction qui initialise le réseau de neurones pour identifier les caractères
    """
    nbr_output = 62
    if name == "very_strong_byclass":
        conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 1 input channel, 16 output channels, kernel_size=3x3
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 1 input channel, 16 output channels, kernel_size=3x3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        #print(summary(conv_layers, input_size=(1, 28, 28))) # Affiche le résumé du modèle

        linear_relu_stack = nn.Sequential(
            nn.Linear(32 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(64, nbr_output)
            )
    elif name == "strong_byclass" or name == "strong_byclass2":
        conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 1 input channel, 16 output channels, kernel_size=3x3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 1 input channel, 16 output channels, kernel_size=3x3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        linear_relu_stack = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(64, nbr_output)
            )
    elif name == "byclass":
        conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 1 input channel, 16 output channels, kernel_size=3x3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        linear_relu_stack = nn.Sequential(
            nn.Linear(16 * 14 * 14, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(32, nbr_output)
            )

    loss_fn = nn.CrossEntropyLoss()
    neural_net = neural_network.NeuralNetwork(linear_relu_stack, conv_layers)
    device = 'cuda' if cuda.is_available() else 'cpu'
    model = neural_network.Model(neural_net, loss_fn, device=device)
    return model

def show_image_from_set(index=None):
    """
    Procédure qui affiche une image aléatoire du dataset de test
    """
    _, test_loader = common_function.get_emnist_char_loader()
    if index == None:
        index = randint(0, len(test_loader.dataset))
        print(index)
    img = test_loader.dataset[index][0]
    print(label_to_char(test_loader.dataset[index][1]))
    print(test_loader.dataset[index][1])
    img = img*0.5 + 0.5
    img = img.reshape(28,28)
    img = img.numpy()
    img = np.array(img*255, dtype=np.uint8)
    img = Image.fromarray(img)
    img.show()

def show_loss(loss_data):
    """
    Procédure qui affiche la courbe de perte
    loss_data : liste des pertes par batch
    """
    for i in loss_data:
        plt.plot(i)
        plt.show()

def main(learn=True, learning_rate=0.075, nbr_epoch=10, load=False, save=True, model_name="byclass", pathLoad='models/character_models/model.pth', pathSave='models/character_models/model.pth'):
    """
    Procédure principale qui permet d'entrainer le réseau de neurones et de le tester
    learn : booléen qui permet de savoir si on veut entrainer le réseau de neurones (True par défaut)
    learning_rate : taux d'apprentissage du réseau de neurones (0.075 par défaut)
    nbr_epoch : nombre d'epoques d'entrainement (10 par défaut)
    load : booléen qui permet de savoir si on veut charger un modèle déjà entrainé (False par défaut)
    save : booléen qui permet de savoir si on veut sauvegarder le modèle entrainé (True par défaut)
    """
    model = init_nn_id_char_type(model_name)
    train_loader, test_loader = common_function.get_emnist_char_loader("byclass")
    if load:
        model.load_model(pathLoad)
    if learn:
        loss_data = model.learn(train_loader, nbr_epoch=nbr_epoch, learning_rate=learning_rate)
        show_loss(loss_data)
        if save:
            model.save_model(pathSave)
    
    model.test_model(test_loader)
    test_images(model)

def get_model(name='model'):
    """
    Fonction qui retourne le modèle entrainé
    """
    model = init_nn_id_char_type(name)
    model.load_model(f'models/character_models/{name}.pth')
    return model

#main(learn=False, learning_rate=0.001, nbr_epoch=2, load=True, save=False, model_name="strong_byclass", pathLoad='models/character_models/strong_byclass2.pth', pathSave='models/character_models/strong_byclass2.pth')
#main(learn=False, learning_rate=0.0001, nbr_epoch=1, load=True, save=False, model_name="strong_byclass", pathLoad='models/character_models/strong_byclass.pth', pathSave='models/character_models/strong_byclass.pth')

"""
for i in range(10):
    show_image_from_set()
"""