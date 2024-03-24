import src.common_function as common_function
from PIL import Image
from torch import nn, zeros, optim, tensor, cat, Tensor, save, load, log_softmax
from torchvision import transforms
from src.algorithm_split import split_image, get_letter_coord

class RNNTransitionDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNTransitionDetector, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 pour les sorties bidirectionnelles

    def forward(self, input):
        h0 = zeros(self.num_layers * 2, input.size(0), self.hidden_size).to(input.device)
        c0 = zeros(self.num_layers * 2, input.size(0), self.hidden_size).to(input.device)

        out, _ = self.lstm(input, (h0, c0))

        out = out[:, -1, :] # On récupère la dernière sortie de la séquence

        out = self.fc(out) # On passe la sortie dans une couche linéaire

        return out
    


# Paramètres du modèle
input_size = 1  # Taille de l'entrée (colonne de 1 pixel de large)
hidden_size = 128  # Taille de l'état caché
num_layers = 2  # Nombre de couches RNN
num_classes = 2  # Nombre de classes de sortie (transition ou non)

device = "cpu"

model = RNNTransitionDetector(input_size, hidden_size, num_layers, num_classes)

criterion = nn.CrossEntropyLoss(tensor([1.0,3.78]))

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
    torch_sequence = Tensor(len(columns),28,1) # Création d'un tenseur de taille (longueur de la séquence, 28, 1)
    cat(columns, out=torch_sequence) # Concaténation des colonnes dans le tenseur
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


def load_data():
    """
    Fonction qui charge les données d'apprentissage et de test
    @return liste de tenseurs d'images transformées et liste de tenseurs de labels
    """
    labels = []
    datas = []
    set_infos = open("dataset/word_sequenced/labels.txt", "r")
    for line in set_infos:
        if line[0] == "#":
            continue
        line = line.split(" ")
        img = Image.open("dataset/word_sequenced/images/" + line[0])
        torch_sequence = convert_image_to_tensor(img)
        datas.append(torch_sequence)
        label = []
        if len(line) > 3:
            for i in range(3, len(line)-1):
                tmp = line[i].split("-")
                if len(tmp) == 1:
                    label.append(int(tmp[0]))
                else:
                    label += range(int(tmp[0]), int(tmp[1]))
        label = convert_label_to_tensor(label, img.width)
        labels.append(label)
    return datas, labels



def learning(nbr_epoch=10, learning_rate=0.1):
    """
    Fonction qui lance l'apprentissage
    @param nbr_epoch nombre d'epoch d'apprentissage
    @param learning_rate taux d'apprentissage
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(nbr_epoch):
        epoch_loss = 0
        for i in range(len(datas)):
            data = datas[i].to("cpu")
            label = labels[i].to("cpu")
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print("Epoch : ", epoch, " Loss :" , epoch_loss/len(datas))

            

def predict_from_path(path):
    """
    Fonction appliquant le modèle à une image
    @param img image à tester
    @return un tenseur du nombre de colonne de l'image de 1 et 0. 1 : transition, 0 : pas de transition
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
    @return un tenseur du nombre de colonne de l'image de 1 et 0. 1 : transition, 0 : pas de transition
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
    pred = pred.numpy() # On convertit le tenseur en tableau numpy
    transition = (0,0) # Coordonnées de la transition [0] = début, [1] = fin
    in_transition = False # Booléen qui indique si on calcule la longueur d'une transition
    for i in range(len(pred)):
        if in_transition: # Si on calcule la longueur d'une transition
            if pred[i] == 1: # Si on est toujours dans une transition on incrémente la fin de la transition
                transition = (transition[0], i)
            else: # Sinon on a trouvé la fin de la transition
                in_transition = False
                split_coord.append(int(transition[0] + (transition[1] - transition[0])/2)) # La coordonnée de transition est au milieu de la transition
        else: 
            if pred[i] == 1: # Si on est pas dans une transition et qu'on en trouve une on commence à calculer la longueur de la transition
                in_transition = True
                transition = (i, i)
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
    else: # Si on a pas trouvé de transition on retourne l'image entière
        images.append(img)
    return images

def save_model(path='models/transition_detector.pth'):
    """
    Procédure de sauvegarde du modèle
    path : chemin du fichier de sauvegarde (par défaut : models/transition_detector.pth)
    """
    save(model.state_dict(), path)
    print("Model saved")

def load_model(path='models/transition_detector.pth'):
    """
    Procédure de chargement du modèle
    path : chemin du fichier de chargement (par défaut : models/transition_detector.pth)
    """
    model.load_state_dict(load(path))
    #print("Model loaded")

def get_model():
    """
    Fonction qui retourne le modèle entrainé
    """
    load_model('models/transition_detector3.pth')
    return model

def get_model_split_images(img):
    pred = predict_from_image(img)
    return get_images_from_pred(pred, img=img)

def eval_diff_split(img):
    pred = predict_from_image(img)



def test_num_letters_split(img_split_func = get_model_split_images):
    """
    Procédure de calcule du nombre de lettre lors du test du split d'une image
    """
    model.eval()
    set_infos = open("dataset/word_sequenced/labels.txt", "r")
    num_true = 0
    avg_diff = 0
    for line in set_infos:
        if line[0] == "#":
            continue
        color = "\033[91m"
        line = line.split(" ")
        img = Image.open("dataset/word_sequenced/images/" + line[0])
        images = img_split_func(img)
        num_letters_pred = len(images)
        num_letters_true = len(line) - 3
        if "0-" in line[3] or line[3] == "0":
            num_letters_true -= 1
        if num_letters_pred == num_letters_true:
            num_true += 1
            color = "\033[92m"
        else:
            avg_diff += abs(num_letters_pred - num_letters_true)
        print(color + "True : ", num_letters_true, " Pred : ", num_letters_pred, line[0])
    print("\033[0m")
    print("Avg diff : ", avg_diff/len(datas))
    print("True : ", num_true, " Total : ", len(datas), " Accuracy : ", (num_true/len(datas))*100, "%")

def compare_sequence(pred, label):
    """
    Fonction qui compare une séquence prédite à une séquence réelle
    @param pred séquence prédite
    @param label séquence réelle
    @return nombre de transitions correctes, nombre de transitions fausses, nombre de transitions manquées
    """
    nbr_correct = 0
    nbr_false = 0
    for elt1, elt2 in zip(pred, label):
        if elt1 == elt2:
            nbr_correct += 1
        else:
            nbr_false += 1
    return nbr_correct, nbr_false

def test_closest_coord_model():
    model.eval()
    avg_correct = 0
    avg_false = 0
    for i in range(len(datas)):
        data = datas[i].to("cpu")
        label = labels[i].to("cpu")
        output = model(data)
        nbr_correct, nbr_false = compare_sequence(output.argmax(1), label)
        avg_correct += nbr_correct/data.size(0)
        avg_false += nbr_false/data.size(0)
    print("Correct : ", avg_correct/len(datas), " False : ", avg_false/len(datas))

def unfold_label(label, size):
    for i in range(len(label)-1):
        tmp = label[i].split("-")
        if len(tmp) == 1:
            label.append(int(tmp[0]))
        else:
            label += range(int(tmp[0]), int(tmp[1]))
    result = []
    for i in range(size):
        if i in label:
            result.append(1)
        else:
            result.append(0)
    return result

def test_closest_coord_split():
    set_infos = open("dataset/word_sequenced/labels.txt", "r")
    avg_correct = 0
    avg_false = 0
    i = 0
    for line in set_infos:
        if line[0] == "#":
            continue
        line = line.split(" ")
        img = Image.open("dataset/word_sequenced/images/" + line[0])
        img = common_function.transform_image_black_white(img)
        coords = get_letter_coord(img)
        print(coords)
        i += 1
        if i > 10:
            break
        """
        pred = unfold_label(coords)
        nbr_correct, nbr_false = compare_sequence(pred, label)
        avg_correct += nbr_correct/len(pred)
        avg_false += nbr_false/len(pred)
        """
    print("Correct : ", avg_correct/len(datas), " False : ", avg_false/len(datas))


datas, labels = load_data()
load_model('models/transition_detector2.pth')
#test_num_letters_split() # Test du modèle
#print("--------------------------------------------------")
#test_num_letters_split(split_image) # Test de l'algorithme
#test_closest_coord_model()
#test_closest_coord_split()
#learning(20,0.00002)
#save_model('models/transition_detector2.pth')

