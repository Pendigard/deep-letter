import src.model_character as model_character
import src.common_function as common_function
from PIL import Image
import numpy as np


def center_letter(img, word_width, word_height):
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
    
    #new_image.show()
    return new_image

def get_letter_coord(img):
    """
    Fonction qui découpe une image avec un mot et récupère les coordonnées des lettres
    img : image à analyser
    Retourne une liste de tuples (x1, x2) où x1 et x2 sont les coordonnées d'abscisse
    du début et de la fin d'une lettre
    """  
    transition_x_begin = 0 # Coordonnée x du début de la liaison entre deux lettres
    letter_x_begin = -1 # Coordonnée x du début d'une lettre
    # La valeur par défaut est -1 parce que l'image peut commencer par un espace vide
    in_letter = False # Indique si on est dans une lettre ou non (sinon on est dans une liason entre deux lettres)
    letters_x_coord = [] # Liste des coordonnées des lettres
    step = 1 # Pas de la boucle
    for i in range(0,img.width-step,step):
        column = img.crop((i, 0, i+step, img.height)) # On récupère la colonne i de l'image de largeur 1
        box = column.getbbox() # On récupère la boîte englobante de la colonne
        #print(box)
        if box is None or (box[3] - box[1] < img.height * 0.12): # Si la boîte englobante est vide ou trop petite on est dans une liaison entre deux lettres
            if in_letter: # Si on était dans une lettre on a trouvé la fin de la lettre
                transition_x_begin = i # on commence à calculer la longueur de la liaison
                in_letter = False
        else: # Sinon on est dans une lettre
            if not(in_letter):
                x_end_of_letter = i - (i - transition_x_begin)/2 # La fin de la lettre est au milieu de la liaison
                if letter_x_begin != -1: # Si on est pas au début de l'image on a trouvé le début d'une lettre
                    letters_x_coord.append((letter_x_begin, x_end_of_letter))
                letter_x_begin = x_end_of_letter # On commence à calculer la longueur de la nouvelle lettre
                in_letter = True
    letters_x_coord.append((letter_x_begin, img.width - (img.width - transition_x_begin)/2))
    return letters_x_coord


def guess_word_type(word):
    num_char = 0
    num_numbers = 0
    for c in word:
        if c in "0123456789":
            num_numbers += 1
        else:
            num_char += 1

    total = num_char + num_numbers
    if num_numbers / total > 0.8:
        return "number"
    else:
        return "word"

def get_word_from_pred(preds, type):
    """
    Fonction qui devine un mot à partir d'un tenseur de probabilités
    preds : tenseur de probabilités
    type : type du mot (nombre ou mot)
    Retourne le mot deviné
    """
    result = ""
    range_min = 10
    range_max = 62
    if type == "number":
        range_min = 0
        range_max = 9
    for i in preds:
        top_correct_char = i.argmax(1).item()
        nbr_try = 0
        while top_correct_char not in range(range_min,range_max):
            nbr_try += 1
            top_correct_char = i.flatten().topk(nbr_try).indices.data[-1].item()
        result += model_character.label_to_char(top_correct_char)
    return result
        
def guess_word(pred, word):
    type_word = guess_word_type(word)
    result = get_word_from_pred(pred, type_word)
    result = result.capitalize()
    return result
    
def recognition(path, model):
    """
    Fonction qui devine un mot à partir d'une image
    img : image à deviner
    Retourne le mot deviné
    """
    word = ""
    pred_chars = []
    img = Image.open(path)
    img = common_function.transform_image_black_white(img)
    letters_x_coord = get_letter_coord(img)
    for coord in letters_x_coord:
        if coord[1] - coord[0] < img.width * 0.05:
            continue
        letter = img.crop((coord[0], 0, coord[1], img.height))
        letter_centered = center_letter(letter, img.width, img.height)
        if letter_centered is None:
            continue
        letter_centered = model_character.load_image_tensor(img=letter_centered)
        pred = model.predict(letter_centered)
        pred_chars.append(pred)
        #print(model_character.tensor_label_to_char_list(pred.flatten().topk(3).indices))
        word += model_character.label_to_char(pred.argmax(1).item())
    word = guess_word(pred_chars, word)
    return word

def split_image(img):
    img = common_function.transform_image_black_white(img)
    letters_x_coord = get_letter_coord(img)
    letters = []
    for coord in letters_x_coord:
        if coord[1] - coord[0] < img.width * 0.05:
            continue
        letter = img.crop((coord[0], 0, coord[1], img.height))
        letter_centered = center_letter(letter, img.width, img.height)
        if letter_centered is None:
            continue
        letters.append(letter_centered)
    return letters




def main():
    model = model_character.get_model("model")
    modelByClass = model_character.get_model("byclass")
    #print(recognition("img_test/python.jpg",model))
    print(recognition("img_test/text.jpg",modelByClass))

#main()

