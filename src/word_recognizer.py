import src.model_character as model_character
import src.common_function as common_function
import src.model_transition as model_transition
import src.word_searcher as word_searcher

def split_wrong_letters(img):
    """
    Fonction qui découpe une image avec un mot et récupère les coordonnées des lettres
    img : image à analyser
    Retourne une liste d'image de lettres
    """  
    space_x_begin = 0 # Coordonnée x du début de la liaison entre deux lettres
    in_space = False # Indique si on est dans une lettre ou non (sinon on est dans une liason entre deux lettres)
    letters = [] # Liste des coordonnées des lettres
    for x in range(img.width):
        column = img.crop((x, 0, x+1, img.height))
        box = column.getbbox()
        if box is None:
            if not(in_space):
                space_x_begin = x
                in_space = True
        else:
            if in_space:
                letters.append(img.crop((space_x_begin, 0, x, img.height)).getbbox())
                in_space = False
    if len(letters) == 0:
        letters.append(img)
    return letters


def get_word(path=None,img=None,lang='fr_FR',model="strong_byclass", show=False):
    if img != None:
        pred = model_transition.predict_from_image(img)
    elif path != None:
        pred = model_transition.predict_from_path(path)
    else:
        raise Exception("You must give a path or an image")
    images = model_transition.get_images_from_pred(pred, path, img) # On récupère les images découpées par le modèle de transition
    tensors = []
    model_char = model_character.get_model(model)
    for i in range(len(images)):
        images[i] = common_function.transform_image_black_white(images[i])
        images[i] = common_function.center_letter(images[i])
        if show:
            images[i].show()
        tensors.append(model_character.load_image_tensor(img=images[i]))
    pred_dict = []
    for i in tensors:
        pred = model_char.predict(i)
        pred_dict.append(model_character.pred_to_char_dict(pred))
    return word_searcher.get_word(pred_dict, lang)

#print(get_word("dataset/word_sequenced/images/a01-003u-01-04.png", "en_US"))
