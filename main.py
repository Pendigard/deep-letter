from src.word_recognizer import get_word
from src.split import Split
from PIL import Image
from src.model_character import get_model
import src.common_function as common_function
from src.model_transition import test_num_letters_split
from src.algorithm_split import split_image

def get_text(src, lang, model="strong_byclass", show=False, model_word=None):
    s = Split(src)
    s.resize()
    tab_img = s.get_resize()
    text = []
    naive_text = []
    for i in range(len(tab_img)):
        line = ""
        naive_line = ""
        for j in range((len(tab_img[i]))):
            line += get_word(img=tab_img[i][j], lang=lang, model=model, show=show)[0] + " "
            naive_line += str(get_word(img=tab_img[i][j], lang=lang, model=model)[1]) + " "
        line = line.capitalize()
        naive_line = naive_line.capitalize()
        text.append(line)
        naive_text.append(naive_line)
    return text, naive_text

def print_text(text, naive_text=False):
    if naive_text != False:
        print("Texte naïf :")
        for i in range(len(naive_text)):
            print(naive_text[i])
    print("Texte reconnu :")
    for i in range(len(text)):
        print(text[i])

def presentation():
    images_path = ["img_test/sport.png", "img_test/ouvrage.jpg", "img_test/salm.png", "img_test/lundi.png", "img_test/lundi2.png", "img_test/Text irl.jpg", "img_test/text.jpg"]
    word_index = [1, 2, len(images_path)-1]
    lang = "fr_FR"
    input("Appuyez sur une touche pour commencer...")
    for i in range(len(images_path)):
        print("Image n°" + str(i+1) + " :")
        Image.open(images_path[i]).show()
        if i not in word_index:
            text, naive_text = get_text(images_path[i], lang, model="strong_byclass2")
            print_text(text, naive_text)
        else:
            show = False
            if i == 1 or i == 2:
                show = True
            if i == word_index[2]:
                lang = "en_US"
            text, naive_text = get_word(path=images_path[i], img=None, lang=lang, model="strong_byclass2", show=show)
            print("Texte naïf :")
            print(naive_text)
            print("Texte reconnu :")
            print(text)
        print("")
        input("Appuyez sur une touche pour continuer...")

src = "img_test/text.jpg"
lang = "fr_FR"
emnist_train_loader, emnist_test_loader = common_function.get_emnist_char_loader("byclass")

# Reconnaissance pour un mot unique
# print(get_word(path=src, lang=lang, model="strong_byclass2"))

# -----------------------------------------------------
# Reconnaissance pour un texte
#strong_text, strong_naive_text = get_text(src, lang, model="strong_byclass2")
#print_text(strong_text, strong_naive_text)

# -----------------------------------------------------
# Code pour la présentation
#presentation()

# -----------------------------------------------------
# Evaluation du modèle de reconnaissance de caractère
#model = get_model("strong_byclass2")
#model.test_model(emnist_test_loader)

# -----------------------------------------------------
# Evaluation du modèle de découpage de mot
#test_num_letters_split()

# -----------------------------------------------------
# Evaluation du modèle de découpage de mot
#test_num_letters_split(split_image)