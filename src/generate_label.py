import tkinter as tk
from PIL import Image, ImageTk
import os


def show_image(img):
    global widget_image
    photo = ImageTk.PhotoImage(img)
    widget_image.config(image=photo)
    widget_image.image = photo

def get_image_to_analyse():
    global line, labels, images_to_analyse, image_already_labeled, current_index, x_image, image, transition, window
    if line != "":
        if transition != None:
            if transition[0] != transition[1]:
                line += f"{transition[0]}-{min(transition[1], image.width-1)} "
            else:
                line += f"{transition[0]} "
        print(line)
        labels.write(line + "\n")
        transition = None
    line = ""
    current_index += 1
    while current_index < len(images_to_analyse) and images_to_analyse[current_index] in image_already_labeled:
        current_index += 1
    if current_index >= len(images_to_analyse):
        print("No image to analyse")
        exit()
    image = Image.open(f"dataset/word_sequenced/images/{images_to_analyse[current_index]}")
    line = f"{images_to_analyse[current_index]} "
    line += f"{image.width} "
    line += f"{image.height} "
    transition = None
    x_image = 0

# Fonction pour afficher l'image actuelle
def treat_current_image(label):
    global current_index, x_image, labels, image, transition, line, images_to_analyse, image_already_labeled, window
    window.title(images_to_analyse[current_index])
    if label == 3:
        line = ""
        get_image_to_analyse()
    if x_image >= image.width:
        get_image_to_analyse()
        red_band = Image.new("RGB", (1,image.height), (255,0,0))
        image_copy = image.copy()
        image_copy.paste(red_band, (x_image,0))
        show_image(image_copy)
    else:
        red_band = Image.new("RGB", (1,image.height), (255,0,0))
        image_copy = image.copy()
        if label != -1:
            if label == 0:
                if transition == None:
                    transition = (x_image, x_image)
                else:
                    transition = (transition[0], x_image)
            else:
                if transition != None:
                    if transition[0] != transition[1]:
                        line += f"{transition[0]}-{transition[1]} "
                    else:
                        line += f"{transition[0]} "
                    transition = None
            if label == 2:
                x_image += 10
            else:
                x_image += 1
        image_copy.paste(red_band, (x_image,0))
        show_image(image_copy)
    return

def filter_file_image():
    global images_to_analyse
    images_to_analyse = [i for i in images_to_analyse if ".png" in i or ".jpg" in i]

def main():
    global window, widget_image, current_index, x_image, image, labels, transition, line, images_to_analyse, image_already_labeled

    transition = None

    line = ""

    # Charger la liste des noms de fichiers d'images à partir du fichier texte
    labels = open("dataset/word_sequenced/labels.txt", "r+")
    image_already_labeled = []

    for l in labels:
        if l[0] != "#":
            image_already_labeled.append(l.split(" ")[0])

    images_to_analyse = []
    images_to_analyse =  os.listdir("dataset/word_sequenced/images")

    filter_file_image()

    if len(images_to_analyse) == 0:
        print("No image to analyse")
        exit()

    current_index = 0
    x_image = 0
    line = ""
    
    get_image_to_analyse()


    # Créer une fenêtre Tkinter
    window = tk.Tk()

    widget_image = tk.Label(window)
    widget_image.pack()


    # Lier la pression de la touche "D" à la fonction image_suivante
    window.bind("a", lambda event: treat_current_image(1))
    window.bind("z", lambda event: treat_current_image(2))
    window.bind("s", lambda event: treat_current_image(3))
    window.bind("p", lambda event: treat_current_image(0))

    # Afficher la première image
    treat_current_image(-1)

    # Lancer la boucle principale de l'application
    window.mainloop()

main()

