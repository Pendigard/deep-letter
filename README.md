# Projet Deep-Letter

Le projet Deep-Letter est une initiative visant à développer un OCR (Optical Character Recognition) sans recourir à des bibliothèques externes dédiées à la reconnaissance de texte. L'objectif principal est de comprendre et d'implémenter les concepts fondamentaux de l'OCR à partir de zéro, en se basant sur le machine learning.

## Objectif

Ce projet est réalisé dans le cadre du cours de LIFPROJET en 2023. L'objectif principal est de créer un modèle de reconnaissance de caractères capables de convertir des images de texte en texte brut, sans utiliser de bibliothèques externes telles que Tesseract ou YOLO. En développant notre propre solution, nous cherchons à approfondir notre compréhension des réseaux de neurones, de la vision par ordinateur et des techniques de traitement d'image.

## Fonctionnalités

- **Prétraitement d'image :** Implémentation de techniques de prétraitement d'image pour améliorer la qualité des données d'entrée.
  
- **Architecture du modèle :** Définition d'une architecture de réseau de neurones adaptée à la reconnaissance de caractères dans des images ainsi qu'au découpage d'image de mot en lettre.

- **Entraînement du modèle :** Collecte de données d'entraînement et mise en œuvre d'un processus d'entraînement pour le modèle.

- **Évaluation :** Mise en place de métriques d'évaluation pour mesurer la performance du modèle sur des ensembles de données de test.

## Dépendances

- **Python 3.9+**
- **Torch**
- **NumPy**
- **Matplotlib**
- **Pillow**
- **OpenCV**
- **TQDM**
- **pyenchant**
- **tkinter**

## Utilisation

Dans `main.py` décommenter les parties qui vous intéresse et exécuter le fichier.

Fonctions importantes :

- **`get_word(path, lang, model)`** : Prend en entrée le chemin d'une image et retourne le mot reconnu en string avant traitement par le reconnaisseur de mot et après.

- **`get_text(path, lang, model)`** : Prend en entrée le chemin d'une image et retourne le texte reconnu en string avant traitement par le reconnaisseur de lettre et après.

## Auteurs

- **VASSON Célian**
- **AKAKPO Godfree**
- **DOXANIS Evan**
