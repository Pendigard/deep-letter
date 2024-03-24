
"""
----------------------------------------------------------------------------------------------------

                                                CROP

----------------------------------------------------------------------------------------------------
    Paramètre de l'image
----------------------------------------------------------------------------------------------------

    * Initialisation avec le lien de l'image en paramètre 
        exemple :
            img = split("img_test/le miel.jpg")

    * Affichage de l'image a découper :
        img.show_image()

    * Modification de l'image a découper :
        img.image_set("Ma/nouvelle/image.png")

----------------------------------------------------------------------------------------------------
    Découpe de l'image
----------------------------------------------------------------------------------------------------

    * Découpe de l'image
        img.resize()

    * Récupération du tableau de découpe de l'image ( Tableau 2D [Ligne] [Mots] ) :
        img.get_resize()
        
    * Affichage des images des découpe des mots
        img.show_resize()

----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
"""



from PIL import Image
import os


class Split:
    def __init__(self, imag_path):
        self.img = Image.open(imag_path)
        self.word = []
    
    def _to_b_w(self, a, b):
        res = a.convert("L")
        return res.point(lambda p: 0 if p < b else 255, '1')
    
    def _filter_list(self, liste):
        i = 0
        while i < len(liste):
            j = i + 1
            while j < len(liste):
                if abs(liste[j] - liste[i]) < 8:
                    del liste[j]
                else:
                    j += 1
            i += 1

    def show_image(self):
        self.img.show()

    def image_set(self, img2):
        self.img = Image.open(img2)

    def get_resize(self):
        return self.word

    def show_resize(self, indice=-1):
        if(indice < 0):
            for i in range(len(self.word)):
                for j in range(len(self.word[i])):
                    self.word[i][j].show()
        elif (indice < len(self.word)):
            for i in range(len(self.word[indice])):
                self.word[indice][i].show()
        else:
            print("ERROR : INDICE OUT OF RANGE IN SPLIT")
        

    def save_resize(self, dir):
        for i in range(len(self.word)):
                for j in range(len(self.word[i])):
                    nom_image = f"image_{i}_{j}.png"
                    chemin_sauvegarde = os.path.join(dir, nom_image)
                    self.word[i][j].save(chemin_sauvegarde) 
    
    def resize(self):

        if hasattr(self.img, '_getexif') and self.img._getexif() is not None:
            orientation = self.img._getexif().get(0x0112, 1)
            if orientation in (2, 3, 4, 5, 6, 7, 8):
                self.img = self.img.rotate(-90, expand=True)

        space = []
        cut_value = []
        tab_ligne = []

        img_to_cut = self.img.resize(   (int(self.img.width * 0.1) , self.img.height  )   )

        img_to_cut = self._to_b_w(img_to_cut, 150)
        
        for i in range(img_to_cut.height): 
            pixels_rangee = list(img_to_cut.getpixel((col, i)) for col in range(img_to_cut.width))
            moy_pixels_rangee = sum(pixels_rangee) / len(pixels_rangee)

            if moy_pixels_rangee < 254:
                space.append(1)
            else:
                space.append(0)

        for i in range(len(space)):
            if(space[i] == 1 and i < len(space)-1):
                if(i > 0):
                    if(space[i-1] == 0):
                        cut_value.append(i)
                    if(space[i+1] == 0):
                        cut_value.append(i)
        if(space[len(space)-1] == 1 and space[len(space)-2] == 1): 
                cut_value.append(len(space)-1)
        
        self._filter_list(cut_value)

        for i in range(0, len(cut_value)-1, 2):
            boite_de_decoupe = (0, cut_value[i], self.img.width, cut_value[i+1])
            img_cadre = self.img.crop(boite_de_decoupe)
            tab_ligne.append(img_cadre)
        
        for ligne in reversed(tab_ligne):
            img_to_cut = self._to_b_w(ligne, 100)
            pixels = img_to_cut.getdata()
            somme_nuances_gris = sum(pixels)

            if((ligne.width * ligne.height) > 0):
                nuance_gris_moyenne = somme_nuances_gris // (ligne.width * ligne.height)

                if nuance_gris_moyenne > 254:
                    tab_ligne.remove(ligne)
            else:
                tab_ligne.remove(ligne)
            
            nuance_gris_moyenne = 0
        
        v = 0
        for ligne in reversed(tab_ligne):
            if(ligne.height > v):
                v=ligne.height
        
        for ligne in reversed(tab_ligne):
            if(ligne.height < v/3):
                tab_ligne.remove(ligne)

        for i in range(len(tab_ligne)):

            del cut_value
            del space
            del pixels_rangee
            cut_value = []
            space = []
  
            ln_to_cut = tab_ligne[i].resize( (  tab_ligne[i].width , int(tab_ligne[i].height * 0.4) ) )

            ln_to_cut = self._to_b_w(ln_to_cut, 170)

            for j in range(ln_to_cut.width): 
                pixels_rangee = list(ln_to_cut.getpixel((j, lig)) for lig in range(ln_to_cut.height))
                moy_pixels_rangee = sum(pixels_rangee) / len(pixels_rangee)

                if moy_pixels_rangee > 254:
                    space.append(1)
                else:
                    space.append(0)

            for j in range(1,len(space)-1,1):
                if(space[j] == 0 and space[j-1] == 1 and space[j+1] == 1):
                    space[j] = 1

            for j in range(1,len(space)-1,1):
                if(space[j] == 1 and space[j-1] == 0 and space[j+1] == 0):
                    space[j] = 0

            for j in range(1, len(space)-1):
                if(space[j] == 1 ):
                    if(j > 0 and j < len(space)-1):
                        if(space[j-1] == 0):
                            cut_value.append(j)
                        if(space[j+1] == 0):
                            cut_value.append(j)
            
            space = []
            space = cut_value
            cut_value = []
            cut_value.append(space[0])

            _len_list = []

            for j in range(1, len(space)-1, 2):
                _len_list.append(space[j+1]-space[j])

            if len(_len_list) > 0:
                _moy_len = sum(_len_list) / len(_len_list)
            else:
                _moy_len = 0

            for j in range(1, len(space)-1, 2):
                if space[j+1]-space[j] > _moy_len :
                    cut_value.append(space[j])
                    cut_value.append(space[j+1])
            
            cut_value.append(space[len(space)-1])

            space = []
            space = cut_value
            cut_value = []
            
            cut_value.append(space[0])
            cut_value.append(space[1])


            for j in range(2, len(space)-1, 2):
                if space[j+1]-space[j] > tab_ligne[i].height/2.5 :
                    cut_value.append(space[j])
                    cut_value.append(space[j+1])
                else : 
                    cut_value.pop(len(cut_value)-1)
                    cut_value.append(space[j+1])

            img = []

            for j in range(0, len(cut_value)-1, 2):
                boite_de_decoupe = (cut_value[j], 0, cut_value[j+1], tab_ligne[i].height,)
                img_cadre = tab_ligne[i].crop(boite_de_decoupe)

                img.append(img_cadre)
            self.word.append(img)