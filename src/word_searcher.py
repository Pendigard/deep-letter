import enchant


def get_word_type(dict_char_list):
    """
    Fonction qui prédit le type de mot associé à un dictionnaire de caractère
    @param dict_char dictionnaire de caractère à transformer
    @return le type de mot associé au dictionnaire de caractère
    """
    min_prop = 0
    maj_prop = 0
    num_prop = 0
    for dico in dict_char_list:
        char = list(dico.keys())[0]
        if char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            maj_prop += 1
        elif char in "abcdefghijklmnopqrstuvwxyz":
            min_prop += 1
        elif char in "0123456789":
            num_prop += 1
    min_prop /= len(dict_char_list)
    maj_prop /= len(dict_char_list)
    num_prop /= len(dict_char_list)
    if num_prop > 0.6:
        return "num"
    elif maj_prop > 0.3:
        return "maj"
    elif min_prop > 0.1:  
        return "min"
    
def delete_accent(char):
    """
    Fonction qui supprime les accents d'un caractère
    @param char caractère à transformer
    @return le caractère transformé
    """
    accents = {'a': 'àâä', 'e': 'éèêë', 'i': 'îï', 'u': 'ùûü', 'o': 'ôö', 'y': 'ÿ', 'c': 'ç'}
    for key in accents.keys():
        if char in accents[key]:
            return key
    return char

def adapt_word(word):
    """
    Fonction qui supprime les accents et les caractères spéciaux d'un mot
    @param word mot à transformer
    @return le mot transformé
    """
    result = ""
    for i in range(len(word)):
        c = delete_accent(word[i])
        if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
            result += c
    return result

def is_capitalized(dict_char_list):
    """
    Fonction qui détermine si un mot commence par une majuscule
    @param dict_char_list liste de dictionnaire de caractère
    @return True si le mot est capitalisé, False sinon
    """
    for key in list(dict_char_list[0].keys()):
        if key not in "0123456789":
            return key in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def get_first_word(dict_char_list, word_type):
    """
    Fonction qui renvoie le premier mot naïf prédit en fonction du type de mot
    @param dict_char_list liste de dictionnaire de caractère
    @param word_type type du mot
    @return le premier mot naïf prédit
    """
    first_word = ""
    num = "0123456789"
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    possibilities = ""
    if word_type == "num":
        possibilities = num
    else:
        possibilities = letters
    for dico in dict_char_list:
        c = list(dico.keys())[0]
        for key in list(dico.keys()):
            if key in possibilities:
                c = key
                break
        first_word += c
    return first_word

def finalize_word(word, word_type, capitalized):
    """
    Fonction qui finalise un mot en fonction de son type et de sa capitalisation
    @param word mot trouvé
    @param word_type type du mot
    @param capitalized True si le mot est capitalisé, False sinon
    @return le mot final
    """
    if word_type == "min":
        word = word.lower()
        if capitalized:
            return word.capitalize()
        return word
    if word_type == "maj":
        return word.upper()
    if word_type == "num":
        return word
    
def get_highest_score_word(word, possible_words):
    """
    Fonction qui calule pour chaque mots possibles un scores et retourne les mots avec le score le plus élevé
    @param word mot naïf prédit
    @param possible_words liste de mots possibles
    @return liste de mots avec le score le plus élevé
    """
    score = 0
    scores = {}
    for possible_word in possible_words:
        for i in range(len(word)):
            if word[i] == possible_word[i]:
                score += 1
            if i > 0 and word[i-1] == word[i] and possible_word[i-1] == possible_word[i]:
                score += 1
        scores[possible_word] = score
        score = 0
    max_score = max(list(scores.values()))
    words = [key for key in scores.keys() if scores[key] == max_score]
    return words

def get_closest_word_prob(closest_words, dict_char_list):
    """
    Fonction qui trouve le mot le plus probable parmi une liste de mots
    @param closest_words liste de mots
    @param dict_char_list liste de dictionnaire de caractère
    @return le mot le plus probable
    """
    sum_proba = 0
    list_probas = []
    for word in closest_words:
        for i in range(len(word)):
            sum_proba += dict_char_list[i][word[i]]
        list_probas.append(sum_proba)
        sum_proba = 0
    max_proba = max(list_probas)
    index = list_probas.index(max_proba)
    return closest_words[index]

    
def get_real_word(dict_char_list, word, dictionnary):
    """
    Fonction qui recherche à partir d'une prediction de caractère le mot le plus proche
    @param dict_char_list liste de dictionnaire de caractère
    @param word mot naïf prédit
    @param dictionnary dictionnaire de mots de pyenchant
    @return le mot le plus proche du mot naïf
    """
    if dictionnary.check(word):
        return word
    word = word.lower()
    possible_words = dictionnary.suggest(word.lower())
    possible_words = [adapt_word(word) for word in possible_words if len(adapt_word(word)) == len(dict_char_list)]
    if len(possible_words) == 0:
        return word
    closest_words = get_highest_score_word(word, possible_words)
    return get_closest_word_prob(closest_words, dict_char_list)


 
def get_word(dict_char_list, lang="fr_FR"):
    """
    Fonction qui prédit le mot associé à un dictionnaire de caractère
    @param dict_char dictionnaire de caractère à transformer
    @param lang langue du dictionnaire (par défaut : fr_FR)
    @return le mot associé au dictionnaire de caractère et le mot trouvé naïvement
    """
    dictionary = enchant.Dict(lang)
    word_type = get_word_type(dict_char_list)
    capitalized = is_capitalized(dict_char_list)
    first_word = get_first_word(dict_char_list, word_type)
    if word_type == "num":
        return first_word, first_word
    word = get_real_word(dict_char_list, first_word, dictionary)
    word = finalize_word(word, word_type, capitalized)
    return word, first_word


