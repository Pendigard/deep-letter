from PIL import Image, ImageFont, ImageDraw, ImageOps
from random_word import RandomWords
import random

def get_text_image(path="dataset/word_sequenced/images"):
    random_word = RandomWords()
    word = random_word.get_random_word()
    random_integer = random.randint(0,2)
    if random_integer == 0:
        word = word.upper()
    elif random_integer == 1:
        word = word.capitalize()
    fonts = ["lato.ttf","lato-italic.ttf","lora.ttf","lora-italic.ttf","oswald.ttf"]
    font_path = f"fonts/{random.choice(fonts)}"
    font = ImageFont.truetype(font_path, 30)
    img = Image.new("RGB", (1000,100), (0,0,0))
    draw = ImageDraw.Draw(img)
    draw.text((0,0), word, font=font, fill=(255,255,255))
    img = img.crop(img.getbbox())
    img = ImageOps.invert(img)
    #img.save(f"{path}/{word}.png")
    img.show()

for i in range(30):
    get_text_image()

