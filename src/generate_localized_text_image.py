from PIL import Image, ImageFont, ImageDraw, ImageOps
from random_word import RandomWords
import random
from tqdm import tqdm

def generate_paragraph(path="dataset/paragraph/images"):
    nbr_word = random.randint(5, 15)
    nbr_line = random.randint(0, 15)
    if nbr_line == 0:
        return
    fonts = ["lato.ttf","lato-italic.ttf","lora.ttf","lora-italic.ttf","oswald.ttf"]
    font_path = f"fonts/{random.choice(fonts)}"
    font = ImageFont.truetype(font_path, 30)
    image = Image.new("RGB", (500,500), (255,255,255))
    x_begin = random.randint(0, 400)
    y_begin = random.randint(0, 400)
    previous_width = 0
    previous_height = 0
    text_width = 0
    for i in tqdm(range(nbr_line)):
        for j in range(nbr_word):
            word = get_word_image(font,path)
            width = word.width
            height = word.height
            if x_begin + width > 500:
                break
            if y_begin + height > 500:
                break
            image.paste(word, (x_begin+previous_width, y_begin+previous_height))
            previous_width += width
        text_width = min(previous_width, 500-x_begin)
        previous_width = 0
        previous_height += height
    previous_height = min(previous_height, 500-y_begin)
    label = f"{text_width} {previous_height} {x_begin} {y_begin}"
    print(label)
    draw = ImageDraw.Draw(image)
    draw.rectangle((x_begin, y_begin, x_begin+text_width, y_begin+previous_height), outline=(0,0,0))
    image.show()



    
       

def get_word_image(font, path="dataset/word_sequenced/images"):
    random_word = RandomWords()
    word = random_word.get_random_word() + " "
    img = Image.new("RGB", (1000,100), (0,0,0))
    draw = ImageDraw.Draw(img)
    draw.text((0,0), text=word, font=font, fill=(255,255,255))
    img = img.crop(img.getbbox())
    img = ImageOps.invert(img)
    #img.save(f"{path}/{word}.png")
    #img.show()
    return img

generate_paragraph()