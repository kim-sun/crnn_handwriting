# -*- coding: UTF-8 -*-
import torch
import torchvision
import torchvision.transforms as Transform
from torch.utils.data import DataLoader


from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from matplotlib.font_manager import fontManager
from fontTools.ttLib import TTFont
import cv2
from skimage.morphology import skeletonize, medial_axis

import os
import sys
import ipdb
import pickle
import random
import traceback
import numpy as np
from tqdm import tqdm

from datasets import DataImage


# font
def has_glyph(font_path, glyph):
    font = TTFont(font_path, fontNumber=0)  # fontNumber 要記得設定 才能使用ttc
    for table in font['cmap'].tables:
        if ord(glyph) in table.cmap.keys():
            return True
    return False

def fonts_to_dict(fonts, pickle_name):
    chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
             'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
             ]
    save_path = 'data/synthetic/'
                # os.path.basename(font_path).split('.')[0]
    trans = Transform.Compose([Transform.ToTensor()])# range [0, 255] -> [0.0,1.0]
    fonts_not_use = ["Bodoni Ornaments.ttf", "cmex10.ttf", "cmsy10.ttf", "Farisi.ttf",
                     "Hoefler Text Ornaments.ttf", "LastResort.otf", "Symbol.ttf", "Webdings.ttf",
                     "Wingdings 2.ttf", "Wingdings 3.ttf", "Wingdings.ttf", "ZapfDingbats.ttf", "Zapfino.ttf", ".DS_Store"]

    rotation = [0]

    _char = []
    char_label = []

    _test_char = []
    test_char_label = []

    font_dict = {}
    i = 0
    j = 0
    for font_path in fonts:
        tem_image = []
        if os.path.basename(font_path) not in fonts_not_use:
            with tqdm(chars ,desc='font now {}'.format(os.path.basename(font_path))) as characters:
                for char in characters:
                    ans = has_glyph(font_path, char)
                    if ans:
                        j+=1
                        font = ImageFont.truetype(
                            font_path, 26, encoding="unic")  # 使用自定義的字體 第二参數表示字符大小
                        bg = Image.new("RGB", (100, 100))  # 生成空白圖像 畫上字母
                        draw = ImageDraw.Draw(bg)
                        
                        width, height = font.getsize(char) # 獲得文字大小
                        offsetx, offsety = font.getoffset(char) # 獲得文字offset位置
                        x, y = (20, 20)  # 左上角座標
                        draw.text((x, y), char, font=font)

                        _copy = bg.crop(bg.getbbox())
                        _im = Image.new("RGB", (28, 28))
                        width, height = _copy.size
                        
                        if height > 28 or width > 28:
                            if height > width:
                                _copy = _copy.resize((int(width*26/height), 26))
                            elif height < width:
                                _copy = _copy.resize((24, int(height*24/width)))
                        width, height = _copy.size
                        _im.paste(_copy, (int((28-width)/2), int((28-height)/2)))
                        if font_dict.get(char, None) == None:
                            font_dict[char] = [_im]
                        else:
                            font_dict[char].append(_im)

    with open(os.path.join(save_path, pickle_name), 'wb') as f:
        pickle.dump(font_dict, f)

# emnist
def emnist_to_dict(img_loader, train=True):
    # data_num = 2000 if train else 400
    data_num = 100 if train else 40
    CHARS_count = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0,
                    'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'J': 0,
                    'K': 0, 'L': 0, 'M': 0, 'N': 0, 'O': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0,
                    'U': 0, 'V': 0, 'W': 0, 'X': 0, 'Y': 0, 'Z': 0,
                    'a': 0, 'b': 0, 'd': 0, 'e': 0, 'f': 0, 
                    'g': 0, 'h': 0, 'n': 0, 'q': 0, 'r': 0, 't': 0}

    merge = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
             'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

    CHARS_dict = {}
    for batch_img, batch_label in img_loader:
        min_CHARS_count = min(CHARS_count, key=CHARS_count.get)
        if CHARS_count[min_CHARS_count] < data_num: # train: 2000 test: 400
            for i in tqdm(range(len(batch_img)), desc='merge {}:{}'.format(min_CHARS_count, CHARS_count[min_CHARS_count])):
                if CHARS_count[merge[batch_label[i]]] == 0:
                    CHARS_dict[merge[batch_label[i]]] = [batch_img[i]]
                    CHARS_count[merge[batch_label[i]]] += 1
                else:
                    if CHARS_count[merge[batch_label[i]]] < data_num:
                        CHARS_dict[merge[batch_label[i]]].append(batch_img[i])
                        CHARS_count[merge[batch_label[i]]] += 1
        else:
            break
    return CHARS_dict



def main():
    # fonts
    # train_path = './font_type/train_font'
    # test_path = './font_type/test_font'

    # train_fonts = [os.path.join(train_path, font) for font in os.listdir(train_path)]
    # test_fonts = [os.path.join(test_path, font) for font in os.listdir(test_path)]

    # fonts_to_dict(train_fonts, 'fonts_dict_train_10.pkl')
    # fonts_to_dict(test_fonts, 'fonts_dict_test_58.pkl')



    # emnist
    trans_data = Transform.Compose([Transform.RandomRotation([90, 90], Image.BILINEAR),
                                    Transform.RandomVerticalFlip(p=1),
                                    Transform.Grayscale(num_output_channels=3),
                                    Transform.ToTensor()
                                    ])
    training_data = torchvision.datasets.EMNIST(  
                root='./synthetic/', # dataset儲存路徑
                split='balanced', # balanced
                train=True, # True表示是train訓練集，False表示test測試集  
                transform=trans_data, # 將原資料規範化到（0,1）區間  
                download=True,  
                )
    test_data = torchvision.datasets.EMNIST(  
                root='./synthetic/', # dataset儲存路徑
                split='balanced',
                train=False, # True表示是train訓練集，False表示test測試集  
                transform=trans_data, # 將原資料規範化到（0,1）區間  
                download=True,  
                )

    # 通過torchvision.datasets獲取的dataset格式可直接可置於DataLoader
    img_loader = DataLoader(dataset=training_data,
                            batch_size=8,  
                            shuffle=True)
    test_loader = DataLoader(dataset=test_data,
                            batch_size=8,  
                            shuffle=True)
    
    trans = Transform.Compose([Transform.ToTensor()])

    train_dict = emnist_to_dict(img_loader, train=True)
    test_dict = emnist_to_dict(test_loader, train=False)

    with open(os.path.join('./synthetic/', 'emnist_dict_100.pkl'), 'wb') as f:
        pickle.dump(train_dict, f)
    with open(os.path.join('./synthetic/', 'emnist_dict_40.pkl'), 'wb') as f:
        pickle.dump(test_dict, f)



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)