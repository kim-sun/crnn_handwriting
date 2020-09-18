import torch
import torchvision.transforms as Transform
import torchvision
from torch.utils.data import DataLoader

import os
import cv2
import sys
import ipdb
import random
import pickle
import logging
import traceback
import numpy as np


from argparse import ArgumentParser
from skimage.morphology import skeletonize, medial_axis
from PIL import Image
from tqdm import tqdm

from datasets import DataImage


def erosion(img):
    skeleton = medial_axis(img, return_distance=False) # distance if return_distance = True
    dist_transform = cv2.distanceTransform(img, cv2.DIST_L2, 3)
    arr = dist_transform
    for i, row in enumerate(arr):
        for j, col in enumerate(row):
            if skeleton[i][j] == True:
                arr[i][j] += np.max(dist_transform)/2
            else:
                arr[i][j] += 0
    return arr * (1/np.max(dist_transform)) * 255

def dilation(img):
    kernel = np.ones((3, 3), np.uint8)
    orig = np.zeros(img.shape, np.uint8)
    pixel_max = 256
    for t in range(3):
        for i, row in enumerate(orig):
            for j, col in enumerate(row):
                if img[i][j] > 0 and t == 0:
                    if img[i][j] >= 192:
                        orig[i][j] = pixel_max - 1
                    elif img[i][j] >= 128 and img[i][j] < 192:
                        orig[i][j] = pixel_max/4 * 3
                    else:
                        orig[i][j] = pixel_max/4 * 2
                    
                elif img[i][j] > 0 and orig[i][j] == 0:
                    orig[i][j] = di_num
        if t == 0:
            di_num = pixel_max/4
        else:
            di_num /= 2
        img = cv2.dilate(img, kernel, iterations = 1)
    return orig.astype('float32')

def paste_bg(bg, img, start, start_array, up_or_down):
    im = img.crop(img.getbbox())
    width, height = im.size
    if start == 1:
        start_array = int((100-height)/2)
    if up_or_down == -1 and start != 1:
        start_array -= 2
    elif up_or_down == 1 and start != 1:
        start_array += 2
    im_array = np.array(im)
    # bg.paste(im, (start, int((28-height)/2)), im)
    bg[start_array:start_array+height, start:start+width] = np.maximum(bg[start_array:start_array+height, start:start+width], im_array)
    # start += (width + random.randint(-1, 3))
    start += (width + -1)
    
    # Image.fromarray(bg).show()
    return bg, start, start_array

def paste_append(bg, bg_append):
    im = bg.crop(bg.getbbox())
    width, height = im.size
    if height > 28:
        im = im.resize((int(width*26/height), 26))
    width, height = im.size
    bg_append.paste(im, (1, int((28-height)/2)))
    return bg_append

def generate_data(dictionary, table_char, num, orfont=False, desctqdm='sym_multi'):
    trans = Transform.Compose([Transform.ToTensor()])
    _orig = []
    orig_label = []
    _imglist = []
    label = []
    page_height = 100
    for i in tqdm(range(num), desc=desctqdm):
        bg_start = 1
        up_or_down = random.randint(-1, 1)  # shift
        # up_or_down = 0 

        label_num = random.randint(2, 8)
        # en_position = random.randint(1, label_num - random.randint(2, label_num-1)) if label_num > 3 else -1
        # en_position = [random.randint(0, 2)] if label_num < 4 else [random.randint(2, 3), random.randint(4, 5)]
        en_position = [-1]

        bg = np.zeros((page_height, 200))
        di_bg = np.zeros((page_height, 200))
        er_bg = np.zeros((page_height, 200))

        label_tem = [random.randint(0, 9) if i not in en_position else random.randint(10, len(table_char)-1) for i in range(label_num)]
        if orfont:
            img_PIL = []
            font = random.randint(0, len(dictionary['0'])-1)
            _img_PIL = [dictionary[table_char[i]][font] for i in label_tem]
            for img in _img_PIL:
                image = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2GRAY)
                img_PIL.append(image)
        else:
            img_PIL = []
            _img_PIL = [Transform.ToPILImage()(dictionary[table_char[i]][random.randint(0, 99)]) for i in label_tem]
            for img in _img_PIL:
                image = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2GRAY)
                thres, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                img_PIL.append(image)

        for image in img_PIL:
            img = image
            # skeleton = skeletonize(img/255)
            # skeleton = medial_axis(img, return_distance=False) # distance if return_distance = True
            
            random_size = random.randint(15, 28)  # scale
            # img = Image.fromarray(img)
            # img = np.asarray(img.resize((random_size, random_size), 0))
            if bg_start == 1:
                di_img = dilation(img)
                di_bg, di_start, di_array = paste_bg(di_bg, Image.fromarray(di_img), 1, 0, up_or_down)
                
                er_img = erosion(img)
                er_bg, er_start, er_array = paste_bg(er_bg, Image.fromarray(er_img), 1, 0, up_or_down)

                bg, bg_start, bg_array = paste_bg(bg, Image.fromarray(img), bg_start, 0, up_or_down)
            else:
                di_img = dilation(img)
                di_bg, di_start, di_array = paste_bg(di_bg, Image.fromarray(di_img), bg_start, di_array, up_or_down)
                
                er_img = erosion(img)
                er_bg, er_start, er_array = paste_bg(er_bg, Image.fromarray(er_img), bg_start, er_array, up_or_down)

                bg, bg_start, bg_array = paste_bg(bg, Image.fromarray(img), bg_start, bg_array, up_or_down)

        bg_append = Image.new('L', (200, 28), (0))
        di_append  = Image.new('L', (200, 28), (0)) # (100, 32)
        er_append  = Image.new('L', (200, 28), (0))

        bg_append = paste_append(Image.fromarray(bg), bg_append)
        di_append = paste_append(Image.fromarray(di_bg), di_append)
        er_append = paste_append(Image.fromarray(er_bg), er_append)

        _orig.append(bg_append)
        orig_label.append(label_tem)
        _imglist.append(di_append)
        label.append(label_tem)
        _imglist.append(er_append)
        label.append(label_tem)


    # ipdb.set_trace()
    # Transform.ToPILImage()
    orig = torch.stack([trans(img) for img in _orig], 0) # /255 因為np轉pil轉totensor 不會[0~1]
    imglist = torch.stack([trans(img) for img in _imglist], 0)
    return orig, imglist, orig_label, label

# real img
def img2pickle(file_path, multi=False, show=0, p_dump=False):
    merge = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
             'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

    trans = Transform.Compose([Transform.ToTensor()])
    _imglist = []
    # _imglist = {'0':[], '1':[], '2':[], '3':[], '4':[], '5':[], '6':[], '7':[], '8':[], '9':[]}
    namelist = []   # '6_24/5/106.png'
    kernel = np.ones((3, 3), np.uint8)
    allFileList = os.listdir(file_path)
    for name in allFileList:
        if os.path.splitext(name)[-1] == '.png':
            im_o = cv2.imread(os.path.join(file_path, name), cv2.IMREAD_GRAYSCALE)
            im_c = 255 - im_o
            # thres, _im = cv2.threshold(im_c, 80, 255,cv2.THRESH_TOZERO)
            thres, _im = cv2.threshold(im_c, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # eq = cv2.equalizeHist(_im)
            
            # _im = cv2.morphologyEx(_im, cv2.MORPH_CLOSE, kernel) # cv2.MORPH_CLOSE
            # _im = cv2.dilate(_im, kernel, iterations = 1)
            # _im = cv2.erode(_im, kernel, iterations = 1)

            # _im = cv2.adaptiveThreshold(im_c, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2) # cv2.THRESH_BINARY_INV
            im = Image.fromarray(_im)
            im = im.crop(im.getbbox())
            width, height = im.size
            # if height > width:
            #     im = im.resize((int(24*width/height), 24), 0) # width, height
            # else:
            #     im = im.resize((24, int(24*height/width)), 0) # width, height
            im = im.resize((int(24*width/height), 24), 0) # width, height
            width, height = im.size
            
            if multi:
                bg = Image.new("RGB", (128, 28))
                bg.paste(im, (2, int((28-height)/2)))
            else:
                bg = Image.new("RGB", (28, 28))
                bg.paste(im, (int((28-width)/2), int((28-height)/2)))
            
            _imglist.append(bg)
            _label = [i for i in os.path.splitext(name)[0].split('_', 1)[1]]
            if multi:
                label_tem = []
                for tem in _label:
                    if tem not in ['.', '/', '-', ':']:
                        if tem not in merge:
                            label_tem.append(merge.index(tem.upper()))
                        else:
                            label_tem.append(merge.index(tem))
                namelist.append(label_tem)
            else:
                for tem in _label:
                    if tem not in merge:
                        namelist.append([merge.index(tem.upper())])
                    else:
                        namelist.append([merge.index(tem)])
    imglist = torch.stack([trans(img) for img in _imglist], 0)
    if show > 0:
        for i, img in enumerate(_imglist):
            if i == show:
                break
            img.show()

    if p_dump and multi:
        with open(os.path.join(file_path, 'lin_data_multi_A4_digit_non_dila.pkl'), 'wb') as f: # lin_data_onechar_gray
            pickle.dump([imglist, namelist], f)
            # pickle.dump(_imglist, f)
    elif p_dump:
        with open(os.path.join(file_path, 'lin_data_onechar_A4_digit_non_dila.pkl'), 'wb') as f: # lin_data_onechar_gray
            pickle.dump([imglist, namelist], f)

def main():
    table_char = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # with open(os.path.join('./synthetic/', 'emnist_dict_100.pkl'), 'rb') as f:
    #     emnist_dict_2000 = pickle.load(f)
    # fonts_dict_test_58.pkl fonts_dict_train_10.pkl
    # with open(os.path.join('./synthetic/', 'fonts_dict_train_10.pkl'), 'rb') as f:
    #     fonts_dict = pickle.load(f)

    # 手寫體為基礎
    # orig, imglist, orig_label, label = generate_data(emnist_dict_2000, table_char, num=100, orfont=False, desctqdm='sym_multi')
    # 印刷體為基礎
    # font_orig: 印刷體原本, font_imglist:印刷體漸層
    # font_orig, font_imglist, font_orig_label, font_label = generate_data(fonts_dict, table_char, num=20000, orfont=True, desctqdm='sym_multi')

    # 沒有作漸層
    # with open(os.path.join('./synthetic/multi', 'syn_multi_mnist_scale_shift_orig.pkl'), 'wb') as f:
    #     pickle.dump([orig, orig_label], f)

    # 用印刷體產生
    # with open(os.path.join('./data/synthetic/multi', 'syn_multi_digit_fonts_scale_shift_10.pkl'), 'wb') as f:
    #     pickle.dump([font_orig, font_orig_label], f)

    # 以上兩個pickle可以cat組合成不同訓練集
    # 訓練時再決定


    # real_img
    img2pickle(file_path='./real/A4_digit/', multi=True, show=30, p_dump=True)
    # img2pickle(file_path='./real/A4_digit/', multi=False, show=30, p_dump=False)




if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
    