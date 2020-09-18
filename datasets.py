import torch
import pandas as pd
import ipdb
import os
import re
import random
import pickle
from tqdm import tqdm
from PIL import Image

class DataImage:
    def __init__(self, file_path, trans):
        self.file_path = file_path  # data/ or ./mnist
        with open(self.file_path, 'rb') as f:
            synthetic_data = pickle.load(f)
        char_num = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
             'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15, 'g': 16, 'h': 17, 'i': 18, 'j': 19,
             'k': 20, 'l': 21, 'm': 22, 'n': 23, 'o': 24, 'p': 25, 'q': 26, 'r': 27, 's': 28, 't': 29,
             'u': 30, 'v': 31, 'w': 32, 'x': 33, 'y': 34, 'z': 35,
             'A': 36, 'B': 37, 'C': 38, 'D': 39, 'E': 40, 'F': 41, 'G': 42, 'H': 43, 'I': 44, 'J': 45,
             'K': 46, 'L': 47, 'M': 48, 'N': 49, 'O': 50, 'P': 51, 'Q': 52, 'R': 53, 'S': 54, 'T': 55,
             'U': 56, 'V': 57, 'W': 58, 'X': 59, 'Y': 60, 'Z': 61
        }
        self.trans = trans
        self.font_set = synthetic_data[0]
        self.font_label = [char for char in synthetic_data[1]]
        # self.font_label = [torch.tensor([char for char in chars]) for chars in synthetic_data[1]]
        self.data_num = len(self.font_set)
    
    def __len__(self):
        return self.data_num


    def __getitem__(self, idx):
        """ Return '[idx].jpg' and its tags. """
        # data = self.data_set[idx]
        img = self.font_set[idx]
        label = self.font_label[idx]
        return img, label

    def collate_fn(self, data_set):
        batch = []
        # batch['img'] = torch.stack([data[0] for data in data_set], 0)
        for i, data in enumerate(data_set):
            if i == 0:
                img = self.trans(data[0]).unsqueeze(0)
                label = torch.tensor(data[1]) #.unsqueeze(0)
                length = torch.tensor(len(data[1])).unsqueeze(0)
            else:
                img = torch.cat((img, self.trans(data[0]).unsqueeze(0)), 0)
                label = torch.cat((label, torch.tensor(data[1])), -1) # .unsqueeze(0)
                length = torch.cat((length, torch.tensor(len(data[1])).unsqueeze(0)), -1)

        batch.append(img)
        batch.append(label) # .squeeze(0)
        batch.append(length)
        return batch
