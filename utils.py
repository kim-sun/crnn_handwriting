import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm


def save_model(model, optimizer, epoch, file_path):
    """ Save model checkpoints. """

    state = {'model' : model.state_dict(),
             'optim' : optimizer.state_dict(),
             'epoch' : epoch}
    torch.save(state, file_path)
    return

def load_model(model, optimizer, file_path):
    """ Load previous checkpoints. """

    prev_state = torch.load(file_path, map_location={'cuda:1': 'cpu'})
    
    model.load_state_dict(prev_state['model'])
    optimizer.load_state_dict(prev_state['optim'])
    
    return model, optimizer


def compute_loss(batch, crnn_net, ctc_loss, device):
    batch_img = batch[0].to(device)
    batch_label = batch[1].to(device)
    batch_length = batch[2].to(device)

    crnn_out = crnn_net(batch_img)

    input_length = torch.full(
        size=(crnn_out.size()[1],), fill_value=crnn_out.size()[0], dtype=torch.long)
    
    loss = ctc_loss(crnn_out, batch_label, input_length, batch_length) / crnn_out.size()[1]
    return crnn_out, loss, input_length
