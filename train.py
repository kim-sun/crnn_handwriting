import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as Transform
from torch.utils.data import DataLoader

import os
import sys
import ipdb
import pickle
import traceback
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from datasets import DataImage
from crnn import crnn
from utils import load_model, save_model, compute_loss

def parse():
    parser = ArgumentParser()
    parser.add_argument('-d', '--device', help='Device to train the model on',
                        default=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"))
                        # choices=['cuda', 'cpu'], type=str)
    parser.add_argument('-e', '--epochs', help='Number of epochs of training',
                        default=10, type=int)
    parser.add_argument('-b', '--batch_size', help='Training batch size',
                        default=16, type=int)
    parser.add_argument('-f', '--fit_epochs', help='Training batch size',
                        default=0, type=int)                    
    parser.add_argument('-t', '--train_dir', help='Training data directory',
                        default='./data/synthetic/', type=str)
    parser.add_argument('-c', '--checkpoint_dir', help='Directory to save model checkpoints',
                        default='./checkpoints/', type=str) # onechar
    parser.add_argument('--check', help='Save model every _ epochs',
                        default=2, type=int)
    parser.add_argument('--lr', help='Learning rate; Default: 0.001',
                        default=0.01, type=float)
    parser.add_argument('--times', help='select train time',
                        type=int)
    parser.add_argument('--multi', help='multi or onechar',
                        action='store_true')
    parser.add_argument('--channel', help='select channel',
                        default=3, type=int)
    args = parser.parse_args()
    return args


def train(data_loader, test_loader, net, optimizer, loss_func, device, epochs, save_path, model_name, times, check, fit=1, multi=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    loss_history = []
    test_loss_history = []
    test_epoch = []
    test_loss_average = 0
    for epoch in range(fit + 1, epochs + 1):
        loss_tem = []
        net.train() # training mode
        with tqdm(data_loader, desc='font train epoch {}'.format(str(epoch))) as batchs:
            for batch in batchs:
                crnn_out, loss, input_length = compute_loss(batch, net, loss_func, device)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_tem.append(loss.item())
                loss_average = sum(loss_tem)/len(loss_tem)
                batchs.set_postfix_str('loss: {:^7.3f}, test_loss: {:^7.3f}'.format(loss_average, test_loss_average))
            loss_history.append(loss_average)


            if epoch % check == 0:
                # test
                test_loss_tem = []
                net.eval() # testing mode
                for i, batch in enumerate(test_loader):
                    crnn_out, val_loss, input_length = compute_loss(batch, net, loss_func, device)
                    test_loss_tem.append(val_loss.item())
                    test_loss_average = sum(test_loss_tem)/len(test_loss_tem)
                test_loss_history.append(test_loss_average)
                test_epoch.append(epoch)

                plt.figure()
                plt.plot(loss_history, 'r--', label="loss")
                plt.plot(test_loss_history, 'g', label="test")
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.savefig(save_path + model_name[1].format(times), bbox_inches='tight')

                save_model(model=net, optimizer=optimizer, epoch=epoch,
                                file_path=os.path.join(save_path, model_name[0].format(times, epoch)))

    save_model(model=net, optimizer=optimizer, epoch=epoch,
                                file_path=os.path.join(save_path, model_name[0].format(times, epoch)))


def main(args):
    device = args.device
    multi = args.multi
    times = args.times
    font_dir = args.train_dir + 'multi/' if multi else args.train_dir + 'onechar/'
    save_dir = args.checkpoint_dir + 'multi/' if multi else args.checkpoint_dir + 'onechar/'
    epochs = args.epochs
    fit_epochs = args.fit_epochs
    batch_size = args.batch_size
    check = args.check
    lr = args.lr
    net_channel = args.channel

    # CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
    #         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    #         'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    #         'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    #         'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '_']

    # CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    #          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    #          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    #          'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't', '_']

    CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_']
    

    trans_emnist = Transform.Compose([Transform.RandomRotation([90, 90], Image.BILINEAR),
                                      Transform.RandomVerticalFlip(p=1),
                                      Transform.Grayscale(num_output_channels=net_channel),
                                      Transform.ToTensor()
                                      ])

    trans_fonts = Transform.Compose([Transform.ToPILImage(),
                                     Transform.Grayscale(num_output_channels=net_channel),
                                     Transform.ToTensor()
                                     ])

    # from torchsummary import summary
    # crnn_net = crnn(class_dim=len(CHARS), channel=net_channel, multi=multi).to(device)
    # summary(crnn_net, (3, 28, 200))
    # ipdb.set_trace()

    if not multi: # 測試用
        train_path = 'digit/'
        # 英數混合 58種印刷體
        # train_filename = 'char_digit_all_58'
        train_filename = ''
        training_data = DataImage(font_dir + train_path + train_filename + '.pkl', trans=trans_fonts)
        testing_data = DataImage(font_dir + 'char_mnist_binary_test.pkl', trans=trans_fonts)
    
        train_loader = DataLoader(dataset=training_data, batch_size=batch_size,
                                shuffle=True, collate_fn=training_data.collate_fn)
        test_loader = DataLoader(dataset=testing_data, batch_size=batch_size,
                                shuffle=False, collate_fn=testing_data.collate_fn)
        #  python3  train.py --times 1 -e 20 -f 0 --check 1 --channel 3 --train_dir ./synthetic/

    # 論文主體
    else: # syn_multi_mnist_train syn_multi_all_digit_train lin_data_multi_A4_digit
        train_path = font_dir
        train_filename = 'syn_multi_mnist_scale_shift_orig'
        training_data = DataImage(train_path + train_filename + '.pkl', trans=trans_fonts)
        testing_data = DataImage('./real/A4_digit/lin_data_multi_A4_digit_non_dila.pkl', trans=trans_fonts)
    
        train_loader = DataLoader(dataset=training_data, batch_size=batch_size,
                                shuffle=True, collate_fn=training_data.collate_fn)
        test_loader = DataLoader(dataset=testing_data, batch_size=batch_size,
                                shuffle=False, collate_fn=testing_data.collate_fn)
        #  python3  train.py --times 1 -e 100 -f 0 --multi --check 1 --channel 3 --train_dir ./synthetic/

    save_dir = save_dir + train_filename + '_{}'.format(times)
    
    crnn_net = crnn(class_dim=len(CHARS), channel=net_channel, multi=multi).to(device)
    # crnn_net = crnn_resnet(class_dim=len(CHARS)).to(device)
    loss_func = torch.nn.CTCLoss(blank=len(CHARS)-1, reduction='mean', zero_infinity=True)
    optimizer = optim.Adadelta(crnn_net.parameters(), lr=lr)


    if fit_epochs > 1:
        state_path = save_dir + '/crnn_net_{}_{}.ckpt'.format(times, fit_epochs)
        crnn_net, optimizer = load_model(crnn_net, optimizer, state_path)
        model_name = [train_filename + '_{}_{}.ckpt', '/' + train_filename + '_loss_fit_{}.png']

    else:
        model_name = [train_filename + '_{}_{}.ckpt', '/' + train_filename + '_loss_{}.png']
    train(train_loader, test_loader, crnn_net, optimizer, loss_func, 
            device, epochs, save_dir, model_name, times, check, fit=fit_epochs, multi=multi)
    


if __name__ == '__main__':
    try:
        args = parse()
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)