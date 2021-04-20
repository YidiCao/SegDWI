import argparse
import logging
import os
import sys
import cv2

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import random

from eval import eval_net
from unet import UNet

from tensorboardX import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.01,
              save_cp=True,
              img_scale=256,
              gpus='1,2',
              augment=True,
              nickname='E00'):

    train = BasicDataset('img.train.list', augment=augment, scale=img_scale)
    val = BasicDataset('img.test.list', augment=False, scale=img_scale)
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment='_{}_LR_{}'.format(nickname, lr))
    global_step = 0

    logging.info('''Starting training:
        Epochs:          {}
        Batch size:      {}
        Learning rate:   {}
        Checkpoints:     {}
        Device:          {}
        Images scaling:  {}
        Augmentation:    {}
    '''.format(epochs, batch_size, lr, save_cp, device.type, img_scale, augment))

    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9, nesterov=False)
 #   scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
 #   scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    print(net)
    net.cuda()
    
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch+1, epochs), unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image'].cuda()
                true_masks = batch['mask'].cuda()

                name = batch['seri'][0]
                assert imgs.shape[1] == net.n_channels, \
                    'Network has been defined with {} input channels, ' \
                    'but loaded images have {} channels. Please check that ' \
                    'the images are loaded correctly.'.format(net.n_channels, imgs.shape[1])

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                writer.add_scalar('Loss/Train_Step', loss.item(), global_step)

                if global_step % (n_train // (10 * batch_size)) == 0:

                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('Weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('Grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                    writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
                    writer.add_images('Images/Train/Image', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('Images/Train/Mask', true_masks, global_step)
                        writer.add_images('Images/Train/Pred', torch.sigmoid(masks_pred) > 0.5, global_step)
                global_step += 1
        
        print('=============================')
        val_score, val_loss, writer = eval_net(net, val_loader, device, nickname=nickname, epoch=epoch+1, writer=writer)

        writer.add_scalar('Loss/Train_Epoch', epoch_loss/n_train, epoch+1)
        writer.add_scalar('Loss/Test_Epoch', val_loss, epoch+1)
        writer.add_scalar('Dice/Test_Epoch', val_score, epoch+1)

        logging.info('Epoch:{}'.format(epoch+1))
        logging.info('LR:{}'.format(scheduler.get_lr()))
        logging.info('Validation Loss:{}'.format(val_loss))
        logging.info('Validation Dice Coeff: {}'.format(val_score))
        scheduler.step()
        print('=============================')

        if save_cp and (epoch+1)%10==0:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            os.makedirs(os.path.join(dir_checkpoint, nickname), exist_ok=True)
            torch.save(net.state_dict(),
                       dir_checkpoint + '/{}/CP_epoch{}.pth'.format(nickname, epoch+1))
            logging.info('Checkpoint {} saved !'.format(epoch+1))

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=200,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default='',
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=256,
                        help='Downscaling factor of the images')
    parser.add_argument('-g', '--gpus', dest='gpus', type=str, default='8,7',
                        help='Gpus')
    parser.add_argument('-d', '--debug', dest='debug', type=bool, default=True,
                        help='Debug')
    parser.add_argument('-a', '--augment', dest='augment', type=bool, default=True,
                        help='Augmentation')
    parser.add_argument('-n', '--nickname', dest='nickname', type=str, default='E00',
                    help='Nickname')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device {}'.format(device))

    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    logging.info('Network:\n\t{} input channels\n\t{} output channels (classes)\n\t{} upscaling'.format(net.n_channels, net.n_classes, "Bilinear" if net.bilinear else "Transposed conv"))

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info('Model loaded from {}'.format(args.load))

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  gpus=args.gpus,
                  augment=args.augment,
                  nickname=args.nickname)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
