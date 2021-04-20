import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
import argparse
import logging

from tqdm import tqdm

from dice_loss import dice_coeff
from unet import UNet
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split


def eval_net(net, loader, device, nickname, epoch=None, writer=None):
    """Evaluation without the densecrf with the dice coefficient"""
    net.cuda()
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    criterion = nn.BCEWithLogitsLoss()
    tot_dice = 0
    tot_loss = 0

    sample_idx = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks, seri = batch['image'], batch['mask'], batch['seri'][0]
            print(seri)
            imgs = imgs.to(device=device, dtype=torch.float32).cuda()
            true_masks = true_masks.to(device=device, dtype=mask_type).cuda()

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                pass
            else:
                val_loss = criterion(mask_pred, true_masks).item()
                tot_loss += val_loss
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot_dice += dice_coeff(pred, true_masks).item()
            pbar.update()

            
            print(os.path.dirname('debug/{}/{}/{}'.format(nickname, epoch, seri)))
            os.makedirs(os.path.dirname('debug/{}/{}/{}'.format(nickname, epoch, seri)), exist_ok=True)
            canvas_out = imgs.detach().cpu().numpy()[0,:,:,:].transpose(1,2,0)*255
            overlap = canvas_out.copy()
            overlap[:,:,1] = pred.detach().cpu().numpy()[0,0,:,:]*255
            overlap[:,:,2] = true_masks.detach().cpu().numpy()[0,0,:,:]*255
            cv2.imwrite('debug/{}/{}/{}'.format(nickname, epoch, seri), np.hstack([canvas_out, overlap]))
            
            if (epoch != None) and (epoch%50==0):
                if sample_idx % (n_val // 10) == 0:
                    try:
                        writer.add_images('Images/Val/Image', imgs, sample_idx)
                        writer.add_images('Images/Val/Mask', true_masks, sample_idx)
                        writer.add_images('Images/Val/Pred', torch.sigmoid(pred) > 0.5, sample_idx)
                    except:
                        print('No writer.')

            sample_idx += 1

    net.train()
    return tot_dice / n_val, tot_loss / n_val, writer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the UNet on images and target masks',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--load', dest='load', type=str, default='',
                        help='Load model from a .pth file')
    parser.add_argument('-g', '--gpus', dest='gpus', type=str, default='8,7',
                        help='Gpus')
    parser.add_argument('-e', '--epoch', dest='epoch', type=int, default=0,
                        help='Gpus')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=256,
                        help='Downscaling factor of the images')
    parser.add_argument('-n', '--nickname', dest='nickname', type=str, default='E00',
                        help='Nickname')
    args = parser.parse_args()

    val = BasicDataset('img.train.list', augment=False, scale=args.scale)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device {}'.format(device))
    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    logging.info('Network:\n\t{} input channels\n\t{} output channels (classes)\n\t{} upscaling'.format(net.n_channels, net.n_classes, "Bilinear" if net.bilinear else "Transposed conv"))
    net.load_state_dict(
        torch.load(args.load, map_location=device)
    )
    logging.info('Model loaded from {}'.format(args.load))
    net.to(device=device)
    val_score, val_loss, writer = eval_net(net, val_loader, device, nickname=args.nickname, epoch=args.epoch, writer=None)
    print('Validation Loss:{}'.format(val_loss))
    print('Validation Dice Coeff: {}'.format(val_score))
