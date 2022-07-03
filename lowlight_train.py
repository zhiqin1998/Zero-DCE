import argparse
import os

import torch
import torch.optim

import Myloss
import dataloader
import model

from PIL import Image
import cv2
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def nanmean(v):
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum() / (~is_nan).float().sum()

def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    DCE_net = model.enhance_net_nopool_shiftrgb().cuda()

    DCE_net.apply(weights_init)
    if config.load_pretrain == True:
        DCE_net.load_state_dict(torch.load(config.pretrain_dir))
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path, size=config.image_size, preload=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)

    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa(clip=0.01)
    L_std = Myloss.L_std()
    L1 = torch.nn.L1Loss()
    L_prox = Myloss.proximity_loss
    L_exp = Myloss.L_exp(16, config.exposure)
    L_TV = Myloss.L_TV()

    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    DCE_net.train()

    for epoch in range(config.num_epochs):
        for iteration, img_lowlight in enumerate(train_loader):

            img_lowlight = img_lowlight.cuda()

            enhanced_image_1, enhanced_image, A, B = DCE_net(img_lowlight)

            # Loss_TV = 200 * L_TV(A, img_lowlight)
            Loss_TV = 200 * L_TV(A)
            Loss_TV_B = 200 * L_TV(B) + L1(B, torch.zeros_like(B))

            loss_spa = torch.mean(L_spa(img_lowlight, enhanced_image))
            # loss_std = torch.mean(L_std(img_lowlight, enhanced_image))

            loss_col = 5 * torch.mean(L_color(enhanced_image))
            # loss_col = 5 * nanmean(L_color(enhanced_image))
            # if torch.isnan(loss_col).any().item():
            #     print('warning color loss is all nan')
            #     loss_col = 0

            loss_exp = 10 * torch.mean(L_exp(enhanced_image))

            # best_loss
            print(Loss_TV.item(), loss_spa.item(), loss_col.item(), loss_exp.item())
            loss = Loss_TV + Loss_TV_B + loss_spa + loss_col + loss_exp
            # loss = Loss_TV + loss_col + loss_exp

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(DCE_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            if ((iteration + 1) % config.display_iter) == 0:
                print("Loss at iteration", iteration + 1, ":", loss.item())
        torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')

    if True:
        # remove all weight except last epoch
        for x in os.listdir(config.snapshots_folder):
            if str(epoch) not in x:
                os.remove(os.path.join(config.snapshots_folder, x))

        # generate test image
        save_dir = os.path.join(config.snapshots_folder, 'test')
        os.makedirs(save_dir, exist_ok=True)
        for file in os.listdir('data/test_data/ICDAR'):
            img_path = os.path.join('data/test_data/ICDAR', file)
            gt_img = Image.open(img_path)
            gt_img = (np.asarray(gt_img) / 255.0)
            input_img = torch.from_numpy(gt_img).float()
            input_img = input_img.permute(2, 0, 1)
            input_img = input_img.cuda().unsqueeze(0)
            with torch.no_grad():
                outputs = DCE_net(input_img)
                enhanced_image = outputs[1]
                enhanced_image = enhanced_image.cpu().squeeze().permute(1, 2, 0).detach().numpy()
            lowlight = Image.fromarray(np.uint8(enhanced_image * 255))
            he_rgb = Image.fromarray(np.uint8(hist_rgb(enhanced_image.copy()) * 255))
            ls_rgb = Image.fromarray(np.uint8(linear_scale_rgb(enhanced_image.copy()) * 255))
            lowlight.save(os.path.join(save_dir, file[:-4] + '_low.jpg'))
            he_rgb.save(os.path.join(save_dir, file[:-4] + '_he.jpg'))
            ls_rgb.save(os.path.join(save_dir, file[:-4] + '_ls.jpg'))


def hist_rgb(img):
    img = (img * 255).astype(np.uint8)
    # equalize histogram of each channel
    img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    img[:,:,1] = cv2.equalizeHist(img[:,:,1])
    img[:,:,2] = cv2.equalizeHist(img[:,:,2])
    return img / 255.

def linear_scale_rgb(img):
    # linear scale of each channel
    img = (img * 255).astype(np.uint8).astype(np.float32)
    img[:,:,0] = (img[:,:,0] - img[:,:,0].min()) / (img[:,:,0].max() - img[:,:,0].min())
    img[:,:,1] = (img[:,:,1] - img[:,:,1].min()) / (img[:,:,1].max() - img[:,:,1].min())
    img[:,:,2] = (img[:,:,2] - img[:,:,2].min()) / (img[:,:,2].max() - img[:,:,2].min())
    return img


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="snapshots/Epoch99.pth")
    parser.add_argument('--exposure', type=float, default=0.6)
    parser.add_argument('--image_size', type=int, default=256)

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)
