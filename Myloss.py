import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16



class L_std(nn.Module):

    def __init__(self):
        super(L_std, self).__init__()

    def forward(self, org, enhance):

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_std = torch.std(org_mean, [2, 3])
        enhance_std = torch.std(enhance_mean, [2, 3])

        s = torch.pow(enhance_std - torch.max(org_std / 5, torch.full(org_std.shape, 0.005).cuda()), 2)

        return s

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        # avoid sqrt 0
        # k = Drg + Drb + Dgb

        return k


class L_color_ratio(nn.Module):

    def __init__(self):
        super(L_color_ratio, self).__init__()

    def forward(self, x, ori):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)

        ori_mean_rgb = torch.mean(ori, [2, 3], keepdim=True)
        ori_mr, ori_mg, ori_mb = torch.split(ori_mean_rgb, 1, dim=1)
        Drg = torch.pow((mr / mg) - (ori_mr / ori_mg), 2)
        Drb = torch.pow((mr / mb) - (ori_mr / ori_mb), 2)
        Dgb = torch.pow((mb / mg) - (ori_mb / ori_mg), 2)
        # k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        k = Drg + Drb + Dgb

        return k


class L_spa(nn.Module):

    def __init__(self, clip=1.):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
        self.clip = (-clip, clip)

    def forward(self, org, enhance):
        b, c, h, w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        # reduce std of input img
        # org_mean_m = torch.mean(org_mean, [2, 3], keepdim=True)
        # org_mean_s = torch.std(org_mean, [2, 3], keepdim=True)
        #
        # org_mean = (org_mean - org_mean_m) / org_mean_s
        # org_mean = org_mean * torch.min(torch.max(org_mean_s / 5, torch.full(org_mean_s.shape, 0.005).cuda()), org_mean_s) + org_mean_m

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)
        # org_pool = org_mean
        # enhance_pool = enhance_mean

        weight_diff = torch.max(
            torch.FloatTensor([1]).cuda() + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),
                                                              torch.FloatTensor([0]).cuda()),
            torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool - org_pool)

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        # scale = 0.05
        # # use log to reduce spatial diff
        # D_org_letf_s = torch.sign(D_org_letf)
        # D_org_right_s = torch.sign(D_org_right)
        # D_org_up_s = torch.sign(D_org_up)
        # D_org_down_s = torch.sign(D_org_down)
        #
        # D_org_letf = D_org_letf_s * scale * torch.log10((9 * torch.abs(D_org_letf)) + 1)
        # D_org_right = D_org_right_s * scale * torch.log10((9 * torch.abs(D_org_right)) + 1)
        # D_org_up = D_org_up_s * scale * torch.log10((9 * torch.abs(D_org_up)) + 1)
        # D_org_down = D_org_down_s * scale * torch.log10((9 * torch.abs(D_org_down)) + 1)
        # use quad to reduce spatial diff
        # D_org_letf = D_org_letf_s * scale * torch.pow(torch.abs(D_org_letf), 2)
        # D_org_right = D_org_right_s * scale * torch.pow(torch.abs(D_org_right), 2)
        # D_org_up = D_org_up_s * scale * torch.pow(torch.abs(D_org_up), 2)
        # D_org_down = D_org_down_s * scale * torch.pow(torch.abs(D_org_down), 2)

        # clamp spatial diff
        # D_org_letf = torch.clamp(D_org_letf, self.clip[0], self.clip[1])
        # D_org_right = torch.clamp(D_org_right, self.clip[0], self.clip[1])
        # D_org_up = torch.clamp(D_org_up, self.clip[0], self.clip[1])
        # D_org_down = torch.clamp(D_org_down, self.clip[0], self.clip[1])
        # D_org_letf = D_org_letf / 2.
        # D_org_right = D_org_right / 2.
        # D_org_up = D_org_up / 2.
        # D_org_down = D_org_down / 2.


        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E

class L_spa_rgb(nn.Module):

    def __init__(self, clip=1.):
        super(L_spa_rgb, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
        self.clip = (-clip, clip)

    def forward(self, org, enhance):
        b, c, h, w = org.shape
        ET = 0.

        for i in range(c):

            org_mean = org[:,i:i+1,:,:]
            enhance_mean = enhance[:, i:i + 1, :, :]

            org_pool = self.pool(org_mean)
            enhance_pool = self.pool(enhance_mean)

            D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
            D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
            D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
            D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

            scale = 0.1
            # use log to reduce spatial diff
            D_org_letf_s = torch.sign(D_org_letf)
            D_org_right_s = torch.sign(D_org_right)
            D_org_up_s = torch.sign(D_org_up)
            D_org_down_s = torch.sign(D_org_down)

            D_org_letf = D_org_letf_s * scale * torch.log10((9 * torch.abs(D_org_letf)) + 1)
            D_org_right = D_org_right_s * scale * torch.log10((9 * torch.abs(D_org_right)) + 1)
            D_org_up = D_org_up_s * scale * torch.log10((9 * torch.abs(D_org_up)) + 1)
            D_org_down = D_org_down_s * scale * torch.log10((9 * torch.abs(D_org_down)) + 1)
            # use quad to reduce spatial diff
            # D_org_letf = D_org_letf_s * scale * torch.pow(torch.abs(D_org_letf), 2)
            # D_org_right = D_org_right_s * scale * torch.pow(torch.abs(D_org_right), 2)
            # D_org_up = D_org_up_s * scale * torch.pow(torch.abs(D_org_up), 2)
            # D_org_down = D_org_down_s * scale * torch.pow(torch.abs(D_org_down), 2)

            # clamp spatial diff
            # D_org_letf = torch.clamp(D_org_letf, self.clip[0], self.clip[1])
            # D_org_right = torch.clamp(D_org_right, self.clip[0], self.clip[1])
            # D_org_up = torch.clamp(D_org_up, self.clip[0], self.clip[1])
            # D_org_down = torch.clamp(D_org_down, self.clip[0], self.clip[1])
            # D_org_letf = D_org_letf / 2.
            # D_org_right = D_org_right / 2.
            # D_org_up = D_org_up / 2.
            # D_org_down = D_org_down / 2.


            D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
            D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
            D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
            D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

            D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
            D_right = torch.pow(D_org_right - D_enhance_right, 2)
            D_up = torch.pow(D_org_up - D_enhance_up, 2)
            D_down = torch.pow(D_org_down - D_enhance_down, 2)
            E = (D_left + D_right + D_up + D_down)
            # E = 25*(D_left + D_right + D_up +D_down)
            ET += E
        return ET

class L_exp(nn.Module):

    def __init__(self, patch_size, mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        if mean_val < 0:
            print('set random exposure for darkening')
        self.mean_val = mean_val

    def forward(self, x):
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        mean_val = self.mean_val
        # random exposure value
        if mean_val < 0:
            mean_val = random.gauss(0.3, 0.1/3)
        d = torch.mean(torch.pow(mean - torch.FloatTensor([mean_val]).cuda(), 2))
        return d


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x, img=None):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2)
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2)
        if img is not None:
            ori_shape = h_tv.shape
            ih_x = x.size()[2]
            iw_x = x.size()[3]
            gray = torch.mean(img, 1, keepdim=True)
            ih_tv = torch.abs((gray[:, :, 1:, :] - gray[:, :, :ih_x - 1, :]))
            iw_tv = torch.abs((gray[:, :, :, 1:] - gray[:, :, :, :iw_x - 1]))
            h_tv = h_tv * (torch.max(ih_tv).item() - ih_tv)
            w_tv = w_tv * (torch.max(iw_tv).item() - iw_tv)
            assert h_tv.shape == ori_shape
        h_tv = h_tv.sum()
        w_tv = w_tv.sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)

    def forward(self, x):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b, c, h, w = x.shape
        # x_de = x.cpu().detach().numpy()
        r, g, b = torch.split(x, 1, dim=1)
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r - mr
        Dg = g - mg
        Db = b - mb
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Db, 2) + torch.pow(Dg, 2), 0.5)
        # print(k)

        k = torch.mean(k)
        return k


class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3

def proximity_loss(real_images, fake_images, p1=10.0, p2=1.0, reduction='mean'):
    """
        calcualtes proximity loss in c3lt.
    :param real_images:
    :param fake_images:
    :param p1:
    :param p2:
    :param reduction:
    :return:
    """
    # convert to gray
    real_images = torch.mean(real_images, 1, keepdim=True)
    fake_images = torch.mean(fake_images, 1, keepdim=True)

    masks = gen_masks(real_images, fake_images, mode='mse')
    L1 = nn.L1Loss(reduction=reduction)
    smooth = smoothness_loss(masks, reduction=reduction)
    entropy = entropy_loss(masks, reduction=reduction)
    prx = L1(real_images, fake_images)
    return (prx + p1 * smooth + p2 * entropy) / (1 + p1 + p2)

def smoothness_loss(masks, beta=2, reduction="mean"):
    """
        smoothness loss that encourages smooth masks.
    :param masks:
    :param beta:
    :param reduction:
    :return:
    """
    # TODO RGB images
    masks = masks[:, 0, :, :]
    a = torch.mean(torch.abs((masks[:, :-1, :] - masks[:, 1:, :]).view(masks.shape[0], -1)).pow(beta), dim=1)
    b = torch.mean(torch.abs((masks[:, :, :-1] - masks[:, :, 1:]).view(masks.shape[0], -1)).pow(beta), dim=1)
    if reduction == "mean":
        return (a + b).mean() / 2
    else:
        return (a + b).sum() / 2


def entropy_loss(masks, reduction="mean"):
    """
        entropy loss that encourages binary masks.
    :param masks:
    :param reduction:
    :return:
    """
    # TODO RGB images
    masks = masks[:, 0, :, :]
    b, h, w = masks.shape
    if reduction == "mean":
        return torch.minimum(masks.view(b, -1), 1.0 - masks.view(b, -1)).mean()
    else:
        return torch.minimum(masks.view(b, -1), 1.0 - masks.view(b, -1)).sum()

def gen_masks(inputs, targets, mode='abs'):
    """
        generates a difference masks give two images (inputs and targets).
    :param inputs:
    :param targets:
    :param mode:
    :return:
    """
    # TODO RGB images
    masks = targets - inputs
    masks = masks.view(inputs.size(0), -1)

    if mode == 'abs':
        masks = masks.abs()
        # normalize 0 to 1
        masks -= masks.min(1, keepdim=True)[0]
        masks /= masks.max(1, keepdim=True)[0]

    elif mode == "mse":
        masks = masks ** 2
        masks -= masks.min(1, keepdim=True)[0]
        masks /= masks.max(1, keepdim=True)[0]

    elif mode == 'normal':
        # normalize -1 to 1
        min_m = masks.min(1, keepdim=True)[0]
        max_m = masks.max(1, keepdim=True)[0]
        masks = 2 * (masks - min_m) / (max_m - min_m) - 1

    else:
        raise ValueError("mode value is not valid!")

    return masks.view(inputs.shape)