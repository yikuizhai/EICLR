#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import math
import time
import shutil
import random
import logging
import argparse
import builtins
import datetime
import warnings
from PIL import ImageFilter

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from torchvision import transforms as TF


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch MoCo Training')
parser.add_argument('--seed', default=10, type=int, help='seed for initializing training. ')
parser.add_argument('--data_sort', default=r'MSTAR', type=str, help="option for data_type; 'UIUC_Sports', 'NEU_CLS','MSTAR'")
parser.add_argument('--pre_data', default=r'', type=str, help='path to dataset')
parser.add_argument('--log_dir', default=r'', type=str, help='path to save state_dict')
parser.add_argument('--temperature', type=float, help='softmax temperature (default: 0.07)')
parser.add_argument('--epochs', type=int, help='number of total epochs to run')
parser.add_argument('--save_freq', type=int, help='save state_dict frequency (default: 10)')

parser.add_argument('--batch_size', default=512, type=int, help='mini-batch size (default: 512), this is the total ')
# options for HX
parser.add_argument('--lamda', default=0., type=float, help='The weight of Entropy_loss')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD solver')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--lr', default=0.03, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--arch', default='resnet18', type=str, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('--workers', default=12, type=int, help='number of data loading workers (default: 32)')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--device', default=torch.device('cuda:3' if torch.cuda.is_available() else 'cpu'), help='GPU id to use.')
# moco specific configs:
parser.add_argument('--moco_dim', default=128, type=int, help='feature dimension (default: 128)')
parser.add_argument('--moco_k', default=65536, type=int, help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco_m', default=0.999, type=float, help='moco momentum of updating key encoder (default: 0.999)')


def main():
    logging.info(f"args: {args}\t")
    logging.info('Using device {} for training'.format(args.device))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = MoCo(base_encoder=models.__dict__[args.arch], out_dim=args.moco_dim, queue_size=args.moco_k, m=args.moco_m, T=args.temperature, infoNCE_layer=2, HX_layer=2).to(args.device)
    print(model)
    model.to(args.device)

    # Data loading code
    augmentation = TF.ToTensor()
    if args.data_sort == 'UIUC_Sports':
        augmentation = TF.Compose(
            [
                TF.Resize((224, 224)),
                TF.RandomResizedCrop(224, scale=(0.90, 1.0)),
                TF.RandomHorizontalFlip(),
                TF.RandomApply([TF.ColorJitter(0.1, 0.8, 0.8, 0.2)], p=0.8),
                TF.RandomApply([TF.RandomGrayscale()], p=0.2),
                TF.ToTensor(),
                TF.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    elif args.data_sort == 'NEU_CLS':
        augmentation = TF.Compose(
            [
                TF.Resize((224, 224)),
                TF.RandomResizedCrop(224, scale=(0.8, 1.0)),
                TF.RandomHorizontalFlip(),
                TF.RandomApply([TF.RandomVerticalFlip()], p=0.5),
                TF.RandomApply([TF.ColorJitter(brightness=0.2)], p=1),
                TF.RandomApply([TF.ColorJitter(contrast=0.2)], p=1),
                TF.ToTensor(),
                TF.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    elif args.data_sort == '15_Scene':
        augmentation = TF.Compose(
            [
                TF.Resize((224, 224)),
                TF.RandomResizedCrop(224, scale=(0.9, 1.0)),
                TF.RandomHorizontalFlip(),
                TF.RandomApply([TF.RandomRotation(30)], p=0.8),
                TF.RandomApply([TF.ColorJitter(0.1, 0.8, 0.8, 0.2)], p=0.8),
                TF.ToTensor(),
                TF.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    elif args.data_sort == 'MSTAR':
        augmentation = TF.Compose(
            [
                TF.Resize((64, 64)),
                TF.Grayscale(3),
                TF.RandomResizedCrop(64, scale=(0.85, 1.0)),
                TF.RandomHorizontalFlip(),
                TF.RandomApply([TF.RandomRotation(30)], p=0.2),
                TF.RandomApply([TF.ColorJitter(0.4, 0.4, 0.4)], p=0.4),
                TF.ToTensor(),
                TF.RandomApply([GaussianNoise()], p=0.2),
                TF.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    train_dataset = datasets.ImageFolder(args.pre_data, TwoCropsTransform(augmentation))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
    for epoch in range(args.start_epoch, args.epochs):
        # switch to train mode
        model.train()
        for batch, (images, _) in enumerate(train_loader):
            optimizer.zero_grad()
            # compute output
            logits, labels, q_out = model(im_q=images[0].to(args.device), im_k=images[1].to(args.device))
            # print("similarity:", logits)
            # print("out_feature:", q_out)
            MoCo_loss = criterion(logits, labels.to(args.device))
            E_loss = args.lamda * Entropy_loss_2D(q_out)
            loss = MoCo_loss + E_loss
            loss.backward()
            optimizer.step()
            logging.info('epoch:({}-{}) MoCo_loss: {:.6f} E_loss: {:.6f} loss: {:.6f}'
                         .format(epoch + 1, batch, MoCo_loss, E_loss, loss))

        # # warmup for the first 10 epochs
        # if (epoch + 1) >= 10:
        #     scheduler.step()

        # if (epoch + 1) % args.save_freq == 0:
        if epoch + 1 == 300 or epoch + 1 == 500 or epoch + 1 == 1000:
            save_checkpoint({'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                            args.log_dir,
                            filename='lamda={}_tau_{}_epoch_{:04d}.pth.tar'.format(args.lamda, args.temperature, epoch + 1))
            logging.info('tau_{}_epoch{:04d}.pth.tar saved!'.format(args.temperature, epoch + 1))

    logging.info("Training has finished.")


def save_checkpoint(state, log_dir, filename):
    filename_path = os.path.join(log_dir, filename)
    torch.save(state, filename_path)
    print(filename + ' has been saved.')

def Entropy_loss_2D(features, epsilon=1e-08):
    P = torch.softmax(features, dim=1)
    H_X = torch.sum((P * (- torch.log2(P + epsilon))), dim=1)
    loss = torch.exp(-torch.mean(H_X))
    # loss = 1 / torch.mean(H_X)
    return loss

#######################################################MoCo_data_transform##################################################################
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR_80% https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.4, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=0.5))
        return x

class GaussianNoise(object):
    """Gaussian Noise Augmentation for tensor"""

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, img):
        noise = torch.randn(1, img.shape[1], img.shape[2])
        noise = torch.cat([noise, noise, noise], dim=0)
        return torch.clamp(img + self.sigma * noise, 0.0, 1.0)

#######################################################MoCo_Pretrain_model##################################################################
class MoCo(nn.Module):
    def __init__(self, base_encoder, out_dim:int, queue_size:int, m:float, T:float, infoNCE_layer:int, HX_layer:int):
        super(MoCo, self).__init__()
        self.queue_size = queue_size # K: queue size; number of negative keys (default: 65536)
        self.m = m # m: moco momentum of updating key encoder (default: 0.999)
        self.T = T # T: softmax temperature (default: 0.07)
        self.infoNCE_layer = infoNCE_layer
        self.HX_layer = HX_layer

        # create the encoders
        # num_classes is the output fc dimension
        self.online_network = base_encoder(num_classes=out_dim)
        self.target_network = base_encoder(num_classes=out_dim)

        dim_mlp = self.online_network.fc.weight.shape[1]
        self.online_network.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.online_network.fc)
        self.target_network.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.target_network.fc)

        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(out_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        # self.online_network.avgpool.register_forward_hook(get_activation('online_network.0projector'))
        # self.online_network.fc[0].register_forward_hook(get_activation('online_network.1linear'))
        q_out = self.online_network(im_q)  # queries: NxC
        # q_gap = torch.squeeze(activation['online_network.0projector'])
        # q_1linear = torch.squeeze(activation['online_network.1linear'])
        # if self.infoNCE_layer == 0:
        #     q = F.normalize(q_gap, dim=1)
        # elif self.infoNCE_layer == 1:
        #     q = F.normalize(q_1linear, dim=1)
        # elif self.infoNCE_layer == 2:
        #     q = F.normalize(q_out, dim=1)

        q = F.normalize(q_out, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            # self.target_network.avgpool.register_forward_hook(get_activation('target_network.0projector'))
            # self.target_network.fc[0].register_forward_hook(get_activation('target_network.1linear'))
            k_out = self.target_network(im_k)  # keys: NxC
            # k_gap = torch.squeeze(activation['online_network.0projector'])
            # k_1linear = torch.squeeze(activation['online_network.1linear'])
            # if self.infoNCE_layer == 0:
            #     k = F.normalize(k_gap, dim=1)
            # elif self.infoNCE_layer == 1:
            #     k = F.normalize(k_1linear, dim=1)
            # elif self.infoNCE_layer == 2:
            #     k = F.normalize(k_out, dim=1)
            k = F.normalize(k_out, dim=1)



        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', q, k).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', q, self.queue.clone().detach())
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits = logits / self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, q_out

        # if self.HX_layer == 0:
        #     return logits, labels, q_gap
        # elif self.HX_layer == 1:
        #     return logits, labels, q_1linear
        # elif self.HX_layer == 2:
        #     return logits, labels, q_out
        # else:
        #     return logits, labels, q_out


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

#######################################################log_file##################################################################
def init_logging(filedir:str):
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    logger = logging.getLogger()
    fh = logging.FileHandler(filename= filedir + '/log_' + get_date_str() + '.txt')
    sh = logging.StreamHandler()
    formatter_fh = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s')
    formatter_sh = logging.Formatter('%(message)s')
    # formatter_sh = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s')
    fh.setFormatter(formatter_fh)
    sh.setFormatter(formatter_sh)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(10)
    fh.setLevel(10)
    sh.setLevel(10)
    return logging

if __name__ == '__main__':
    args = parser.parse_args()
    logging = init_logging(args.log_dir)
    main()
