# -*-coding:utf-8-*-
import os
import time
import yaml
import shutil
import random
import warnings
import argparse
import logging
import datetime

import torch
import torchvision
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
from torchvision import transforms as TF

from PIL import Image
from PIL import ImageFilter

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR Pretrain')
parser.add_argument('--data_sort', default=r'MSTAR', type=str, help="option for data_type; 'UIUC_Sports', 'NEU_CLS', 'MSTAR', '15_Scene'")
parser.add_argument('--pre_data', default=r'', type=str, help='path to dataset')
parser.add_argument('--log_dir', default=r'', type=str, help='path to save state_dict')
parser.add_argument('--batch_size', default=512, type=int, help='mini-batch size (default: 256), this is the total')
parser.add_argument('--temperature', default=0.5, type=float, help='softmax temperature (default: 0.5)')

# options for the hyparamter of Self_Supervised loss
parser.add_argument('--hyp_CL', type=float, help='The weight of SimCLR_loss')
parser.add_argument('--hyp_BT', type=float, help='The weight of BT_loss')
parser.add_argument('--lambd', type=float, help='weight on off-diagonal terms')
parser.add_argument('--hyp_E', type=float, help='The weight of Entropy_loss')

# options for Self_Supervised head
parser.add_argument('--CL_out', type=int, help='feature dimension (default: 128)')
parser.add_argument('--BT_out', type=int, help='feature dimension (default: 512)')
parser.add_argument('--E_out', type=int, help='feature dimension (default: 512)')

parser.add_argument('--seed', default=10, type=int, help='seed for initializing training. ')
parser.add_argument('--epochs', default=1000, type=int, help='number of total epochs to run')
parser.add_argument('--lr', default=0.03, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--save_freq', default=100, type=int, help='save state_dict frequency (default: 10)')
parser.add_argument('--arch', default='resnet18', help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('--workers', default=12, type=int, help='number of data loading workers (default: 32)')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--n_views', default=2, type=int, help='Number of views for contrastive learning training.')
parser.add_argument('--device', default=torch.device('cuda:3' if torch.cuda.is_available() else 'cpu'), help='GPU id to use.')


def main():
    logging.info('Using device {} for training'.format(args.device))
    logging.info("args:{}".format(args))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."

    dataset_simclr = dataset_SimCLR(args.pre_data, args.data_sort)
    train_loader = torch.utils.data.DataLoader(dataset_simclr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    model = model_Pretrian(base_model_name=args.arch, CL_num=args.CL_out, BT_num=args.BT_out, E_num=args.E_out)
    logging.info("Pretrain_model:{}".format(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
    simclr.train(train_loader)


class SimCLR(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        # normalization layer for the representations z1 and z2 for BarlowTwins
        self.BT_bn = nn.BatchNorm1d(self.args.BT_out, affine=False).to(self.args.device)  # 每个样本的特征维度, # N*M中的M

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        # print('positive:', positives.T[0][0:20])

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        # print('negatives:', negatives[0][0:20])

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature

        loss_infoNCE = self.criterion(logits, labels)

        return loss_infoNCE

    def BarlowTwins_loss(self, features):

        # compute embeddings
        features_tuple = torch.chunk(features, chunks=2, dim=0)
        feature_i, feature_j = features_tuple[0], features_tuple[1]

        N = feature_i.shape[0]
        D = feature_i.shape[1]

        # normalize representation along the batch dimension and cross-correlation matrix
        c = self.BT_bn(feature_i).T @ self.BT_bn(feature_j) / N

        # compyt loss
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss_BT = on_diag + self.args.lambd * off_diag

        return loss_BT

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def Entropy_loss_2D(self, features, epsilon=1e-08):
        P = torch.softmax(features, dim=1)
        H_X = torch.sum((P * (- torch.log2(P + epsilon))), dim=1)
        loss = torch.exp(-torch.mean(H_X))
        # loss = 1 / torch.mean(H_X)
        return loss

    def train(self, train_loader):

        # # save config file
        # save_config_file(self.writer.log_dir, self.args)
        for epoch_counter in range(self.args.epochs):
            for batch, (img_i, img_j) in enumerate(train_loader):
                images = torch.cat((img_i, img_j), dim=0)
                images = images.to(self.args.device)

                self.optimizer.zero_grad()
                # self.model.backbone.avgpool.register_forward_hook(get_activation('backbone.avgpool'))
                # self.model.backbone.fc[0].register_forward_hook(get_activation('backbone.1linear'))
                features_CL, features_BT = self.model(images)

                SimCLR_loss = self.args.hyp_CL * self.info_nce_loss(features_CL)
                E_loss = self.args.hyp_E * self.Entropy_loss_2D(features_CL)
                BT_loss = self.args.hyp_BT * self.BarlowTwins_loss(features_BT)
                loss = SimCLR_loss + E_loss + BT_loss
                # torch.set_printoptions(profile='full')
                # logging.info('____________________features____________________:{}'.format(features))
                loss.backward()
                self.optimizer.step()
                logging.info('epoch:({}-{}) lr: {:.6f} SimCLR_loss: {:.6f} E_loss: {:.6f} BT_loss: {:.6f} loss: {:.6f}'
                             .format(epoch_counter + 1, batch, self.scheduler.get_lr()[0], SimCLR_loss, E_loss, BT_loss, loss))

            # warmup for the first 10 epochs
            if (epoch_counter + 1) >= 10:
                self.scheduler.step()

            # if (epoch_counter + 1) % args.save_freq == 0:
            if epoch_counter + 1 == 300 or epoch_counter + 1 == 500 or epoch_counter + 1 == 1000:
                save_checkpoint({'epoch': epoch_counter + 1, 'arch': args.arch, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()},
                                args.log_dir,
                                filename='BTD_{}_hypBT_{}_lambd_{}_epoch_{:04d}.pth.tar'.format(args.BT_out, args.hyp_BT, args.lambd, epoch_counter + 1))
                logging.info('epoch{:04d}.pth.tar saved!'.format(epoch_counter + 1))

        logging.info("Training has finished.")


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output

    return hook


def save_checkpoint(state, log_dir, filename):
    filename_path = os.path.join(log_dir, filename)
    torch.save(state, filename_path)
    print(filename + ' has been saved.')


#######################################################data_augmentation##################################################################
class dataset_SimCLR(Dataset):
    def __init__(self, root: str, data_sort: str):
        self.root = root
        self.imgs_path = self.get_img_paths(root)
        if data_sort == 'UIUC_Sports':
            self.transform = TF.Compose(
                [
                    TF.Resize((224, 224)),
                    TF.RandomResizedCrop(224, scale=(0.90, 1.0)),
                    TF.RandomHorizontalFlip(),
                    TF.RandomApply([TF.ColorJitter(0.1, 0.8, 0.8, 0.2)], p=0.8),
                    TF.RandomApply([TF.RandomGrayscale()], p=0.2),
                    TF.ToTensor(),
                    TF.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])

        elif data_sort == 'NEU_CLS':
            self.transform = TF.Compose(
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

        elif data_sort == '15_Scene':
            self.transform = TF.Compose(
                [
                    TF.Resize((224, 224)),
                    TF.RandomResizedCrop(224, scale=(0.9, 1.0)),
                    TF.RandomHorizontalFlip(),
                    TF.RandomApply([TF.RandomRotation(30)], p=0.8),
                    TF.RandomApply([TF.ColorJitter(0.1, 0.8, 0.8, 0.2)], p=0.8),
                    TF.ToTensor(),
                    TF.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])

        elif data_sort == 'MSTAR':
            self.transform = TF.Compose(
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

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, item):
        start_time = time.perf_counter()
        path = self.imgs_path[item]
        with open(path, 'rb') as f:
            with Image.open(f) as origin_img:
                origin_img = origin_img.convert('RGB')  ## 如果不使用，convert('RGB')进行转换的话，读出来的图像是RGBA四通道的，A通道为透明通道
                # print('origin_img', origin_img)

        if self.transform is not None:
            img_i = self.transform(origin_img)
            img_j = self.transform(origin_img)

            end_time = time.perf_counter()
            # print('end_time - start_time', end_time - start_time)
            return img_i, img_j

    def get_img_paths(self, root):
        imgs = []
        for subset in os.listdir(root):
            subset_path = os.path.join(root, subset)
            for img_name in os.listdir(subset_path):
                imgs.append(os.path.join(subset_path, img_name))
        random.shuffle(imgs)
        return imgs


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


# **************************************** model_Pretrain for MT_SimCLR ******************************************* #
class BaseSimCLRException(Exception):
    """Base exception"""


class InvalidBackboneError(BaseSimCLRException):
    """Raised when the choice of backbone Convnet is invalid."""


class InvalidDatasetSelection(BaseSimCLRException):
    """Raised when the choice of dataset is invalid."""


class model_Pretrian(nn.Module):
    def __init__(self, base_model_name, CL_num, BT_num, E_num):
        super(model_Pretrian, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=128),
                            "resnet34": models.resnet34(pretrained=False, num_classes=128),
                            "resnet50": models.resnet50(pretrained=False, num_classes=128)}

        # delete the fc layer of resnet.backbone
        self.backbone = nn.Sequential()
        base_model = self._get_basemodel(base_model_name)
        for name, module in base_model.named_children():
            if name != 'fc':
                self.backbone.add_module(name, module)

        # add mlp projection head
        dim_mlp = self._get_basemodel(base_model_name).fc.in_features
        self.head_CL = nn.Sequential(nn.Linear(dim_mlp, 512), nn.ReLU(), nn.Linear(512, CL_num))
        self.head_E = nn.Sequential(nn.Linear(dim_mlp, 512), nn.ReLU(), nn.Linear(512, E_num))
        self.head_BT = nn.Sequential(
            nn.Linear(dim_mlp, BT_num, bias=False),
            nn.BatchNorm1d(BT_num),
            nn.ReLU(inplace=True),
            nn.Linear(BT_num, BT_num, bias=False),
            nn.BatchNorm1d(BT_num),
            nn.ReLU(inplace=True),
            nn.Linear(BT_num, BT_num, bias=False),
        )

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError("Invalid backbone architecture. Check the config file and pass one of: resnet18, resnet34 or resnet50")
        else:
            return model

    def forward(self, x):
        h = self.backbone(x)
        h = torch.flatten(h, 1)

        z_CL = self.head_CL(h)
        z_BT = self.head_BT(h)
        # z_E = self.head_E(h)

        return z_CL, z_BT


#######################################################log_file##################################################################
def init_logging(filedir: str):
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    logger = logging.getLogger()
    fh = logging.FileHandler(filename=filedir + '/log_' + get_date_str() + '.txt')
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


if __name__ == "__main__":
    args = parser.parse_args()
    logging = init_logging(args.log_dir)
    main()