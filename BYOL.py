# -*-coding:utf-8-*-
import os
import random
import logging
import argparse
import datetime
from PIL import ImageFilter

import torch
import torchvision
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from torchvision import transforms as TF
from torch.utils.data.dataloader import DataLoader

parser = argparse.ArgumentParser(description='PyTorch MoCo Training')
parser.add_argument('--seed', default=10, type=int, help='seed for initializing training. ')
parser.add_argument('--data_sort', default=r'MSTAR', type=str, help="option for data_type; 'UIUC_Sports', 'NEU_CLS','MSTAR'")
parser.add_argument('--pre_data', default=r'', type=str, help='path to dataset')
parser.add_argument('--log_dir', default=r'', type=str, help='path to save state_dict')
parser.add_argument('--epochs', type=int, help='number of total epochs to run')
parser.add_argument('--save_freq', type=int, help='save state_dict frequency (default: 10)')
parser.add_argument('--batch_size', default=512, type=int, help='mini-batch size (default: 512), this is the total ')

parser.add_argument('--mlp_hidden_size', default=512, type=int, help='feature dimension (default: 512)')
parser.add_argument('--projection_size', default=128, type=int, help='feature dimension (default: 512)')
parser.add_argument('--BYOL_m', default=0.999, type=float, help='BYOL momentum of updating key encoder (default: 0.999)')

parser.add_argument('--workers', default=12, type=int, help='number of data loading workers (default: 32)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD solver')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--lr', default=0.03, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--arch', default='resnet18', type=str, help='model architecture')
parser.add_argument('--device', default=torch.device('cuda:3' if torch.cuda.is_available() else 'cpu'), help='GPU id to use.')


def main(args):
    logging.info(f"args: {args}\t")
    logging.info('Using device {} for training'.format(args.device))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True

    # create model
    print("=> creating model '{}'".format(args.arch))
    # online network
    online_network = model_Pretrain(args.arch, args.mlp_hidden_size, args.projection_size).to(args.device)
    # predictor network
    predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features, mlp_hidden_size=args.mlp_hidden_size, projection_size=args.projection_size).to(args.device)
    # target encoder
    target_network = model_Pretrain(args.arch, args.mlp_hidden_size, args.projection_size).to(args.device)
    logging.info("Pretrain_online:\n{}\nPredictor:\n{}\nPretrain_tatget_net:\n{}".format(online_network, predictor, target_network))

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
    optimizer = torch.optim.SGD(params=list(online_network.parameters()) + list(predictor.parameters()),
                                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    trainer = BYOLTrainer(online_network=online_network, target_network=target_network, predictor=predictor, optimizer=optimizer, args=args)
    trainer.train(train_dataset)


####################################################### BYOL_Trainer ##################################################################
class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, args):
        self.online_network = online_network
        self.target_network = target_network
        self.predictor = predictor
        self.optimizer = optimizer

        self.device = args.device
        self.max_epochs = args.epochs
        self.m = args.BYOL_m
        self.batch_size = args.batch_size
        self.num_workers = args.workers

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False, shuffle=True)
        niter = 0
        self.initializes_target_network()
        for epoch_counter in range(self.max_epochs):
            for batch, (imgs_list, _) in enumerate(train_loader):
                batch_view_1 = imgs_list[0].to(self.device)
                batch_view_2 = imgs_list[1].to(self.device)

                # if niter == 0:
                #     grid = torchvision.utils.make_grid(batch_view_1[:32])
                #     self.writer.add_image('views_1', grid, global_step=niter)
                #
                #     grid = torchvision.utils.make_grid(batch_view_2[:32])
                #     self.writer.add_image('views_2', grid, global_step=niter)

                loss = self.update(batch_view_1, batch_view_2)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self._update_target_network_parameters()  # update the key encoder
                niter += 1
                logging.info('epoch:({}-{}) BYOL_loss: {:.6f}'.format(epoch_counter + 1, batch + 1, loss))

            if (epoch_counter + 1) % args.save_freq == 0:
                # if epoch_counter + 1 == 300 or epoch_counter + 1 == 500 or epoch_counter + 1 == 1000:
                save_checkpoint({'epoch': epoch_counter + 1, 'arch': args.arch, 'state_dict': self.online_network.state_dict(), 'optimizer': self.optimizer.state_dict()},
                                args.log_dir,
                                filename='epoch_{:04d}.pth.tar'.format(epoch_counter + 1))
                logging.info('epoch{:04d}.pth.tar saved!'.format(epoch_counter + 1))

        logging.info("Training has finished.")

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()

    # ############################## 保存模型需要修改 ##############################
    # def save_model(self, PATH):
    #     torch.save({
    #         'online_network_state_dict': self.online_network.state_dict(),
    #         'target_network_state_dict': self.target_network.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #     }, PATH)


def save_checkpoint(state, log_dir, filename):
    filename_path = os.path.join(log_dir, filename)
    torch.save(state, filename_path)
    print(filename + ' has been saved.')


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


####################################################### model_Pretrain ##################################################################
class model_Pretrain(torch.nn.Module):
    def __init__(self, arch_name: str, mlp_hidden_size: int, projection_size: int):
        super(model_Pretrain, self).__init__()
        resnet = None
        if arch_name == 'resnet18':
            resnet = models.resnet18(pretrained=False)
        elif arch_name == 'resnet50':
            resnet = models.resnet50(pretrained=False)
        else:
            KeyError("Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")

        self.backbone = torch.nn.Sequential()
        for name, module in resnet.named_children():
            if name != 'fc':
                self.backbone.add_module(name, module)
        self.projetion = MLPHead(in_channels=resnet.fc.in_features, mlp_hidden_size=mlp_hidden_size, projection_size=projection_size)

    def forward(self, x):
        h = self.backbone(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)


class MLPHead(torch.nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mlp_hidden_size),
            torch.nn.BatchNorm1d(mlp_hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

####################################################### log_file ##################################################################
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


# 利用权重初始化模拟训练网络的模型参数
def update_parmeter(model):
    for name, p in model.named_parameters():
        p.data.fill_(9999)


if __name__ == '__main__':
    args = parser.parse_args()
    logging = init_logging(args.log_dir)
    main(args)
