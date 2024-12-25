import os
import random
import logging
import datetime
import warnings
import argparse
import numpy as np

import torch
import torchvision
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
from torchvision import datasets
import torch.utils.data.distributed
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch lin_cls')
parser.add_argument('--data_sort', default=r'MSTAR', type=str, help="option for data_type; 'UIUC_Sports', 'NEU_CLS','MSTAR', '15_Scene'")
parser.add_argument('--pretrained', default=r'', type=str, help='path to pretrained checkpoint')
parser.add_argument('--traindir', default=r'/media/wenqiwang/file/ubuntu/data_set/Classification/00subset/MSTAR/5_shots_noise_0.0', help='path to load train_data')
parser.add_argument('--valdir', default=r'/media/wenqiwang/file/ubuntu/data_set/Classification/MSTAR/test', help='path to load val_data')
parser.add_argument('--log_dir', default=r'', type=str, help='path to save state_dict')
parser.add_argument('--classes', default=10, type=int, help='number of class')
parser.add_argument('--softmax_lincls', default=False, type=bool, help='option for linear evaluation; True for linear evaluation and False for fine-tune')
parser.add_argument('--pretrain_model', default=r'', type=str, help="option for pretrian_model; 'BIDFC', 'SimCLR','MoCo', 'BYOL', 'SimSiam', 'BarlowTwins', 'MT_CL'")
parser.add_argument('--seed', default=10, type=int, help='seed for initializing training.')
parser.add_argument('--workers', default=12, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, help='number of total iteration to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=512, type=int, help='mini-batch size (default: 256), this is the total ' 'batch size of all GPUs on the current node when ' 'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', default=0.03, type=float, help='initial learning rate')
parser.add_argument('--arch', default='resnet18', help='simclrmodel architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('--schedule', default=[200], type=int, help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--SGD', default=False, type=bool, help='option for optimizer(SGD or Adam)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD solver')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--device', default=torch.device("cuda:2" if torch.cuda.is_available() else "cpu"), help='torch.device("cuda:0" if torch.cuda.is_available() else "cpu")')
parser.add_argument('--MLClassifier', default=None, type=str, help='classifier from machine learning; option: KNN,logistic_regression')
parser.add_argument('--strong_augmentation', default=True, type=bool, help='option for strong augmentation or weak augmentation')

acc_list = []
loss_list = []

def main():
    logging.info(f"args: {args}\t")
    best_acc = 0

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. ' 'This will turn on the CUDNN deterministic setting, ' 'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting ' 'from checkpoints.')

    traintransformer = transforms.ToTensor()
    validtransformer = transforms.ToTensor()
    if args.strong_augmentation:
        if args.data_sort == 'UIUC_Sports':
            traintransformer = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224, scale=(0.80, 1.00)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.RandomRotation(10)], p=0.4),
                # transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        elif args.data_sort == 'NEU_CLS':
            traintransformer = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224, scale=(0.80, 1.00)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.RandomRotation(10)], p=0.4),
                # transforms.RandomApply([transforms.ColorJitter(brightness=0.2)], p=1),
                # transforms.RandomApply([transforms.ColorJitter(contrast=0.2)], p=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        elif args.data_sort == '15_Scene':
            traintransformer = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224, scale=(0.80, 1.00)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.RandomRotation(10)], p=0.4),
                # transforms.RandomApply([transforms.ColorJitter(brightness=0.2)], p=1),
                # transforms.RandomApply([transforms.ColorJitter(contrast=0.2)], p=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        elif args.data_sort == 'MSTAR':
            traintransformer = transforms.Compose([
                # transforms.Grayscale(3),
                transforms.RandomResizedCrop(64, scale=(0.90, 1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomApply([transforms.RandomRotation(10)], p=0.4),
                # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4)], p=0.4),
                transforms.ToTensor(),
                # transforms.RandomApply([GaussianNoise()], p=0.2),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    else:
        if args.data_sort == 'UIUC_Sports':
            traintransformer = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        elif args.data_sort == 'NEU_CLS':
            traintransformer = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        elif args.data_sort == '15_Scene':
            traintransformer = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        elif args.data_sort == 'MSTAR':
            traintransformer = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    if args.data_sort == 'UIUC_Sports':
        validtransformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    elif args.data_sort == 'NEU_CLS':
        validtransformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    elif args.data_sort == '15_Scene':
        validtransformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    elif args.data_sort == 'MSTAR':
        validtransformer = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    # Data loading code
    # Prepare for Training Configuration
    # traindir = os.path.join(args.path, 'train')
    train_dataset = datasets.ImageFolder(args.traindir, traintransformer)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # Prepare for Validation Configuration
    # valdir = os.path.join(args.path, 'test')
    val_dataset = datasets.ImageFolder(args.valdir, validtransformer)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # create downstream model
    print('=> creating downstream model')
    model = models.resnet18(pretrained=False, num_classes=args.classes).to(args.device)
    # print(model)

    # load from pre-trained_model weight
    if args.pretrained is not None:
        if os.path.isfile(args.pretrained):
            logging.info('=> loading checkpoint {}'.format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):

                if args.pretrain_model == 'BIDFC':
                    if k.startswith('backbone.'):
                        if k.startswith('backbone') and not k.startswith('backbone.fc'):
                            state_dict[k[len("backbone."):]] = state_dict[k]
                    del state_dict[k]

                elif args.pretrain_model == "MoCo":
                    if k.startswith('online_network.'):
                        if k.startswith('online_network') and not k.startswith('online_network.fc'):
                            state_dict[k[len("online_network."):]] = state_dict[k]
                    del state_dict[k]

                elif args.pretrain_model == "SimCLR" or args.pretrain_model == "BarlowTwins" or args.pretrain_model == "MT_CL" or args.pretrain_model == "BYOL":
                    if k.startswith('backbone.'):
                        state_dict[k[len("backbone."):]] = state_dict[k]
                    del state_dict[k]

                elif args.pretrain_model == "SimSiam":
                    if k.startswith('encoder.'):
                        if k.startswith('encoder') and not k.startswith('encoder.fc'):
                            state_dict[k[len("encoder."):]] = state_dict[k]
                    del state_dict[k]

            log = model.load_state_dict(state_dict, strict=False)
            assert log.missing_keys == ['fc.weight', 'fc.bias']  # Python的assert是用来检查一个条件，如果它为真，就不做任何事。如果它为假，则会抛出AssertError并且包含错误信息。
            logging.info("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.pretrained))


    # if args.MLClassifier:
    #     MLClassifier(model,train_loader,val_loader,args, print_info=True)
    #     return

    # freeze all layers but the last fc
    if args.softmax_lincls:
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        logging.info("=> Linear Evaluation Mode.")
    else:
        parameters = model.parameters()
        logging.info("=> Fine Tuning Mode.")

    logging.info("=> train_dataset:{}".format(args.traindir))
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)
    if args.SGD:
        optimizer = torch.optim.SGD(parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(parameters, args.lr)

    model.to(args.device)
    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        train_info = train(train_loader, model, criterion, optimizer, args)
        # evaluate on validation set
        val_info, acc, Loss = validate(val_loader, model, criterion, args)

        logging.info("[epoch {}] {} {}".format(epoch + 1, train_info, val_info))
        # remember best acc and save checkpoint
        acc_list.append(acc)
        loss_list.append(Loss)
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if is_best:
            save_checkpoint({'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(), 'best_acc1': best_acc, 'optimizer' : optimizer.state_dict()},
                            args.log_dir)
            print('Best checkpoint has been saved!')

    logging.info('Done!')
    logging.info("The best acc is {}".format(best_acc))

def train(train_loader, model, criterion, optimizer, args):
    model.train()

    correct = 0.0
    running_loss = 0.0
    for i,(data,target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        pred = out.argmax(dim=1)
        correct += torch.eq(pred, target).sum().float().item()
        running_loss += loss.item()
    running_loss = running_loss / len(train_loader)
    correct = correct / len(train_loader.dataset) * 100
    return "Train : loss: {:.6f}  avg_acc:{:.2f}%".format(running_loss, correct)

def validate(val_loader, model, criterion, args):
    model.eval()
    correct = 0.0
    running_loss = 0.0
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(args.device), target.to(args.device)
            out = model(data)
            loss = criterion(out,target)

            # measure accuracy and record loss
            pred = out.argmax(dim=1)
            correct += torch.eq(pred, target).sum().float().item()
            running_loss += loss.item()
        running_loss = running_loss / len(val_loader)
        correct = correct / len(val_loader.dataset)*100
        return "Test : loss: {:.6f}  avg_acc:{:.2f}%".format(running_loss, correct), correct, running_loss

def save_checkpoint(state, log_dir, filename='Best_ACC.pth.tar'):
    filename = os.path.join(log_dir, filename)
    torch.save(state, filename)

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class GaussianNoise(object):
    """Gaussian Noise Augmentation for tensor"""

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, img):
        noise = torch.randn(1,img.shape[1],img.shape[2])
        noise = torch.cat([noise,noise,noise],dim=0)
        return torch.clamp(img + self.sigma * noise,0.0,1.0)


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