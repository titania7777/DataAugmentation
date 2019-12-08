# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py

import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import resnet as RN
import utils
import numpy as np
import warnings

from Augs.Channelsplit import ChannelSplit, ChannelSplit2
from Augs.Decalcomanie import Decalcomanie
from Augs.Dropin import Dropin
from Augs.Puzzle import Puzzle

warnings.filterwarnings("ignore")
model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Augmentations Train')
parser.add_argument('--net_type', default='resnet', type=str, help='resnet50, resnet101')
parser.add_argument('--aug_type', default='original', type=str, help='original, channelsplit, channelsplitv2, decalcomanie, dropin, puzzle')
parser.add_argument('--aug_prob', default=0.5, type=float, help='Aug Prob')
parser.add_argument('-r', '--resolution', default='x8', type=str, help='ChannelSplit Resolution[(v1)x1, x8, x64, x512 /(v2)x1, x2, x4, x8]')
parser.add_argument('--skip', default=False, type=bool, help='ChannelSplit Skip')
parser.add_argument('--choice', default=1, type=int, help='Channel Split number of Channels')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N', help='Workers')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='Epochs')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N', help='Batch_size')
parser.add_argument('--lr', '--learning-rate', default=0.25, type=float, metavar='LR', help='Learning Rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Momentum')
parser.add_argument( '--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='Weight Decay')
parser.add_argument('-p', '--print-freq', default=98, type=int, metavar='N', help='Print Frequency')
parser.add_argument('--bottleneck', default=True, type=bool, help='Bottleneck')
parser.add_argument('--dataset', dest='dataset', default='cifar100', type=str, help='cifar10, cifar100')
parser.add_argument('--save_path', default='./models', type=str, help='Save Models')

best_err1 = 100
best_err5 = 100

def main():
    global args, best_err1, best_err5, temp
    args = parser.parse_args()

    #transforms
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    # args.aug_type
    # args.aug_prob
    # args.resolution
    # args.skip
    # args.choice

    transform = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if args.aug_type == 'channelsplit':
        transform.append(ChannelSplit(args.resolution, args.skip, args.prob))
    if args.aug_type == 'channelsplit2':
        transform.append(ChannelSplit2(args.resolution, args.choice, args.skip, args.prob))
    if args.aug_type == 'decalcomanie':
        transform.append(Decalcomanie())
    if args.aug_type == 'dropin':
        transform.append(Dropin(1, 8))
    if args.aug_type == 'puzzle':
        transform.append(Puzzle())
    transform.append([
        transforms.ToTensor(),
        normalize,
    ])

    transform_train = transforms.Compose(transform)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    #data load
    if args.dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        numberofclass = 100
    elif args.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        numberofclass = 10


    print("=> creating model '{}'".format(args.net_type))
    if args.net_type == 'resnet50':
        model = RN.ResNet(args.dataset, 50, numberofclass, args.bottleneck).cuda()
    elif args.net_type == 'resnet101':
        model = RN.ResNet(args.dataset, 101, numberofclass, args.bottleneck).cuda()
    #model = torch.nn.DataParallel(model).cuda()

    print(model)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # loss function, optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    cudnn.benchmark = True
    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        err1, err5, val_loss = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best:
            best_err5 = err5

        print('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.net_type,
            'state_dict': model.state_dict(),
            'best_err1': best_err1,
            'best_err5': best_err5,
            'optimizer': optimizer.state_dict(),
        }, is_best)

    # print('Best accuracy (top-1 and 5 error):', best_err1, best_err5)
    # o = open('logging.txt', 'a')
    # o.write('classes {} Best accuracy (top-1 and 5 error): {} {} \n'.format(clasess, best_err1, best_err5))
    # o.close()


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input, requires_grad=True)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    return losses.avg


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                   epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = args.save_path
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + '/' + filename
    torch.save(state, filename)
    if is_best:
        filename_ = directory + '/[best]' + filename
        shutil.copyfile(filename, filename_)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
    # lists = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300]
    # for i in lists:
    #     main(i)
