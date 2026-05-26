import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#usinf python 10 and requirements.txt for non conflicts
#a)run python main1_quant.py --dataset CIFAR10 --job_dir ./log/quant/cifar10/vgg16-4bit --epochs 5 --workers 0
#b) run: python main0_fp.py --dataset CIFAR10 --job_dir ./log/fp/cifar10/vgg16 --workers 0

import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import time, datetime
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from torch.utils.data import DataLoader

from collections import OrderedDict
from utils import data_loaders
from utils import common
from utils.functions import split_weights,TET_loss
# from thop import profile, clever_format

from models.fp_vgg import vgg_16_bn #will be using 70% ish fp for pretrainign
from models.quant_resnet_cifar import resnet_20


parser = argparse.ArgumentParser("cifar100 quant")

parser.add_argument(
    '--arch',
    type=str,
    default='vgg_16_bn',
    help='architecture')

parser.add_argument(
    '--fp',
    action='store_true',
    help='train full precision model (stage 1)')

parser.add_argument(
    '-bit',
    default=4,
    type=int,
    metavar='N',
    help='bitwidth of weight')

parser.add_argument(
    '--job_dir',
    type=str,
    # default='./log/quant/cifar100/vgg16-4bit',
    default='./log/fp/cifar10/vgg16',
    help='path for saving trained models')

parser.add_argument(
    '--batch_size',
    type=int,
    default=256,
    help='batch size')

parser.add_argument(
    '--epochs',
    type=int,
    default=50,
    help='num of training epochs')

parser.add_argument(
    '--lr',
    type=float,
    default=1e-3,   # 1e-3 for vgg16 Adam
    help='init learning rate')

parser.add_argument(
    '--resume',
    action='store_true',
    help='whether continue training from the same directory')

parser.add_argument(
    '--pretrained',
    type=str,
    default=None,
    help='path to FP pretrained checkpoint for QAT')

parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')

parser.add_argument(
    '--dataset',
    default='CIFAR100',
    type=str,
    help='dataset name',
    choices=['CIFAR10', 'CIFAR100', 'ImageNet', 'TinyImageNet'])

parser.add_argument(
    '--fp', 
    action='store_true', 
    help='train full precision model (stage 1)')
parser.add_argument(
    '--pretrained', 
    type=str, 
    default=None, 
    help='path to fp pretrained checkpoint (stage 2)')

parser.add_argument(
    '-j',
    '--workers',
    default=16,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 16)')

args = parser.parse_args()
print_freq = (256*50)//args.batch_size

common.record_config(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
logger = common.get_logger(os.path.join(args.job_dir, 'logger'+now+'.log'))

if not os.path.isdir(args.job_dir):
    os.makedirs(args.job_dir)

# use for loading pretrain model
if len(args.gpu)>1:
    name_base='module.'
else:
    name_base=''

def train(epoch, train_loader, model, criterion, optimizer, scheduler):
    batch_time = common.AverageMeter('Time', ':6.3f')
    data_time = common.AverageMeter('Data', ':6.3f')
    losses = common.AverageMeter('Loss', ':.4e')
    top1 = common.AverageMeter('Acc@1', ':6.2f')

    model.train()
    end = time.time()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    logger.info('learning_rate: ' + str(cur_lr))

    num_iter = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.to(device)
        target = target.to(device)

        # compute outputy
        logits = model(images)
        out = logits.mean(1)
        loss = TET_loss(logits, target, criterion, 1.0, 0.001)

        # measure accuracy and record loss
        prec1 = common.accuracy(out, target, topk=(1,))[0]
        n = images.size(0)
        losses.update(loss.item(), n)   #accumulated loss
        top1.update(prec1.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info(
                'Epoch[{0}]({1}/{2}): Loss {loss.avg:.4f} Prec@1(1) {top1.avg:.2f}'
                .format(epoch, i, num_iter, loss=losses,top1=top1))

    scheduler.step()

    return losses.avg, top1.avg

def validate(epoch, val_loader, model, criterion, args):
    batch_time = common.AverageMeter('Time', ':6.3f')
    losses = common.AverageMeter('Loss', ':.4e')
    top1 = common.AverageMeter('Acc@1', ':6.2f')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            logits = model(images)
            out = logits.mean(1)
            loss = TET_loss(logits, target, criterion, 1.0, 0.001)

            # measure accuracy and record loss
            pred1 = common.accuracy(out, target, topk=(1, ))[0]
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        logger.info(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return losses.avg, top1.avg

def main():
    cudnn.benchmark = True
    cudnn.enabled=True
    logger.info("args = %s", args)

    # load training data
    if args.dataset == 'CIFAR10':
        trainset, testset = data_loaders.build_cifar(cutout=True, use_cifar10=True, download=False)
        CLASSES = 10
    elif args.dataset == 'CIFAR100':
        trainset, testset = data_loaders.build_cifar(cutout=True, use_cifar10=False, download=False)
        CLASSES = 100
    elif args.dataset == 'DVSCIFAR10':
        trainset, testset = data_loaders.build_dvscifar()
        CLASSES = 10
    elif args.dataset == 'DVS128':
        trainset, testset = data_loaders.build_dvs128(T=args.time)
        CLASSES = 11
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # load model
    logger.info('==> Building model..')
    logger.info('=== Bit width===:'+str(args.bit))
    if args.fp:
        from models.fp_vgg import vgg_16_bn as fp_vgg_16_bn
        model = fp_vgg_16_bn(compress_rate=[0.]*100, num_classes=CLASSES, num_bits=args.bit)
    else:
        model = eval(args.arch)(compress_rate=[0.]*100, num_bits=args.bit, num_classes=CLASSES)
    model.to(device)
    logger.info(model)

    # calculate model size
    # input_image_size=32
    # input_image = torch.randn(1, 3, input_image_size, input_image_size).to(device)
    # flops, params = profile(model, inputs=(input_image,))
    # flops, params = clever_format([flops, params], "%.3f")
    # logger.info('Params: %s' % (params))
    # logger.info('Flops: %s' % (flops))

    if len(args.gpu) > 1:
        device_id = []
        for i in range((len(args.gpu) + 1) // 2):
            device_id.append(i)
        model = nn.DataParallel(model, device_ids=device_id).cuda()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    all_parameters = model.parameters()
    weight_parameters = []

    for pname, p in model.named_parameters():
        if p.ndimension() == 4 or 'conv' in pname:
            weight_parameters.append(p)

    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    optimizer = torch.optim.Adam(
        [{'params': other_parameters},
         {'params': weight_parameters, 'weight_decay': 1e-5}], lr=args.lr, )
    # optimizer = torch.optim.SGD(params=split_weights(model), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    start_epoch = 0
    best_top1_acc= 0

    # load the checkpoint if it exists
    if args.resume:
        checkpoint_dir = os.path.join(args.job_dir, 'checkpoint.pth.tar')
        logger.info('loading checkpoint {} ..........'.format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir)
        start_epoch = checkpoint['epoch'] + 1
        best_top1_acc = checkpoint['best_top1_acc']

        # deal with the single-multi GPU problem
        new_state_dict = OrderedDict()
        tmp_ckpt = checkpoint['state_dict']
        if len(args.gpu) > 1:
            for k, v in tmp_ckpt.items():
                new_state_dict['module.' + k.replace('module.', '')] = v
        else:
            for k, v in tmp_ckpt.items():
                new_state_dict[k.replace('module.', '')] = v

        model.load_state_dict(new_state_dict)
        logger.info("loaded checkpoint {} epoch = {}".format(checkpoint_dir, checkpoint['epoch']))

        # adjust the learning rate according to the checkpoint
        for epoch in range(start_epoch):
            scheduler.step()
    else:
        if args.pretrained:
            checkpoint = torch.load(args.pretrained, map_location=device)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logger.info('loaded FP pretrained weights from {}'.format(args.pretrained))
        if args.pretrained is not None:
            logger.info('loading pretrained fp model: {}'.format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location=device)
            tmp_ckpt = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for k, v in tmp_ckpt.items():
                new_state_dict[k.replace('module.', '')] = v
            model.load_state_dict(new_state_dict, strict=False)
            logger.info('pretrained fp model loaded')
        
        logger.info('training from scratch')
        

    # train the model
    epoch = start_epoch
    while epoch < args.epochs:
        train_obj, train_top1_acc = train(epoch,  train_loader, model, criterion, optimizer, scheduler)
        valid_obj, valid_top1_acc = validate(epoch, val_loader, model, criterion, args)

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        common.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_top1_acc': best_top1_acc,
            'optimizer' : optimizer.state_dict(),
            }, is_best, args.job_dir)

        epoch += 1
        logger.info("=>Best accuracy {:.3f}".format(best_top1_acc))#

if __name__ == '__main__':
  main()
