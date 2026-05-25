import os
import numpy as np
import time, datetime
import argparse
import copy
# from thop import profile, clever_format
from collections import OrderedDict
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from models.quant_function import ReScaWConv

from utils import data_loaders
from models.quant_vgg import vgg_16_bn
from models.quant_resnet_cifar import resnet_20
import utils.common as utils
from utils.functions import split_weights


parser = argparse.ArgumentParser("cifar100 prun training")

parser.add_argument(
    '--arch',
    type=str,
    # default='resnet_20',
    default='vgg_16_bn',
    help='architecture')

parser.add_argument(
    '--job_dir',
    type=str,
    default='./log/prun/cifar100/res20-4bit',
    # default='./log/prun/cifar100/vgg16-4bit',
    help='path for saving trained models')

parser.add_argument(
    '--batch_size',
    type=int,
    default=512,
    help='batch size')

parser.add_argument(
    '--epochs',
    type=int,
    default=300,
    help='num of training epochs')

parser.add_argument(
    '--lr',
    type=float,
    default=1e-3,
    help='init learning rate')

parser.add_argument(
    '--resume',
    action='store_true',
    help='whether continue training from the same directory')

parser.add_argument(
    '--use_pretrain',
    action='store_true',
    default=True,
    help='whether use pretrain model')

parser.add_argument(
    '-bit',
    default=4,
    type=int,
    metavar='N',
    help='bitwidth of weight')

parser.add_argument(
    '--pretrain_dir',
    type=str,
    default='./log/quant/cifar100/res20-4bit/model_best.pth.tar',
    # default='./log/quant/cifar100/vgg16-4bit/model_best.pth.tar',
    help='pretrain model path')

parser.add_argument(
    '--rank_conv_prefix',
    type=str,
    default='./rank_conv/vgg_16_bn_limit6',
    help='rank conv file folder')

parser.add_argument(
    '--compress_rate',
    type=str,
    default='[0.1]+[0.35]*3+[0.75]*16',
    # default='[0.45]*7+[0.7]*5',
    help='compress rate of each conv')

parser.add_argument(
    '--test_only',
    action='store_true',
    help='whether it is test mode')

parser.add_argument(
    '--test_model_dir',
    type=str,
    help='test model path')

parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')

parser.add_argument('--dataset',
                    default='CIFAR100',
                    type=str,
                    help='dataset name',
                    choices=['CIFAR10', 'CIFAR100', 'ImageNet', 'TinyImageNet'])

args = parser.parse_args()
print_freq = (256*50)//args.batch_size

if not os.path.isdir(args.job_dir):
    os.makedirs(args.job_dir)

utils.record_config(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
logger = utils.get_logger(os.path.join(args.job_dir, 'logger'+now+'.log'))

#use for loading pretrain model
if len(args.gpu)>1:
    name_base='module.'
else:
    name_base=''

def load_vgg_model(model, oristate_dict):
    state_dict = model.state_dict()
    last_select_index = None #Conv index selected in the previous layer

    cnt=0
    prefix = args.rank_conv_prefix+'/rank_conv'
    subfix = ".npy"
    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d) or isinstance(module, ReScaWConv):

            cnt+=1
            oriweight = oristate_dict[name + '.weight']
            curweight =state_dict[name_base+name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num:

                cov_id = cnt
                if cov_id != 13:
                    logger.info('loading rank from: ' + prefix + str(cov_id) + subfix)
                    rank = np.load(prefix + str(cov_id) + subfix)
                    select_index = np.argsort(rank)[orifilter_num-currentfilter_num:]  # preserved filter id
                    select_index.sort()

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+name + '.weight'][index_i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                           state_dict[name_base+name + '.weight'][index_i] = \
                                oristate_dict[name + '.weight'][i]

                    last_select_index = select_index

            elif last_select_index is not None:
                for i in range(orifilter_num):
                    for index_j, j in enumerate(last_select_index):
                        state_dict[name_base+name + '.weight'][i][index_j] = \
                            oristate_dict[name + '.weight'][i][j]
            else:
                state_dict[name_base+name + '.weight'] = oriweight
                last_select_index = None

    model.load_state_dict(state_dict)
    print('!!!!!!!!!!!!!!!!!!!load original model successfully!!!!!!!!!!!!!!!!!')

def load_resnet_model(model, oristate_dict, layer):
    cfg = {
        20: [3, 3, 3],
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.rank_conv_prefix+'/rank_conv'
    subfix = ".npy"

    cnt=1
    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d) or isinstance(module, ReScaWConv):
            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name_base + name + '.weight']
            all_conv_weight.append(name_base + name + '.weight')
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            cov_id = cnt
            logger.info('loading rank from: ' + prefix + str(cov_id) + subfix)
            rank = np.load(prefix + str(cov_id) + subfix)
            select_index = np.argsort(rank)[orifilter_num - currentfilter_num:]  # preserved filter id
            select_index.sort()

            if last_select_index == None:
                for index_i, i in enumerate(select_index):
                    state_dict[name_base + name + '.weight'][index_i] = \
                        oristate_dict[name + '.weight'][i]
            last_select_index = select_index
            break

    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            for l in range(2):

                cnt+=1
                cov_id=cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'
                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight =state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:
                    logger.info('loading rank from: ' + prefix + str(cov_id) + subfix)
                    rank = np.load(prefix + str(cov_id) + subfix)
                    select_index = np.argsort(rank)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d) or isinstance(module,ReScaWConv):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)
    print('!!!!!!!!!!!!!!!!!!!load original model successfully!!!!!!!!!!!!!!!!!')

def main():

    cudnn.benchmark = True
    cudnn.enabled=True
    logger.info("args = %s", args)

    if args.compress_rate:
        import re
        cprate_str = args.compress_rate
        cprate_str_list = cprate_str.split('+')
        pat_cprate = re.compile(r'\d+\.\d*')
        pat_num = re.compile(r'\*\d+')
        cprate = []
        for x in cprate_str_list:
            num = 1
            find_num = re.findall(pat_num, x)
            if find_num:
                assert len(find_num) == 1
                num = int(find_num[0].replace('*', ''))
            find_cprate = re.findall(pat_cprate, x)
            assert len(find_cprate) == 1
            cprate += [float(find_cprate[0])] * num

        compress_rate = cprate

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
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
        val_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # load model
    logger.info('compress_rate:' + str(compress_rate))
    logger.info('==> Building model..')
    model = eval(args.arch)(compress_rate=compress_rate,num_bits=args.bit,num_classes=CLASSES)
    model.to(device)
    logger.info(model)

    # calculate model size
    # input_image_size=32
    # input_image = torch.randn(1, 3, input_image_size, input_image_size).to(device)
    # flops, params = profile(model, inputs=(input_image,))
    # flops, params = clever_format([flops, params], "%.3f")
    # logger.info('Params: %s' % (params))
    # logger.info('Flops: %s' % (flops))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    if args.test_only:
        if os.path.isfile(args.test_model_dir):
            logger.info('loading checkpoint {} ..........'.format(args.test_model_dir))
            checkpoint = torch.load(args.test_model_dir)
            model.load_state_dict(checkpoint['state_dict'])
            valid_obj, valid_top1_acc, valid_top5_acc = validate(0, val_loader, model, criterion, args)
        else:
            logger.info('please specify a checkpoint file')
        return

    if len(args.gpu) > 1:
        device_id = []
        for i in range((len(args.gpu) + 1) // 2):
            device_id.append(i)
        model = nn.DataParallel(model, device_ids=device_id).cuda()
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
    else:
        if args.use_pretrain:
            logger.info('resuming from pretrain model')
            origin_model = eval(args.arch)(compress_rate=[0.] * 100,num_bits=args.bit,num_classes=CLASSES).cuda()
            ckpt = torch.load(args.pretrain_dir, map_location='cuda:0')

            #if args.arch=='resnet_56':
            #    origin_model.load_state_dict(ckpt['state_dict'],strict=False)
            if args.arch == 'densenet_40' or args.arch == 'resnet_110':
                new_state_dict = OrderedDict()
                for k, v in ckpt['state_dict'].items():
                    new_state_dict[k.replace('module.', '')] = v
                origin_model.load_state_dict(new_state_dict)
            else:
                origin_model.load_state_dict(ckpt['state_dict'])

            oristate_dict = origin_model.state_dict()

            if args.arch == 'vgg_16_bn':
                load_vgg_model(model, oristate_dict)
            elif args.arch == 'resnet_20':
                load_resnet_model(model, oristate_dict, 20)
            elif args.arch == 'resnet_110':
                load_resnet_model(model, oristate_dict, 110)
            else:
                raise
        else:
            logger.info('training from scratch')

    # adjust the learning rate according to the checkpoint
    for epoch in range(start_epoch):
        scheduler.step()

    # train the model
    epoch = start_epoch
    while epoch < args.epochs:
        train_obj, train_top1_acc = train(epoch,  train_loader, model, criterion, optimizer, scheduler)
        valid_obj, valid_top1_acc = validate(epoch, val_loader, model, criterion, args)

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_top1_acc': best_top1_acc,
            'optimizer' : optimizer.state_dict(),
            }, is_best, args.job_dir)

        epoch += 1
        logger.info("=>Best accuracy {:.3f}".format(best_top1_acc))#


def train(epoch, train_loader, model, criterion, optimizer, scheduler):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')

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
        logits = model(images)  # BTL
        out = logits.mean(1)
        loss = criterion(out, target)
        # loss = TET_loss(logits, target, criterion, 1.0, 0)

        # measure accuracy and record loss
        prec1 = utils.accuracy(out, target, topk=(1,))[0]
        n = images.size(0)
        losses.update(loss.item(), n)   #accumulated loss
        top1.update(prec1.item(), n)
        # top5.update(prec5.item(), n)

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
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')

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
            loss = criterion(out, target)
            # loss = TET_loss(logits, target, criterion, 1.0, 0)

            # measure accuracy and record loss
            pred1 = utils.accuracy(out, target, topk=(1, ))[0]
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        logger.info(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return losses.avg, top1.avg


if __name__ == '__main__':
  main()
