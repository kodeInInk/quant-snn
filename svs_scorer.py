import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,5'

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils import data_loaders
import utils.common as utils
from utils.functions import TET_loss
from models.quant_vgg import vgg_16_bn, vggsnn
from models.quant_resnet_cifar import resnet_20

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Rank extraction')

parser.add_argument(
    '--dataset',
    default='CIFAR100',
    type=str,
    help='dataset name',
    choices=['CIFAR10', 'CIFAR100', 'ImageNet', 'TinyImageNet'])

parser.add_argument(
    '--arch',
    type=str,
    default='vgg_16_bn',
    choices=('resnet_20','vgg_16_bn','resnet_56','resnet_110'),
    help='The architecture to prune')

parser.add_argument(
    '--pretrain_dir',
    type=str,
    default='./log/quant/cifar100/vgg16-4bit/model_best.pth.tar',
    help='load the model from the specified checkpoint')

parser.add_argument(
    '--bit',
    type=int,
    default='4',
    help='Select gpu to use')

parser.add_argument(
    '--limit',
    type=int,
    default=6,
    help='The num of batch to get rank.')

parser.add_argument(
    '--batch_size',
    type=int,
    default=128,
    help='Batch size for training.')

parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
print('==> Preparing data..')
if args.dataset == 'CIFAR10':
    trainset, testset = data_loaders.build_cifar(cutout=True, use_cifar10=True, download=False)
    CLASSES = 10
elif args.dataset == 'CIFAR100':
    trainset, testset = data_loaders.build_cifar(cutout=True, use_cifar10=False, download=False)
    CLASSES = 100
elif args.dataset == 'ImageNet':
    trainset, testset = data_loaders.build_imagenet()
    CLASSES = 1000
elif args.dataset == 'DVSCIFAR10':
    trainset, testset = data_loaders.build_dvscifar10()
    CLASSES = 10
elif args.dataset == 'TinyImageNet':
    trainset, testset = data_loaders.build_tiny_imagenet()
    CLASSES = 200
elif args.dataset == 'DVS128':
    trainset, testset = data_loaders.build_dvs128(T=args.time)
    CLASSES = 11
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
val_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)


# Model
print('==> Building model..')
net = eval(args.arch)(compress_rate=[0.0]*100, num_bits=args.bit, num_classes=CLASSES)
net = net.cuda()
print(net)

if len(args.gpu)>1 and torch.cuda.is_available():
    device_id = []
    for i in range((len(args.gpu) + 1) // 2):
        device_id.append(i)
    net = torch.nn.DataParallel(net, device_ids=device_id)

if args.pretrain_dir:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.pretrain_dir, map_location='cuda:'+args.gpu)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        new_state_dict[k.replace('ops.', 'module.',1)] = v
    net.load_state_dict(new_state_dict)          
else:
    print('please speicify a pretrain model ')
    raise NotImplementedError

criterion = nn.CrossEntropyLoss()
feature_result = torch.tensor(0.)
total = torch.tensor(0.)
fpath = './rank/resnet20/'
batch_num = 0
layer_num = 0


def get_feature_hook(self, input, output):

    global feature_result
    global entropy
    global total
    global batch_num
    global layer_num
    output = output.mean(1)
    a = output.shape[0]
    b = output.shape[1]
    c = torch.tensor([torch.linalg.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b)])

    c = c.view(a, -1).float()
    c = c.sum(0)
    # torch.save(c/a,fpath+'batch_'+str(batch_num))
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total
    # torch.save(feature_result, fpath+f'accum_ResNet20_conv{layer_num}_'+str(batch_num))
    batch_num = (batch_num + 1) % args.limit


def inference():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    limit = args.limit

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            #use the first 5 batches to estimate the rank.
            if batch_idx >= limit:
               break

            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            loss = TET_loss(outputs, targets, criterion, 1.0, 0.001)

            test_loss += loss.item()
            _, predicted = outputs.mean(1).max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            utils.progress_bar(batch_idx, limit, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))#'''

if args.arch=='vgg_16_bn' or args.arch=='vggsnn':

    if len(args.gpu) > 1:
        relucfg = net.module.relucfg
    else:
        relucfg = net.relucfg

    for i, cov_id in enumerate(relucfg):
        cov_layer = net.features[cov_id]
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        layer_num += 1
        handler.remove()

        if not os.path.isdir('rank_conv/'+args.arch+'_limit%d'%(args.limit)):
            os.mkdir('rank_conv/'+args.arch+'_limit%d'%(args.limit))
        np.save('rank_conv/'+args.arch+'_limit%d'%(args.limit)+'/rank_conv' + str(i + 1) + '.npy',
                feature_result.numpy())

        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

elif args.arch=='resnet_20':

    cov_layer = eval('net.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    layer_num += 1
    handler.remove()

    if not os.path.isdir('rank_conv/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir('rank_conv/' + args.arch+'_limit%d'%(args.limit))
    np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit)+ '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    # ResNet56 per block
    cnt=1
    for i in range(3):
        block = eval('net.layer%d' % (i + 1))
        for j in range(3):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            layer_num += 1
            handler.remove()
            np.save('rank_conv/' + args.arch +'_limit%d'%(args.limit)+ '/rank_conv%d'%(cnt + 1)+'.npy', feature_result.numpy())
            cnt+=1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            layer_num += 1
            handler.remove()
            np.save('rank_conv/' + args.arch +'_limit%d'%(args.limit)+ '/rank_conv%d'%(cnt + 1)+'.npy',
                    feature_result.numpy())
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

