

from torch.utils.data import RandomSampler
from randaugment import RandAugmentMC
from apex import amp
from lr_scheduler import get_scheduler
from models.preact_resnet import *
from other_utils import *
from MemoryMoCo import MemoryMoCo
from queue_with_pro import *
from test_eval import test_eval
from utils_noise_v2 import *
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import time
from dataset.cifar_dataset import *
import torch.utils.data as data
from torch import optim
from torchvision import datasets, transforms, models
from sample_selection import *
import random
import sys

sys.path.append('../utils')


def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')


    parser.add_argument('--lr', '--base-learning-rate', '--base-lr',type=float, default=0.15, help='learning rate')
    parser.add_argument('--lr-scheduler', type=str, default='cosine',choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--lr-warmup-epoch', type=int,default=5, help='warmup epoch')
    parser.add_argument('--lr-warmup-multiplier', type=int,default=100, help='warmup multiplier')
    parser.add_argument('--lr-decay-epochs', type=int, default=[80, 100, 130, 170, 220], nargs='+',help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr-decay-rate', type=float, default=0.5,help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--initial_epoch', type=int, default=1,help="Star training at initial_epoch")

    parser.add_argument('--batch_size', type=int, default=256,help='#images in each mini-batch')
    parser.add_argument('--test_batch_size', type=int,default=100, help='#images in each mini-batch')

    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

    parser.add_argument('--train_root', default='./dataset',help='root for train data')
    parser.add_argument('--out', type=str, default='./out',help='Directory of the output')
    parser.add_argument('--experiment_name', type=str, default='Proof',help='name of the experiment (for the output files)')

    parser.add_argument('--download', type=bool,default=True, help='download dataset')

    parser.add_argument('--network', type=str,default='PR18', help='Network architecture')
    parser.add_argument('--low_dim', type=int, default=128,help='Size of contrastive learning embedding')
    parser.add_argument('--headType', type=str,default="Linear", help='Linear, NonLinear')
    parser.add_argument('--seed_initialization', type=int,default=1, help='random seed (default: 1)')
    parser.add_argument('--seed_dataset', type=int,default=42, help='random seed (default: 1)')

    parser.add_argument('--alpha_m', type=float, default=1.0,help='Beta distribution parameter for mixup')
    parser.add_argument('--alpha_moving', type=float,default=0.999, help='exponential moving average weight')

    parser.add_argument('--uns_queue_k', type=int,default=10000, help='uns-cl num negative sampler')
    parser.add_argument('--uns_t', type=float, default=0.1,help='uns-cl temperature')
    parser.add_argument('--sup_t', default=0.1, type=float,help='sup-cl temperature')

    parser.add_argument('--sup_queue_begin', type=int, default=3,help='Epoch to begin using queue for sup-cl')
    parser.add_argument('--queue_per_class', type=int, default=100,help='Num of samples per class to store in the queue. queue size = queue_per_class*num_classes*2')
    parser.add_argument('--aprox', type=int, default=1,help='Approximation for numerical stability taken from supervised contrastive learning')
    parser.add_argument('--k_val', type=int, default=250,help='k for k-nn correction')
    parser.add_argument('--k_val2', type=float,default=0.05)

    parser.add_argument('--DA_len', default=32)

    parser.add_argument('--epoch', type=int, default=300,help='training epoches')
    
    parser.add_argument('--dataset', type=str,default='CIFAR-100', help='CIFAR-10, CIFAR-100')
    parser.add_argument('--num_classes', type=int, default=100,help='Number of in-distribution classes')

    parser.add_argument('--sup_queue_use', type=int,default=1, help='1: Use queue for sup-cl')
    parser.add_argument('--ema_w', type=float, default=0.7)
    parser.add_argument('--threshold_theta', type=float,default=0.85, help='Number of in-distribution classes')
    parser.add_argument('--pro_w', type=float, default=0.9)

    parser.add_argument('--warmup-epoch', type=int,default=1, help='warmup epoch')
    parser.add_argument('--alpha', type=float, default=0.75,help='example selection th')
    parser.add_argument('--aprow', type=float, default=3,help='example selection th')

    parser.add_argument('--tt', type=float, default=0.15)
    parser.add_argument('--st', type=float, default=0.15)
    
    parser.add_argument('--lambda_l', type=float, default=1,help='example selection th')
    parser.add_argument('--lambda_i', type=float,default=0.5, help='weight of loss_s')
    parser.add_argument('--lambda_c', type=float,default=0.5, help='weight of loss_s')

    parser.add_argument('--cuda_dev', type=int, default=7, help='GPU to select')
    parser.add_argument('--noise_ratio', type=float, default=0.9, help='percent of noise')
    parser.add_argument('--noise_type', default='symmetric',help='symmetric or asymmetric')
    
    args = parser.parse_args()
    return args

def data_config(args, transform_train, transform_test):
    trainset, testset = get_dataset(args, transform_train, transform_test)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print('############# Data loaded #############')

    return train_loader, test_loader, trainset


def build_model(args, device):
    model = PreActResNet18(num_classes=args.num_classes,
                           low_dim=args.low_dim, head=args.headType).to(device)
    model_ema = PreActResNet18(
        num_classes=args.num_classes, low_dim=args.low_dim, head=args.headType).to(device)
    # copy weights from `model' to `model_ema'
    moment_update(model, model_ema, 0)
    return model, model_ema


def main(args):

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    best_acc1 = 0
    best_acc_ema = 0
    args.best_acc = 0


    time_now = time.time()
    exp_path = os.path.join(args.out, 'SI{}_SD{} --{}'.format(args.seed_initialization,
                            args.seed_dataset, time_now), args.noise_type, str(args.noise_ratio))
    res_path = os.path.join(args.out, 'SI{}_SD{} --{}'.format(args.seed_initialization,
                            args.seed_dataset, time_now), args.noise_type, str(args.noise_ratio))

    if not os.path.isdir(res_path):
        os.makedirs(res_path)

    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)

    __console__ = sys.stdout
    name = "/results"
    log_file = open(res_path+name+".log", 'a')
    sys.stdout = log_file
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_dev)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed_initialization)  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed(args.seed_initialization)
        torch.cuda.manual_seed_all(args.seed_initialization)
        # GPU seed
    os.environ['PYTHONHASHSEED'] = str(args.seed_initialization)
    # python seed for image transformation
    random.seed(args.seed_initialization)
    np.random.seed(args.seed_initialization)

    if args.dataset == 'CIFAR-10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif args.dataset == 'CIFAR-100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]

    transform_train_complex = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_train_strong = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=4, padding_mode='reflect'),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_train = Transform_list(transform_train_complex, transform_train_strong)

    train_loader, test_loader, trainset = data_config(args, transform_train, transform_test)

    model, model_ema = build_model(args, device)
    uns_contrast = MemoryMoCo(args.low_dim, args.uns_queue_k, args.uns_t, thresh=0).cuda()

    prototype = dce_loss(args.num_classes, args.low_dim).cuda()
    Da = DA(args.num_classes)

    optimizer_s = optim.SGD(prototype.parameters(),lr=0.01, momentum=0.9, weight_decay=1e-4)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", num_losses=2)
    scheduler = get_scheduler(optimizer, len(train_loader), args)

    if args.sup_queue_use == 1:
        queue = queue_with_pro(args, device)
    else:
        queue = []

    
    noisy_dataloader = torch.utils.data.DataLoader(
        trainset,
        sampler=RandomSampler(trainset, replacement=True, num_samples=50000),
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
        # drop_last=True
    )
    Ema_feature = torch.randn((args.low_dim, len(trainset.targets)), requires_grad=False).cuda()
    Ema_feature = nn.functional.normalize(Ema_feature, dim=0)
    
    Corr_sample1 = torch.ones(50000).cuda() * (-1)
    Corr_sample2 = torch.ones(50000).cuda() * (-1)
    Corr_arr1 = torch.ones(3, 50000).cuda() * (-1)
    Corr_arr2 = torch.ones(3, 50000).cuda() * (-1)
    sample_weight = torch.ones(50000).cuda()

    ssl_ptr = torch.tensor(0).cuda()

    noisy_labels = torch.tensor(train_loader.dataset.targets).cuda()

    for epoch in range(args.initial_epoch, args.epoch + 1):

        st = time.time()
        print("=================>    ", args.experiment_name, args.noise_ratio)
        if (epoch <= args.warmup_epoch):
            
            train_sup(noisy_labels, sample_weight, prototype, optimizer_s, args, scheduler, model,
                          model_ema, uns_contrast, queue, device, train_loader, train_loader, optimizer, epoch)

        else:
            train_selected_loader = torch.utils.data.DataLoader(
                trainset, batch_size=args.batch_size, num_workers=4, pin_memory=True, sampler=torch.utils.data.WeightedRandomSampler(iter_weight, len(iter_weight)))

            soft_labels2 = train_sel(noisy_labels, Da, corr_sample, clean_sample, optimizer_s, anchor_sample, prototype, sample_weight, 
                                                        noisy_dataloader, Ema_feature, args, scheduler, model, model_ema, uns_contrast, queue, device, train_loader, train_selected_loader, optimizer, epoch)

        features = compute_features(args, model, train_loader, test_loader)

        with torch.no_grad():
            if epoch < 3:
                Ema_feature = feature_update(Ema_feature, model_ema, train_loader, 0)
            else:
                Ema_feature = feature_update(Ema_feature, model_ema, train_loader, args.ema_w)

        if (epoch >= args.warmup_epoch):
            ##Collaborative Selection
            soft_labels1 = get_soft_labels1(args, features, train_loader)
            clean_sample1, c_weight1 = csample_selection(args, train_loader, soft_labels1)
            sample_weight, Corr_sample1 = nsample_selection(soft_labels1, args.threshold_theta*(0.999**epoch), Corr_arr1, ssl_ptr, Corr_sample1)

            anchor_sample = torch.ones(clean_sample1.size()).cuda() * (-1)
            clean_sample = torch.ones(clean_sample1.size()).cuda() * (-1)
            corr_sample = torch.ones(clean_sample1.size()).cuda() * (-1)

            anchor_sample[Corr_sample1 >= 0] = Corr_sample1[Corr_sample1 >= 0]
            anchor_sample[clean_sample1 >=0] = clean_sample1[clean_sample1 >= 0]

            clean_sample[clean_sample1 >=0] = clean_sample1[clean_sample1 >= 0]

            if epoch > args.warmup_epoch:
                clean_sample2, c_weight2 = csample_selection(args, train_loader, soft_labels2)
                sample_weight, Corr_sample2 = nsample_selection(soft_labels2, args.threshold_theta*(0.999**epoch), Corr_arr2, ssl_ptr, Corr_sample2)
                
                clean_idx = torch.eq(clean_sample1, clean_sample2) & (clean_sample1 != -1)
                clean_sample = torch.ones(clean_sample1.size()).cuda() * (-1)
                clean_sample[clean_idx] = clean_sample2[clean_idx]

                corr_sample[Corr_sample2 >=0] = Corr_sample2[Corr_sample2 >= 0]
                corr_sample[clean_sample >= 0] = -1

            iter_weight = torch.zeros(clean_sample.size()).cuda()
            iter_weight[clean_sample >= 0] = 1

            ssl_ptr = ssl_ptr + 1
            if ssl_ptr == 3:
                ssl_ptr = 0

            Corr_arr1[ssl_ptr] = -1
            Corr_arr2[ssl_ptr] = -1

        print('######## Test ########')
        loss, acc = test_eval(args, model, device, test_loader)
        loss_ema, acc_ema = test_eval(args, model_ema, device, test_loader)

        if acc > best_acc1:
            best_acc1 = acc
        if acc_ema > best_acc_ema:
            best_acc_ema = acc_ema

        print("best {}  best_ema {}".format(best_acc1, best_acc_ema))

        print('Epoch time: {:.2f} seconds\n'.format(time.time()-st))

        log_file.flush()

        if (epoch == (args.epoch - 1)):
            save_model(model, optimizer, args, epoch, exp_path + '/' + '-' + str(
                args.num_classes) + '-' + str(args.noise_ratio) + '-' + " Sel-CL_model.pth")
            save_model(model_ema, optimizer, args, epoch, exp_path + '/' + '-' + str(
                args.num_classes) + '-' + str(args.noise_ratio) + '-' + " Sel-CL_model_ema.pth")
         

if __name__ == "__main__":
    args = parse_args()
    main(args)
