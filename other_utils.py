from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



@torch.no_grad()
def feature_update(bank,model,train_loader,ema_w):
    for batch_idx, (img, labels, index) in enumerate(train_loader):
        img = img[2].cuda()
        #img = img[0].cuda()
        _, embed = model(img)
        bank[:, index] =  F.normalize(bank[:, index] * ema_w +  embed.t() * (1-ema_w))      
    return bank


class DA(nn.Module):
    def __init__(self, num_classes):
        super(DA, self).__init__()
        self.DA_len = 32
        self.DA_queue = torch.zeros(self.DA_len, num_classes, dtype=torch.float).cuda()
        self.DA_ptr = torch.zeros(1, dtype=torch.long).cuda()
        
    def forward(self, probs):
        probs_bt_mean = probs.mean(0)
        ptr = int(self.DA_ptr)
        self.DA_queue[ptr] = probs_bt_mean
        self.DA_ptr[0] = (ptr + 1) % self.DA_len
        probs = probs / self.DA_queue.mean(0)
        probs = probs / probs.sum(dim=1, keepdim=True)
        probs = probs.detach()
        return probs


def get_pseudo_label(index_u, embed_uw, centers, prob_uw,soft_labels2,args):
    feat_label = F.softmax(embed_uw @ centers / 0.1, dim=1)   # (batch_size, num_classes)
    pseudo_label = prob_uw * args.pro_w + feat_label * (1-args.pro_w)
    soft_labels2[index_u] = pseudo_label
    return soft_labels2


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
        
class TwoTransform:
    """Create two Transform of the same image"""
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return [self.transform1(x), self.transform2(x)]

def tofloat(x):
    return x
        

def set_bn_train(model):
    def set_bn_train_helper(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()

    model.eval()
    model.apply(set_bn_train_helper)


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)

class Transform_list:
    """Create two Transform of the same image"""
    def __init__(self, transform_complex, transform_strong):
        self.transform1 = transform_complex
        self.transform2 = transform_strong

    def __call__(self, x):
        return [self.transform1(x), self.transform1(x), self.transform2(x), self.transform2(x)]
    


def compute_features(args,net,trainloader,testloader):
    net.eval()
    total = 0
    testsize = testloader.dataset.__len__()

    with torch.no_grad():
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)
        for batch_idx, (inputs, _,_) in enumerate(temploader):
            
            batchSize = inputs.size(0)
            inputs = inputs.cuda()

            _,features = net(inputs)
            if batch_idx == 0:
                trainFeatures = features.data
            else:
                trainFeatures = torch.cat((trainFeatures, features.data), 0)
                    
    trainloader.dataset.transform = transform_bak
    
    return trainFeatures
