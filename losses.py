import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import numpy as np
import scipy
import torch
from other_utils import *
import torch.nn as nn
import torch.nn.functional as F
from utils_mixup_v2 import *

criterionCE = torch.nn.CrossEntropyLoss(reduction="none")

eps = 1e-12

def CalPairwise(dist):
    dist[dist < 0] = 0
    Pij = torch.exp(-dist)
    return Pij

def Distance_squared(x, y, featdim=1):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    d = dist.clamp(min=eps)
    d[torch.eye(d.shape[0]) == 1] = eps
    return d

def Loc_loss(pred_q,feat_k,bsz,args):
    pred_ntd = nn.functional.normalize(pred_q, dim=1)
    pred_sim = Distance_squared(pred_ntd, pred_ntd)
    feat_sim = Distance_squared(feat_k, feat_k)
    feat_sim = CalPairwise(feat_sim)
    largest_per_row = torch.topk(feat_sim, k=int(bsz * args.k_val2), dim=1).values[:, -1].contiguous().view(-1,1)
    feat_sim = torch.where(feat_sim < largest_per_row, torch.tensor(0.0).cuda(), feat_sim)
    loss_loc = torch.mean(feat_sim.detach() * pred_sim)
    return loss_loc


def Corr_loss(args, Corr_sample, index_u, ims_u_strong, ims_u_strong2, model, device):
    mask_all = Corr_sample[index_u].clone()
    mask_all[mask_all >= 0] = 1
    mask_all[mask_all < 0] = 0
    label_u = Corr_sample[index_u].long()
    label_u[mask_all == 0] = 0
    img1, y_a1, y_b1, mix_index1, lam1 = mix_data_lab(ims_u_strong, label_u, args.alpha_m, device)
    img2, y_a2, y_b2, mix_index2, lam2 = mix_data_lab(ims_u_strong2, label_u, args.alpha_m, device)

    pred_us1, embedA = model(img1)
    pred_us2, embedB = model(img2)

    preds = torch.cat((pred_us1, pred_us2), dim=0)
    mix_index = torch.cat((mix_index1, mix_index2), dim=0)

    targets_1 = torch.cat((y_a1, y_a2), dim=0)
    targets_2 = torch.cat((y_b1, y_b2), dim=0)
    ones_vec = torch.ones((pred_us1.size()[0],)).float().to(device)
    lam_vec = torch.cat((lam1 * ones_vec, lam2 * ones_vec), dim=0).to(device)

    mask_all2 = torch.cat((mask_all, mask_all), dim=0)
    loss = lam_vec * criterionCE(preds, targets_1) * mask_all2 + (1 - lam_vec) * criterionCE(preds, targets_2) * mask_all[mix_index]
    loss_u = loss.mean()
    return loss_u


def Ins_loss(embed_uw, Ema_feature, index_x, prob_uw, confident_sample, embed_us, args):
    bu = embed_uw.shape[0]
    with torch.no_grad():
        teacher_logits = embed_uw @ Ema_feature[:, index_x]
        teacher_prob_ori = F.softmax(teacher_logits / args.st, dim=1)  
        factor = prob_uw.gather(1, confident_sample[index_x].long().expand([bu, -1]))
        teacher_prob = teacher_prob_ori * factor
        teacher_prob /= torch.sum(teacher_prob, dim=1, keepdim=True)
   
    student_logits = embed_us @ Ema_feature[:, index_x]
    student_prob = F.softmax(student_logits / args.st, dim=1)
    loss_in = torch.sum(-teacher_prob.detach() * torch.log(student_prob), dim=1)
    loss_in = loss_in.mean()
    return loss_in


def Supervised_ContrastiveLearning_loss(args, pairwise_comp, maskSup, mask2Sup, maskUnsup, mask2Unsup, logits_mask, lam1, lam2, bsz):

    logits = torch.div(pairwise_comp, args.sup_t)

    exp_logits = torch.exp(logits) * logits_mask  # remove diagonal

    if args.aprox == 1:
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  ## Approximation for numerical stability taken from supervised contrastive learning
    else:
        log_prob = torch.log(torch.exp(logits) + 1e-7) - torch.log(exp_logits.sum(1, keepdim=True) + 1e-7)

    exp_logits2 = torch.exp(logits) * logits_mask  # remove diagonal

    if args.aprox == 1:
        log_prob2 = logits - torch.log(exp_logits2.sum(1, keepdim=True))  ## Approximation for numerical stability taken from supervised contrastive learning
    else:
        log_prob2 = torch.log(torch.exp(logits) + 1e-7) - torch.log(exp_logits2.sum(1, keepdim=True) + 1e-7)

    # compute mean of log-likelihood over positive (weight individual loss terms with mixing coefficients)
    mean_log_prob_pos_sup = (maskSup * log_prob).sum(1) / ((maskSup>0).sum(1) + maskUnsup.sum(1))

    mean_log_prob_pos_unsup = (maskUnsup * log_prob).sum(1) / ((maskSup>0).sum(1) + maskUnsup.sum(1))
    ## Second mixup term log-probs
    mean_log_prob_pos2_sup = (mask2Sup * log_prob2).sum(1) / ((mask2Sup>0).sum(1) + mask2Unsup.sum(1))
    mean_log_prob_pos2_unsup = (mask2Unsup * log_prob2).sum(1) / ((mask2Sup>0).sum(1) + mask2Unsup.sum(1))

    ## Weight first and second mixup term (both data views) with the corresponding mixing weight

    ##First mixup term. First mini-batch part. Unsupervised + supervised loss separated
    loss1a = -lam1 * mean_log_prob_pos_unsup[:int(len(mean_log_prob_pos_unsup) / 2)] - lam1 * mean_log_prob_pos_sup[:int(len(mean_log_prob_pos_sup) / 2)]
    ##First mixup term. Second mini-batch part. Unsupervised + supervised loss separated
    loss1b = -lam2 * mean_log_prob_pos_unsup[int(len(mean_log_prob_pos_unsup) / 2):] - lam2 * mean_log_prob_pos_sup[int(len(mean_log_prob_pos_sup) / 2):]
    ## All losses for first mixup term
    loss1 = torch.cat((loss1a, loss1b))

    ##Second mixup term. First mini-batch part. Unsupervised + supervised loss separated
    loss2a = -(1.0 - lam1) * mean_log_prob_pos2_unsup[:int(len(mean_log_prob_pos2_unsup) / 2)] - (1.0 - lam1) * mean_log_prob_pos2_sup[:int(len(mean_log_prob_pos2_sup) / 2)]
    ##Second mixup term. Second mini-batch part. Unsupervised + supervised loss separated
    loss2b = -(1.0 - lam2) * mean_log_prob_pos2_unsup[int(len(mean_log_prob_pos2_unsup) / 2):] - (1.0 - lam2) * mean_log_prob_pos2_sup[int(len(mean_log_prob_pos2_sup) / 2):]
    ## All losses secondfor first mixup term
    loss2 = torch.cat((loss2a, loss2b))

    ## Final loss (summation of mixup terms after weighting)
    loss = loss1 + loss2

    loss = loss.view(2, bsz).mean(dim=0)
    
    loss = ((maskSup[:bsz].sum(1))>0)*(loss.view(bsz))
    return loss.mean()


def ClassificationLoss(args, predsA, predsB, y_a1, y_b1, y_a2, y_b2, mix_index1, mix_index2, lam1, lam2, criterionCE, epoch, device):

    preds = torch.cat((predsA, predsB), dim=0)

    targets_1 = torch.cat((y_a1, y_a2), dim=0)
    targets_2 = torch.cat((y_b1, y_b2), dim=0)
    mix_index = torch.cat((mix_index1, mix_index2), dim=0)

    ones_vec = torch.ones((predsA.size()[0],)).float().to(device)
    lam_vec = torch.cat((lam1 * ones_vec, lam2 * ones_vec), dim=0).to(device)

    loss = lam_vec * criterionCE(preds, targets_1) + (1 - lam_vec) * criterionCE(preds, targets_2)
    loss = loss.mean()
    return loss

