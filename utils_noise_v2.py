from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from AverageMeter import AverageMeter
from NCECriterion import NCESoftmaxLoss
from other_utils import *
from utils_mixup_v2 import *
from losses import *
import time
import warnings
import os, sys
from apex import amp
import faiss

warnings.filterwarnings('ignore')

def train_sel(noisy_labels,Da,Corr_sample,clean_sample,optimizer_s,anchor_sample, prototype, sample_weight,  noisy_dataloader,Ema_feature,args, scheduler,model,model_ema,contrast,queue,device, train_loader, train_selected_loader, optimizer, epoch):
    train_loss_1 = AverageMeter()
    train_loss_2 = AverageMeter()
    train_loss_3 = AverageMeter() 
    train_loss_4 = AverageMeter() 
   
    # switch to train mode
    model.train()
    set_bn_train(model_ema)
    end = time.time()
    counter = 1

    criterionCE = torch.nn.CrossEntropyLoss(reduction="none")
    criterion = NCESoftmaxLoss(reduction="none").cuda()
    train_selected_loader_iter = iter(train_selected_loader)
    noisy_dataloader = iter(noisy_dataloader)
 
    soft_labels2 = torch.zeros(len(sample_weight), args.num_classes).cuda()

    confident_sample = torch.ones(len(sample_weight)).cuda() * (-1)
    confident_sample[clean_sample >=0] = clean_sample[clean_sample >=0]
    confident_sample[Corr_sample >=0] = Corr_sample[Corr_sample >=0]
    confident_idx = (confident_sample>=0).nonzero()[:,0]

    for batch_idx, (img, labels, index) in enumerate(train_loader):

        img1, img2, labels, index = img[0].to(device), img[1].to(device), labels.to(device), index.to(device)
        img_s = img[2].to(device)
        img_s2 = img[3].to(device)
        bsz = img1.shape[0]

        ##compute uns-cl loss
        pred_q,feat_q = model(img1)

        with torch.no_grad():
            _, feat_k= model_ema(img2)
            _, feat_s= model_ema(img_s)

        out = contrast(feat_q, feat_s, feat_k, update=True)
        uns_loss = criterion(out)   

        centers = prototype.centers   
        centers = nn.functional.normalize(centers, dim=0)

        ##Feature-to-semantic regularization  L_loc
        loss_loc = Loc_loss(pred_q,feat_k,bsz,args)
       
        imgs_u, _, index_u = next(noisy_dataloader)
        img_u_weak = imgs_u[0].cuda()
        img_u_strong = imgs_u[2].cuda()
        img_u_strong2 = imgs_u[3].cuda()
    
        pred_us, embed_us = model(img_u_strong)
        pred_uw, embed_uw = model(img_u_weak)
        prob_uw = F.softmax(pred_uw, dim=-1)

        
        with torch.no_grad():
            soft_labels2 = get_pseudo_label(index_u, embed_uw, centers, prob_uw,soft_labels2,args)
            prob_uw = Da(prob_uw)
        
        ##Semantic-to-feature regularization  L_ins
        loss_ins = Ins_loss(embed_uw, Ema_feature, confident_idx, prob_uw, confident_sample, embed_us, args)
        
            
        img1, y_a1, y_b1, mix_index1, lam1 = mix_data_lab(img1, labels, args.alpha_m, device)
        img2, y_a2, y_b2, mix_index2, lam2 = mix_data_lab(img2, labels, args.alpha_m, device)

        predsA, embedA = model(img1)
        predsB, embedB = model(img2)
    
        predsA = F.softmax(predsA,-1)
        predsB = F.softmax(predsB,-1)

        with torch.no_grad():
            predsA_ema, embedA_ema = model_ema(img1)
            predsB_ema, embedB_ema = model_ema(img2)
            predsA_ema = F.softmax(predsA_ema,-1)
            predsB_ema = F.softmax(predsB_ema,-1)

        
        #prototype_update
        anc_idx = (anchor_sample[index]>=0).nonzero()[:, 0]
        index_p =  index[anc_idx]
        centers, x = prototype(Ema_feature[:, index_p].t())
        output = F.log_softmax(args.aprow * x, dim=1)
        loss1 = F.nll_loss(output, anchor_sample[index_p].long())
        loss_pro = loss1
        optimizer_s.zero_grad()
        loss_pro.backward()
        optimizer_s.step()

        if args.sup_queue_use == 1:
            queue.enqueue_dequeue(torch.cat((embedA_ema.detach(), embedB_ema.detach()), dim=0), torch.cat((predsA_ema.detach(), predsB_ema.detach()), dim=0), torch.cat((index.detach().squeeze(), index.detach().squeeze()), dim=0))

        if args.sup_queue_use == 1 and epoch > args.sup_queue_begin:
            queue_feats, queue_pros, queue_index = queue.get()
                
        else:
            queue_feats, queue_pros, queue_index = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
        

        maskUnsup_batch, maskUnsup_mem, mask2Unsup_batch, mask2Unsup_mem = unsupervised_masks_estimation(args, queue, mix_index1, mix_index2, epoch, bsz, device)

        embeds_batch = torch.cat([embedA, embedB], dim=0)

        pairwise_comp_batch = torch.matmul(embeds_batch, embeds_batch.t())
        

        if args.sup_queue_use == 1 and epoch > args.sup_queue_begin:
            embeds_mem = torch.cat([embedA, embedB, queue_feats], dim=0)
        
            pairwise_comp_mem = torch.matmul(embeds_mem[:2 * bsz], embeds_mem[2 * bsz:].t()) ##Compare mini-batch with memory
          
       
        maskSup_batch, maskSup_mem, mask2Sup_batch, mask2Sup_mem, temp_graph  = \
            supervised_masks_estimation2(sample_weight, confident_sample, args, index, queue, queue_index, mix_index1, mix_index2, epoch, bsz, device)

        ## compute class loss with selected examples

        logits_mask_batch = (torch.ones_like(maskSup_batch) - torch.eye(2 * bsz).to(device))  ## Negatives mask, i.e. all except self-contrast sample

        loss_sup = Supervised_ContrastiveLearning_loss(args, pairwise_comp_batch, maskSup_batch, mask2Sup_batch, maskUnsup_batch, mask2Unsup_batch, logits_mask_batch, lam1, lam2, bsz)
     
        ## using queue
        if args.sup_queue_use == 1 and epoch > args.sup_queue_begin:

            logits_mask_mem = torch.ones_like(maskSup_mem) ## Negatives mask, i.e. all except self-contrast sample

            if queue.ptr == 0:
                logits_mask_mem[:, -2 * bsz:] = logits_mask_batch
            else:
                logits_mask_mem[:, queue.ptr - (2 * bsz):queue.ptr] = logits_mask_batch

            loss_mem = Supervised_ContrastiveLearning_loss(args, pairwise_comp_mem, maskSup_mem, mask2Sup_mem, maskUnsup_mem, mask2Unsup_mem, logits_mask_mem, lam1, lam2, bsz)

            loss_sup = loss_sup + loss_mem
    
            
            sel_mask=((maskSup_batch[:bsz]>0).sum(1)+(maskSup_mem[:bsz]>0).sum(1))<2
        else:
            sel_mask=((maskSup_batch[:bsz]>0).sum(1))<1


        ## compute class loss with selected examples
        try:
            img, _, index  = next(train_selected_loader_iter)
        except StopIteration:
            train_selected_loader_iter = iter(train_selected_loader)
            img, _, index = next(train_selected_loader_iter)

        img1, img2,  labels = img[0].to(device), img[1].to(device), labels.to(device)

        assert torch.all(clean_sample[index] >= 0), "train_select_loader error"
        labels = clean_sample[index].long()

        img1, y_a1, y_b1, mix_index1, lam1 = mix_data_lab(img1, labels, args.alpha_m, device)
        img2, y_a2, y_b2, mix_index2, lam2 = mix_data_lab(img2, labels, args.alpha_m, device)

        predsA, embedA = model(img1)
        predsB, embedB = model(img2)


        lossClassif = ClassificationLoss(args, predsA, predsB, y_a1, y_b1, y_a2, y_b2, mix_index1,
                                            mix_index2, lam1, lam2, criterionCE, epoch, device)
        
        loss_corr = Corr_loss(args, Corr_sample, index_u, img_u_strong, img_u_strong2, model, device)

        ## compute sel_loss by combining uns-cl loss and  sup-cl loss 
    
        sel_loss = loss_sup + (sel_mask*uns_loss).mean()

        loss = args.lambda_c * sel_loss + (lossClassif + loss_corr) + loss_ins * args.lambda_i + args.lambda_l * loss_loc


        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer,loss_id=0) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        scheduler.step()

        moment_update(model, model_ema, args.alpha_moving)
      
        train_loss_1.update((sel_mask*uns_loss).mean().item(), img1.size(0))
        train_loss_2.update(loss_sup.item(), img1.size(0))
        train_loss_3.update(lossClassif.item(), img1.size(0))  
        train_loss_4.update(loss_ins.item(), img1.size(0))  
       
        if counter % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Learning rate: {:.6f}'.format(
                epoch, counter * len(img1), len(train_loader.dataset),
                       100. * counter / len(train_loader), 0,
                optimizer.param_groups[0]['lr']))
         
        counter = counter + 1

    print('train_uns_loss',train_loss_1.avg,'train_sup_loss',train_loss_2.avg,'train_class_loss',train_loss_3.avg,'ssl_loss',train_loss_4.avg)
    print('train time', time.time()-end)

    return soft_labels2


def train_sup(noisy_labels, sample_weight, prototype, optimizer_s, args, scheduler,model,model_ema,contrast,queue,device, train_loader, train_selected_loader, optimizer, epoch):
    train_loss_1 = AverageMeter()
    train_loss_3 = AverageMeter()      

    model.train()
    set_bn_train(model_ema)
    end = time.time()
    counter = 1

    criterionCE = torch.nn.CrossEntropyLoss(reduction="none")
    train_selected_loader_iter = iter(train_selected_loader)
    for batch_idx, (img, labels, index) in enumerate(train_loader):

        img1, img2, labels, index = img[0].to(device), img[1].to(device), labels.to(device), index.to(device)

        bsz = img1.shape[0]

        model.zero_grad()
        
        ## update uns queue
        _,feat_q = model(img1)

        with torch.no_grad():
            _, feat_k= model_ema(img2)

        contrast(feat_q, feat_k, feat_k, update=True)
        
        ##compute sup-cl loss with noisy pairs (adapted from MOIT)
        img1, y_a1, y_b1, mix_index1, lam1 = mix_data_lab(img1, labels, 0, device)
        img2, y_a2, y_b2, mix_index2, lam2 = mix_data_lab(img2, labels, 0, device)

        predsA, embedA = model(img1)
        predsB, embedB = model(img2)
        predsA = F.softmax(predsA,-1)
        predsB = F.softmax(predsB,-1)
        
        with torch.no_grad():
            predsA_ema, embedA_ema = model_ema(img1)
            predsB_ema, embedB_ema = model_ema(img2)
            predsA_ema = F.softmax(predsA_ema,-1)
            predsB_ema = F.softmax(predsB_ema,-1)

        centers = prototype.centers   
        centers = nn.functional.normalize(centers, dim=0)

        embed =  torch.cat((embedA, embedB), dim=0)
        targets_1 = torch.cat((y_a1, y_a2), dim=0)
        targets_2 = torch.cat((y_b1, y_b2), dim=0)
        ones_vec = torch.ones((embedA.size()[0],)).float().to(device)
        lam_vec = torch.cat((lam1 * ones_vec, lam2 * ones_vec), dim=0).to(device)
        centers, x = prototype(embed.detach())
        output = F.log_softmax(2 * x, dim=1)
        loss1 = F.nll_loss(output, targets_1,reduction = 'none') * lam_vec + (1 - lam_vec) * F.nll_loss(output, targets_2,reduction = 'none')
        loss_pro = loss1.mean()
        optimizer_s.zero_grad()
        loss_pro.backward()
        optimizer_s.step()
        
        if args.sup_queue_use == 1:
            queue.enqueue_dequeue(torch.cat((embedA_ema.detach(), embedB_ema.detach()), dim=0), torch.cat((predsA_ema.detach(), predsB_ema.detach()), dim=0), torch.cat((index.detach().squeeze(), index.detach().squeeze()), dim=0))


        if args.sup_queue_use == 1 and epoch > args.sup_queue_begin:
            queue_feats, queue_pros, queue_index = queue.get()
                
        else:
            queue_feats, queue_pros, queue_index = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])

        maskUnsup_batch, maskUnsup_mem, mask2Unsup_batch, mask2Unsup_mem = unsupervised_masks_estimation(args, queue, mix_index1, mix_index2, epoch, bsz, device)

        embeds_batch = torch.cat([embedA, embedB], dim=0)
        pros_batch = torch.cat([predsA, predsB], dim=0)
        pairwise_comp_batch = torch.matmul(embeds_batch, embeds_batch.t())

        if args.sup_queue_use == 1 and epoch > args.sup_queue_begin:
            embeds_mem = torch.cat([embedA, embedB, queue_feats], dim=0)
            pairwise_comp_mem = torch.matmul(embeds_mem[:2 * bsz], embeds_mem[2 * bsz:].t()) 

        maskSup_batch, maskSup_mem, mask2Sup_batch, mask2Sup_mem,_ = \
            supervised_masks_estimation2(sample_weight, noisy_labels, args, index, queue, queue_index, mix_index1, mix_index2, epoch, bsz, device)
                                       
        logits_mask_batch = (torch.ones_like(maskSup_batch) - torch.eye(2 * bsz).to(device))

        loss_sup = Supervised_ContrastiveLearning_loss(args, pairwise_comp_batch, maskSup_batch, mask2Sup_batch, maskUnsup_batch, mask2Unsup_batch, logits_mask_batch, lam1, lam2, bsz)
        
        if args.sup_queue_use == 1 and epoch > args.sup_queue_begin:

            logits_mask_mem = torch.ones_like(maskSup_mem) 

            if queue.ptr == 0:
                logits_mask_mem[:, -2 * bsz:] = logits_mask_batch
            else:
                logits_mask_mem[:, queue.ptr - (2 * bsz):queue.ptr] = logits_mask_batch

            loss_mem = Supervised_ContrastiveLearning_loss(args, pairwise_comp_mem, maskSup_mem, mask2Sup_mem, maskUnsup_mem, mask2Unsup_mem, logits_mask_mem, lam1, lam2, bsz)

            loss_sup = loss_sup + loss_mem 
            
        ## compute class loss with noisy examples
        try:
            img, labels, _  = next(train_selected_loader_iter)
        except StopIteration:
            train_selected_loader_iter = iter(train_selected_loader)
            img, labels, _ = next(train_selected_loader_iter)
        img1, img2,  labels = img[0].to(device), img[1].to(device), labels.to(device)
        
        img1, y_a1, y_b1, mix_index1, lam1 = mix_data_lab(img1, labels, 0, device)
        img2, y_a2, y_b2, mix_index2, lam2 = mix_data_lab(img2, labels, 0, device)


        predsA, embedA = model(img1)
        predsB, embedB = model(img2)


        lossClassif = ClassificationLoss(args, predsA, predsB, y_a1, y_b1, y_a2, y_b2, mix_index1,
                                            mix_index2, lam1, lam2, criterionCE, epoch, device)
        
            
        loss = loss_sup.mean() + lossClassif
        
        with amp.scale_loss(loss, optimizer,loss_id=0) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        scheduler.step()

        moment_update(model, model_ema, args.alpha_moving)
      
        train_loss_1.update(loss_sup.item(), img1.size(0))
        train_loss_3.update(lossClassif.item(), img1.size(0))        
          
        if counter % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Learning rate: {:.6f}'.format(
                epoch, counter * len(img1), len(train_loader.dataset),
                       100. * counter / len(train_loader), 0,
                optimizer.param_groups[0]['lr']))
        counter = counter + 1
    print('train_loss_sup',train_loss_1.avg,'train_class_loss',train_loss_3.avg)
    print('train time', time.time()-end)
    