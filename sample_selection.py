
import torch
import faiss
import numpy as np

def get_soft_labels1(args, features, trainloader):
    """
    features_numpy = features.cpu().numpy()
    index = faiss.IndexFlatIP(features_numpy.shape[1])
    index.add(features_numpy)
    """
    features_numpy = features.cpu().numpy()
    index = faiss.IndexFlatIP(features_numpy.shape[1])
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(features_numpy)

    labels = torch.LongTensor(trainloader.dataset.targets)
    soft_labels = torch.zeros(len(labels), args.num_classes).scatter_(1, labels.view(-1,1), 1)
    
    D,I = index.search(features_numpy,args.k_val+1)  
    neighbors = torch.LongTensor(I)
    weights = torch.exp(torch.Tensor(D[:,1:])/args.sup_t)  #weight is calculated by embeddings' similarity
    N = features_numpy.shape[0]
    score = torch.zeros(N,args.num_classes)
    
    for n in range(N):           
        neighbor_labels = soft_labels[neighbors[n,1:]]
        score[n] = (neighbor_labels*weights[n].unsqueeze(-1)).sum(0)  #aggregate labels from neighbors
    pseudo_labels = torch.max(score,-1)[1]
    soft_labels = torch.zeros(len(pseudo_labels), args.num_classes).scatter_(1, pseudo_labels.view(-1,1), 1)
    
    for n in range(N):           
        neighbor_labels = soft_labels[neighbors[n,1:]]
        score[n] = (neighbor_labels*weights[n].unsqueeze(-1)).sum(0)  #aggregate labels from neighbors
    soft_labels = score/score.sum(1).unsqueeze(-1)
    
    return soft_labels.cuda()

def csample_selection(args, trainloader, soft_labels):

    labels = torch.LongTensor(trainloader.dataset.targets).cuda()
    N = soft_labels.shape[0]
    prob_temp = soft_labels[torch.arange(0, N), labels]
    prob_temp[prob_temp<=1e-2] = 1e-2
    prob_temp[prob_temp > (1-1e-2)] = 1-1e-2
    discrepancy_measure2 = -torch.log(prob_temp)
    agreement_measure = (torch.max(soft_labels, dim=1)[1]==labels).float().data

    ## select examples 
    num_clean_per_class = torch.zeros(args.num_classes).cuda()
    for i in range(args.num_classes):
        idx_class = labels==i
        idx_class = idx_class.float() == 1.0
        num_clean_per_class[i] = torch.sum(agreement_measure[idx_class])
        
    if(args.alpha==0.5):
        num_samples2select_class = torch.median(num_clean_per_class)
    elif(args.alpha==1.0):
        num_samples2select_class = torch.max(num_clean_per_class)
    elif(args.alpha==0.0):
        num_samples2select_class = torch.min(num_clean_per_class)
    else:
        num_samples2select_class = torch.quantile(num_clean_per_class,args.alpha)

    agreement_measure = torch.ones((len(labels),)) * (-1)
    c_weight = torch.zeros((len(labels),)).cuda()

    for i in range(args.num_classes):
        idx_class = labels==i
        samplesPerClass = idx_class.sum()
        idx_class = idx_class.float()# == 1.0
        idx_class = (idx_class==1.0).nonzero().squeeze()
        discrepancy_class = discrepancy_measure2[idx_class]

        if num_samples2select_class>=samplesPerClass:
            k_corrected = samplesPerClass
        else:
            k_corrected = num_samples2select_class

        top_clean_class_relative_idx = torch.topk(discrepancy_class, k=int(k_corrected), largest=False, sorted=False)[1]

        agreement_measure[idx_class[top_clean_class_relative_idx]] = i
        c_weight[idx_class[top_clean_class_relative_idx]] = prob_temp[idx_class[top_clean_class_relative_idx]]

    selected_examples=agreement_measure
  
    return selected_examples.cuda(),c_weight.cuda()


def nsample_selection(soft_labels, threshold, Corr_arr, ssl_ptr, Corr_sample):
    max_probs, max_targets = torch.max(soft_labels, dim=-1)
    mask = max_probs.ge(threshold)
    index_unchoose = mask.nonzero()[:, 0]
    Corr_arr[ssl_ptr,index_unchoose] = max_targets[index_unchoose].float()

    ssl_choose = torch.eq(Corr_arr[0], Corr_arr[1])\
                    & torch.eq(Corr_arr[0], Corr_arr[2])\
                        & (Corr_arr[0] != -1)
    
  
    Corr_sample[ssl_choose] = Corr_arr[0, ssl_choose]
    return max_probs, Corr_sample


