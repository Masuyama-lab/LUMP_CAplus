from tqdm import tqdm
import torch.nn.functional as F 
import torch
import numpy as np
from utils.metrics import mask_classes

# code copied from https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb#scrollTo=RI1Y8bSImD7N
# test using a knn monitor
def knn_monitor(net, dataset, memory_data_loader, test_data_loader, device, cl_default, task_id, k=200, t=0.1, hide_progress=False):
    net.eval()
    # classes = len(memory_data_loader.dataset.classes)
    classes = 100
    total_top1 = total_top1_mask = total_top5 = total_num = 0.0
    feature_bank = []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting', leave=False, disable=True):
            if cl_default:
                feature = net(data.cuda(non_blocking=True), return_features=True)
            else:
                # feature = net(data.cuda(non_blocking=True))
                feature = net(data.to(device, non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        # feature_labels = torch.tensor(memory_data_loader.dataset.targets - np.amin(memory_data_loader.dataset.targets), device=feature_bank.device)
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader, desc='kNN', disable=True)
        for data, target in test_bar:
            # data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            if cl_default:
                feature = net(data, return_features=True)
            else:
                feature = net(data)
            feature = F.normalize(feature, dim=1)
            
            pred_scores = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

            total_num += data.shape[0]
            _, preds = torch.max(pred_scores.data, 1)
            total_top1 += torch.sum(preds == target).item()
            
            pred_scores = mask_classes(pred_scores, dataset, task_id)
            _, preds = torch.max(pred_scores.data, 1)
            total_top1_mask += torch.sum(preds == target).item()

    return total_top1 / total_num * 100, total_top1_mask / total_num * 100


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    return pred_scores
