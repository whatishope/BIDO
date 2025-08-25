import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import config



class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)


##################################
# Metric
##################################
class Metric(object):
    pass


class AverageMeter(Metric):
    def __init__(self, name='loss'):
        self.name = name
        self.reset()

    def reset(self):
        self.scores = 0.
        self.total_num = 0.

    def __call__(self, batch_score, sample_num=1):
        self.scores += batch_score
        self.total_num += sample_num
        return self.scores / self.total_num


class TopKAccuracyMetric(Metric):
    def __init__(self, topk=(1,)):
        self.name = 'topk_accuracy'
        self.topk = topk
        self.maxk = max(topk)
        self.reset()

    def reset(self):
        self.corrects = np.zeros(len(self.topk))
        self.num_samples = 0.

    def __call__(self, output, target):
        """Computes the precision@k for the specified values of k"""
        self.num_samples += target.size(0)
        _, pred = output.topk(self.maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for i, k in enumerate(self.topk):
            correct_k = correct[:k].reshape(-1).float().sum(0)
            # correct_k = correct[:k].view(-1).float().sum(0)
            self.corrects[i] += correct_k.item()

        return self.corrects * 100. / self.num_samples


##################################
# Callback
##################################
class Callback(object):
    def __init__(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, *args):
        pass


class ModelCheckpoint(Callback):
    def __init__(self, savepath, monitor='val_topk_accuracy', mode='max'):
        self.savepath = savepath
        self.monitor = monitor
        self.mode = mode
        self.reset()
        super(ModelCheckpoint, self).__init__()

    def reset(self):
        if self.mode == 'max':
            self.best_score = float('-inf')
        else:
            self.best_score = float('inf')

    def set_best_score(self, score):
        if isinstance(score, np.ndarray):
            self.best_score = score[0]
        else:
            self.best_score = score

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, logs, net, **kwargs):
        current_score = logs[self.monitor]
        if isinstance(current_score, np.ndarray):
            current_score = current_score[0]

        if (self.mode == 'max' and current_score > self.best_score) or \
            (self.mode == 'min' and current_score < self.best_score):
            self.best_score = current_score

            if isinstance(net, torch.nn.DataParallel):
                state_dict = net.module.state_dict()
            else:
                state_dict = net.state_dict()

            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()

            if 'feature_center' in kwargs:
                feature_center = kwargs['feature_center']
                feature_center = feature_center.cpu()

                torch.save({
                    'logs': logs,
                    'state_dict': state_dict,
                    'feature_center': feature_center}, self.savepath)
            else:
                torch.save({
                    'logs': logs,
                    'state_dict': state_dict}, self.savepath)



def get_transform(resize, phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.RandomCrop(resize),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.126, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    


def con_loss(features, labels):
    B,_,_=features.shape
    features=F.normalize(features.view(B,-1))
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - 0.4
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    # loss = torch.log(loss)
    loss /= (B*B)
    return loss

import torch
import torch.nn as nn

class MahalanobisMetric(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.L = nn.Parameter(torch.randn(dim, dim))  
        self.L = nn.Parameter(torch.eye(dim) + 0.01 * torch.randn(dim, dim))

    def forward(self, x1, x2):
       
        M = self.L.T @ self.L  
        diff = x1 - x2  # [B * B, D]
        dist = torch.sum((diff @ M) * diff, dim=1)  # [B * B]
        return dist

import torch.nn.functional as F

def con_loss_mahalanobis(features, labels, metric, margin=0.4, pos_weight=1.0, neg_weight=1.0):
    
    B = features.size(0)
    labels = labels.view(-1, 1)

  
    pos_mask = (labels == labels.T).float()
    neg_mask = 1.0 - pos_mask
    diag = torch.eye(B, device=features.device)
    pos_mask -= diag
    neg_mask *= (1 - diag)

    
    feat1 = features.unsqueeze(1).expand(B, B, -1).reshape(-1, features.size(1))  # [B*B, D]
    feat2 = features.unsqueeze(0).expand(B, B, -1).reshape(-1, features.size(1))  # [B*B, D]

    # print("L requires_grad:", metric.L.requires_grad) 


 
    dists = metric(feat1, feat2).reshape(B, B)  # [B, B]

   
    pos_loss = dists * pos_mask
    pos_loss = pos_loss.sum() / (pos_mask.sum() + 1e-8)

   
    neg_margin = torch.clamp(margin - dists, min=0.0)
    neg_loss = neg_margin * neg_mask
    neg_loss = neg_loss.sum() / (neg_mask.sum() + 1e-8)

    loss = pos_weight * pos_loss + neg_weight * neg_loss
    return loss





##################################
# crop and drop
##################################
def crop_drop(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map=attention_map[batch_index]
            # atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            # atten_map=[26,26],(imgH, imgW)=(500,500)
            # F.upsample_bilinear--上采样需要输入的维度是[1,1,26,26]
            # 因此引入unsqueeze
            atten_map=atten_map.unsqueeze(dim=0)
            atten_map=atten_map.unsqueeze(dim=0)
            crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c
            # crop_mask[0, 0, ...]=[500,500]
            # nonzero_indices中保留了不为0元素的位置
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            # 利用F.upsample_bilinear采样将image的区域height_min:height_max, width_min:width_max放大到(imgH, imgW)
            crop_images.append(
                F.upsample_bilinear(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)
    
class classifier(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.input=embeddings.view(embeddings.size(0),-1)
        self.layers=nn.ModuleList(
            [
                torch.nn.BatchNorm1d(self.input.size(-1)),
                nn.Linear(self.input.size(-1),2048),
                torch.nn.BatchNorm1d(2048),
                nn.Linear(2048,config.num_classes),
                torch.nn.BatchNorm1d(config.num_classes)      
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x=layer(x)
        return x
