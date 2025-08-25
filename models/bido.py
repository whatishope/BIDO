import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from torchvision.models import resnet18
from models.inception import inception_v3, BasicConv2d
from models.multihead_attention import SelfAttentionLayer as SAL
from torch.nn import MultiheadAttention

import os
import sys
parent_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import config

EPSILON = 1e-12 


# Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W))
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        feature_matrix = F.normalize(feature_matrix, dim=-1)
        return feature_matrix

class AMG(nn.Module):
    def __init__(self, in_channels, num_masks):
        super(AMG, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_masks, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        # x = self.softmax(x)
        return x

# BIDO: Image-based Interpretable Malware Detector
class BIDO(nn.Module):
    def __init__(self, num_classes, M=32, net='inception_mixed_6e', pretrained=False, num_attn_layers=3, D_xml=512):
        super(BIDO, self).__init__()
        self.num_classes = num_classes
        self.M = M
        self.net = net
        self.D_xml = D_xml
        self.num_atten_layer = num_attn_layers
        self.atten_layer = nn.ModuleList(
            [
                SAL(model_dim=768, feed_forward_dim=2048, num_heads=8, dropout=0.1, mask=False)
                for _ in range(num_attn_layers)
            ]
        )

        # Network Initialization
        if 'inception' in net:
            if net == 'inception_mixed_6e':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_6e()
                self.num_features = 768
            elif net == 'inception_mixed_7c':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_7c()
                self.num_features = 2048
            else:
                raise ValueError('Unsupported net: %s' % net)


        self.attentions = BasicConv2d(self.num_features, self.M, kernel_size=1)
        self.amg = AMG(self.num_features, self.M)


        self.bap = BAP(pool='GAP')

        self.dropout = torch.nn.Dropout(config.drop_rate)


        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.num_features))  # [1, 1, 768]
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.M + 1, self.num_features))  # [1, 33, 768]


        self.fc = nn.Linear(self.num_features, self.num_classes, bias=False)

        self.fc1 = nn.Linear(self.M * self.num_features, self.num_classes, bias=False)

        self.fc2 = nn.Linear((self.M + 1) * self.num_features, self.num_classes, bias=False)

        self.fusion_fc = nn.Linear((self.M + 1) * self.num_features * self.D_xml, self.num_classes,
                                   bias=False)  # Fusion output

        self.xml_cnn = resnet18(pretrained=True)
        self.xml_cnn.fc = nn.Identity()  # [B, 512]

        self.xml_proj = nn.Linear(512, D_xml)  # [B, D_xml]

        self.emb_proj = nn.Sequential(
            nn.Linear(25344, 8192),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 512),
            nn.LayerNorm(512)
        )

        self.fused_att = MultiheadAttention(embed_dim=D_xml, num_heads=8,batch_first=True)  # batch_first=True
        self.fused_fc = nn.Linear(512 * self.D_xml, self.num_classes)

        self.cross_attn_fusion = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.fusion_fc = nn.Linear(512, self.num_classes)

        self.fusion_c_fc = nn.Linear(1024, num_classes)

        self.xml_fc = nn.Linear(512, self.num_classes)

        logging.info(
            'IIDM: using {} as feature extractor, num_classes: {}, num_attentions: {}'.format(net, self.num_classes,
                                                                                              self.M))

    def forward(self, x, x_xml_img):
        batch_size = x.size(0)

        feature_maps = self.features(x) #torch.Size([16, 768, 26, 26])
        # print(feature_maps.shape)
        if self.net != 'inception_mixed_7c':
            attention_maps = self.amg(feature_maps)
        else:
            attention_maps = feature_maps[:, :self.M, ...]
        feature_matrix = self.bap(feature_maps, attention_maps)

        p1 = self.fc1(F.normalize(feature_matrix.view(batch_size, -1), dim=-1) * 100.)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, 768]
        embeddings = torch.cat((feature_matrix, cls_tokens), dim=1)  # [B, 33, 768]
        embeddings = embeddings + self.pos_embedding

        for attn in self.atten_layer:
            embeddings = attn(embeddings)

        p2 = self.fc2(F.normalize(embeddings.view(batch_size, -1), dim=-1) * 100.)

        # XML image features
        xml_feat = self.xml_cnn(x_xml_img) # [B, 512]
        xml_cls = self.xml_fc(xml_feat)



        emb_flat = embeddings.view(batch_size, -1) # [B, 25344]
        emb_flat = self.emb_proj(emb_flat)  # [B, 512]
        fused_feats = torch.bmm(emb_flat.unsqueeze(2), xml_feat.unsqueeze(1))  # [B, (M+1)*C, D_xml]

        fused_flat = fused_feats.reshape(batch_size, -1)
        p3 = self.fused_fc(fused_flat)
        # p3 = self.fusion_fc(fused_flat)

        # p3 = self.fusion_fc(fused_feats)

        # [B, N, D_xml]，N=(M+1)*C
        # fused_feats = self.fused_transformer(fused_feats)


        return p1, p2, embeddings, p3, xml_cls

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            logging.info('%s: All params loaded' % type(self).__name__)
        else:
            logging.info('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            logging.info(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

        model_dict.update(pretrained_dict)
        super(BIDO, self).load_state_dict(model_dict)


