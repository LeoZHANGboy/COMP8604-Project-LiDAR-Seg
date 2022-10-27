import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_utils as pt_utils
from helper_tool import DataProcessing as DP
import numpy as np
from sklearn.metrics import confusion_matrix
import sys
import os
import warnings
import copy
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

warnings.filterwarnings(action='ignore')
sys.stderr = open(os.devnull, "w")  # silence stderr
sys.stderr = sys.__stderr__  # unsilence stderr


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, n_head, d_forward=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = torch.nn.MultiheadAttention(d_model, n_head)
        # self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.w_1 = nn.Linear(d_model, d_forward)  # position-wise
        self.w_2 = nn.Linear(d_forward, d_model)  # position-wise
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU(True)

    def forward(self, q, enc_src):
        enc_src2 = self.slf_attn(q, enc_src, enc_src)[0]

        enc_src = enc_src + self.dropout1(enc_src2)

        enc_src = self.norm1(enc_src)

        enc_src2 = self.w_2(self.dropout(self.activation(self.w_1(enc_src))))
        enc_src = enc_src + self.dropout2(enc_src2)
        enc_src = self.norm2(enc_src)

        return enc_src


class TransformerEncoder(nn.Module):

    def __init__(
            self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.ffn = nn.Sequential(
            nn.Linear(20, 16),
            nn.ReLU(True),
            nn.Linear(16, 1)
        )
        self.norm = norm

    def forward(self, q, src):
        # src = src.permute(1, 0, 2)
        output = src

        for mod in self.layers:
            output = mod(q, output)

        output = output.permute(1, 2, 0)  # From seq_len, batch_size, dim to BS, dim, seq_len
        # BS = output.size()[0]
        output = self.ffn(output)
        output = output.squeeze(-1)  # Final BS, dim
        return output


def cart2hom(pts_3d):
    """ Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    """
    n = pts_3d.size()[0]
    pts_3d_hom = torch.cat([pts_3d, torch.ones(n, 1, device=pts_3d.device)], dim=-1)
    return pts_3d_hom


def l2I(pts_3d, calib):
    v2c = calib[1]
    p = calib[0]
    pts_3d_rect = torch.matmul(cart2hom(pts_3d), torch.t(v2c))
    pts_3d_rect = cart2hom(pts_3d_rect)
    pts_2d = torch.matmul(pts_3d_rect, torch.t(p))
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def fuse_feat(fpn_feat: dict, pts_3d, pc_feat, trans, calib):
    """
    :param fpn_feat: dict 4 levels of image feature, for each: (BS, 256, (size/4))
    :param pts_3d: point cloud projection on image coor (BS, num_points, 3)
    :param pc_feat: point_cloud feature (BS, 512, num_points)
    :param trans: transformer module for feature match
    :param calib: calib matrix (BS,2,4,3) Tr first, P2 second
    :return: sampled_feat (BS, num_points, 36, 256) 4 level * 9 each
    """
    ## padding and put channel at last.

    fpn_feat = {key: torch.cat([val, val], dim=1) for key, val in fpn_feat.items()}
    fpn_feat = {key: nn.ConstantPad2d(1, 0)(val).permute(0, 2, 3, 1) for key, val in fpn_feat.items()}

    BS = pts_3d.size()[0]
    num_pts = pts_3d.size()[1]
    fused_feat = torch.zeros(BS, num_pts, 1024).to(pts_3d.device)
    for i in range(BS):
        # print(f'currently {i}th sample')
        cur_pts_3d = pts_3d[i]
        cur_sampled_feat = torch.zeros(20, num_pts, 512).to(pts_3d.device)
        cur_pts_2d = l2I(cur_pts_3d, calib[i])
        fov_inds = (
                (cur_pts_2d[:, 0] < 1226)
                & (cur_pts_2d[:, 0] >= 0)
                & (cur_pts_2d[:, 1] < 370)
                & (cur_pts_2d[:, 1] >= 0)
        )
        # print(torch.sum(fov_inds.long()))
        fov_inds = (fov_inds & (cur_pts_3d[:, 0] > 2.0)).unsqueeze(1)
        pts_2d_rect = fov_inds.long() * cur_pts_2d  # num_points, 2
        # print(torch.sum(fov_inds.long()))
        # pts_2d_l0 = torch.floor(torch.div(pts_2d_rect, 4.)).long()+1
        # pts_2d_l1 = torch.floor(torch.div(pts_2d_rect, 8.)).long()+1
        # pts_2d_l2 = torch.floor(torch.div(pts_2d_rect, 16.)).long()+1
        # pts_2d_l3 = torch.floor(torch.div(pts_2d_rect, 32.)).long()+1
        pts_2d_l0 = ((pts_2d_rect // 4) + 1).long()
        pts_2d_l0 = (
                            (pts_2d_l0[:, 0] < 308)
                            & (pts_2d_l0[:, 0] > 0)
                            & (pts_2d_l0[:, 1] < 94)
                            & (pts_2d_l0[:, 1] > 0)
                    ).unsqueeze(1).long() * pts_2d_l0

        pts_2d_l1 = ((pts_2d_rect // 8) + 1).long()
        pts_2d_l1 = (
                            (pts_2d_l1[:, 0] < 155)
                            & (pts_2d_l1[:, 0] > 0)
                            & (pts_2d_l1[:, 1] < 48)
                            & (pts_2d_l1[:, 1] > 0)
                    ).unsqueeze(1).long() * pts_2d_l1
        pts_2d_l2 = ((pts_2d_rect // 16) + 1).long()
        pts_2d_l2 = (
                            (pts_2d_l2[:, 0] < 78)
                            & (pts_2d_l2[:, 0] > 0)
                            & (pts_2d_l2[:, 1] < 25)
                            & (pts_2d_l2[:, 1] > 0)
                    ).unsqueeze(1).long() * pts_2d_l2
        pts_2d_l3 = ((pts_2d_rect // 32) + 1).long()
        pts_2d_l3 = (
                            (pts_2d_l3[:, 0] < 40)
                            & (pts_2d_l3[:, 0] > 0)
                            & (pts_2d_l3[:, 1] < 13)
                            & (pts_2d_l3[:, 1] > 0)
                    ).unsqueeze(1).long() * pts_2d_l3

        cur_sampled_feat[0, ...] = fpn_feat['0'][i][pts_2d_l0[:, 1], pts_2d_l0[:, 0], :]
        cur_sampled_feat[1, ...] = fpn_feat['0'][i][pts_2d_l0[:, 1] + 1, pts_2d_l0[:, 0], :]
        cur_sampled_feat[2, ...] = fpn_feat['0'][i][pts_2d_l0[:, 1] + 1, pts_2d_l0[:, 0] + 1, :]
        cur_sampled_feat[3, ...] = fpn_feat['0'][i][pts_2d_l0[:, 1] + 1, pts_2d_l0[:, 0] - 1, :]
        cur_sampled_feat[4, ...] = fpn_feat['0'][i][pts_2d_l0[:, 1], pts_2d_l0[:, 0] + 1, :]
        cur_sampled_feat[5, ...] = fpn_feat['0'][i][pts_2d_l0[:, 1], pts_2d_l0[:, 0] - 1, :]
        cur_sampled_feat[6, ...] = fpn_feat['0'][i][pts_2d_l0[:, 1] - 1, pts_2d_l0[:, 0], :]
        cur_sampled_feat[7, ...] = fpn_feat['0'][i][pts_2d_l0[:, 1] - 1, pts_2d_l0[:, 0] + 1, :]
        cur_sampled_feat[8, ...] = fpn_feat['0'][i][pts_2d_l0[:, 1] - 1, pts_2d_l0[:, 0] - 1, :]

        cur_sampled_feat[9, ...] = fpn_feat['1'][i][pts_2d_l1[:, 1], pts_2d_l1[:, 0], :]
        cur_sampled_feat[10, ...] = fpn_feat['1'][i][pts_2d_l1[:, 1] + 1, pts_2d_l1[:, 0], :]
        cur_sampled_feat[11, ...] = fpn_feat['1'][i][pts_2d_l1[:, 1] + 1, pts_2d_l1[:, 0] + 1, :]
        cur_sampled_feat[12, ...] = fpn_feat['1'][i][pts_2d_l1[:, 1] + 1, pts_2d_l1[:, 0] - 1, :]
        cur_sampled_feat[13, ...] = fpn_feat['1'][i][pts_2d_l1[:, 1], pts_2d_l1[:, 0] + 1, :]
        cur_sampled_feat[14, ...] = fpn_feat['1'][i][pts_2d_l1[:, 1], pts_2d_l1[:, 0] - 1, :]
        cur_sampled_feat[15, ...] = fpn_feat['1'][i][pts_2d_l1[:, 1] - 1, pts_2d_l1[:, 0], :]
        cur_sampled_feat[16, ...] = fpn_feat['1'][i][pts_2d_l1[:, 1] - 1, pts_2d_l1[:, 0] + 1, :]
        cur_sampled_feat[17, ...] = fpn_feat['1'][i][pts_2d_l1[:, 1] - 1, pts_2d_l1[:, 0] - 1, :]

        cur_sampled_feat[18, ...] = fpn_feat['2'][i][pts_2d_l2[:, 1], pts_2d_l2[:, 0]]
        cur_sampled_feat[19, ...] = fpn_feat['3'][i][pts_2d_l3[:, 1], pts_2d_l3[:, 0]]

        cur_img_feat = trans(torch.t(pc_feat[i]).unsqueeze(0), cur_sampled_feat)  # transpose from 512,704 to 1, 704,512
        cur_img_feat = fov_inds.long() * cur_img_feat
        fused_feat[i] = torch.cat([torch.t(pc_feat[i]), cur_img_feat], dim=-1)
    return fused_feat.permute(0, 2, 1)  ## now 4, 704, 1024 to 4, 1024, 704

def fuse_feat_cat(fpn_feat: dict, pts_3d, pc_feat, calib):
    """
    :param fpn_feat: dict 4 levels of image feature, for each: (BS, 256, (size/4))
    :param pts_3d: point cloud projection on image coor (BS, num_points, 3)
    :param pc_feat: point_cloud feature (BS, 512, num_points)
    :param trans: transformer module for feature match
    :param calib: calib matrix (BS,2,4,3) Tr first, P2 second
    :return: sampled_feat (BS, num_points, 36, 256) 4 level * 9 each
    """
    ## padding and put channel at last.

    fpn_feat = {key: torch.cat([val, val], dim=1) for key, val in fpn_feat.items()}
    fpn_feat = {key: nn.ConstantPad2d(1, 0)(val).permute(0, 2, 3, 1) for key, val in fpn_feat.items()}

    BS = pts_3d.size()[0]
    num_pts = pts_3d.size()[1]
    fused_feat = torch.zeros(BS, num_pts, 512*21).to(pts_3d.device)
    for i in range(BS):
        # print(f'currently {i}th sample')
        cur_pts_3d = pts_3d[i]
        cur_sampled_feat = torch.zeros(20, num_pts, 512).to(pts_3d.device)
        cur_pts_2d = l2I(cur_pts_3d, calib[i])
        fov_inds = (
                (cur_pts_2d[:, 0] < 1226)
                & (cur_pts_2d[:, 0] >= 0)
                & (cur_pts_2d[:, 1] < 370)
                & (cur_pts_2d[:, 1] >= 0)
        )
        # print(torch.sum(fov_inds.long()))
        fov_inds = (fov_inds & (cur_pts_3d[:, 0] > 2.0)).unsqueeze(1)
        pts_2d_rect = fov_inds.long() * cur_pts_2d  # num_points, 2
        # print(torch.sum(fov_inds.long()))
        # pts_2d_l0 = torch.floor(torch.div(pts_2d_rect, 4.)).long()+1
        # pts_2d_l1 = torch.floor(torch.div(pts_2d_rect, 8.)).long()+1
        # pts_2d_l2 = torch.floor(torch.div(pts_2d_rect, 16.)).long()+1
        # pts_2d_l3 = torch.floor(torch.div(pts_2d_rect, 32.)).long()+1
        pts_2d_l0 = ((pts_2d_rect // 4) + 1).long()
        pts_2d_l0 = (
                            (pts_2d_l0[:, 0] < 308)
                            & (pts_2d_l0[:, 0] > 0)
                            & (pts_2d_l0[:, 1] < 94)
                            & (pts_2d_l0[:, 1] > 0)
                    ).unsqueeze(1).long() * pts_2d_l0

        pts_2d_l1 = ((pts_2d_rect // 8) + 1).long()
        pts_2d_l1 = (
                            (pts_2d_l1[:, 0] < 155)
                            & (pts_2d_l1[:, 0] > 0)
                            & (pts_2d_l1[:, 1] < 48)
                            & (pts_2d_l1[:, 1] > 0)
                    ).unsqueeze(1).long() * pts_2d_l1
        pts_2d_l2 = ((pts_2d_rect // 16) + 1).long()
        pts_2d_l2 = (
                            (pts_2d_l2[:, 0] < 78)
                            & (pts_2d_l2[:, 0] > 0)
                            & (pts_2d_l2[:, 1] < 25)
                            & (pts_2d_l2[:, 1] > 0)
                    ).unsqueeze(1).long() * pts_2d_l2
        pts_2d_l3 = ((pts_2d_rect // 32) + 1).long()
        pts_2d_l3 = (
                            (pts_2d_l3[:, 0] < 40)
                            & (pts_2d_l3[:, 0] > 0)
                            & (pts_2d_l3[:, 1] < 13)
                            & (pts_2d_l3[:, 1] > 0)
                    ).unsqueeze(1).long() * pts_2d_l3

        cur_sampled_feat[0, ...] = fpn_feat['0'][i][pts_2d_l0[:, 1], pts_2d_l0[:, 0], :]
        cur_sampled_feat[1, ...] = fpn_feat['0'][i][pts_2d_l0[:, 1] + 1, pts_2d_l0[:, 0], :]
        cur_sampled_feat[2, ...] = fpn_feat['0'][i][pts_2d_l0[:, 1] + 1, pts_2d_l0[:, 0] + 1, :]
        cur_sampled_feat[3, ...] = fpn_feat['0'][i][pts_2d_l0[:, 1] + 1, pts_2d_l0[:, 0] - 1, :]
        cur_sampled_feat[4, ...] = fpn_feat['0'][i][pts_2d_l0[:, 1], pts_2d_l0[:, 0] + 1, :]
        cur_sampled_feat[5, ...] = fpn_feat['0'][i][pts_2d_l0[:, 1], pts_2d_l0[:, 0] - 1, :]
        cur_sampled_feat[6, ...] = fpn_feat['0'][i][pts_2d_l0[:, 1] - 1, pts_2d_l0[:, 0], :]
        cur_sampled_feat[7, ...] = fpn_feat['0'][i][pts_2d_l0[:, 1] - 1, pts_2d_l0[:, 0] + 1, :]
        cur_sampled_feat[8, ...] = fpn_feat['0'][i][pts_2d_l0[:, 1] - 1, pts_2d_l0[:, 0] - 1, :]

        cur_sampled_feat[9, ...] = fpn_feat['1'][i][pts_2d_l1[:, 1], pts_2d_l1[:, 0], :]
        cur_sampled_feat[10, ...] = fpn_feat['1'][i][pts_2d_l1[:, 1] + 1, pts_2d_l1[:, 0], :]
        cur_sampled_feat[11, ...] = fpn_feat['1'][i][pts_2d_l1[:, 1] + 1, pts_2d_l1[:, 0] + 1, :]
        cur_sampled_feat[12, ...] = fpn_feat['1'][i][pts_2d_l1[:, 1] + 1, pts_2d_l1[:, 0] - 1, :]
        cur_sampled_feat[13, ...] = fpn_feat['1'][i][pts_2d_l1[:, 1], pts_2d_l1[:, 0] + 1, :]
        cur_sampled_feat[14, ...] = fpn_feat['1'][i][pts_2d_l1[:, 1], pts_2d_l1[:, 0] - 1, :]
        cur_sampled_feat[15, ...] = fpn_feat['1'][i][pts_2d_l1[:, 1] - 1, pts_2d_l1[:, 0], :]
        cur_sampled_feat[16, ...] = fpn_feat['1'][i][pts_2d_l1[:, 1] - 1, pts_2d_l1[:, 0] + 1, :]
        cur_sampled_feat[17, ...] = fpn_feat['1'][i][pts_2d_l1[:, 1] - 1, pts_2d_l1[:, 0] - 1, :]

        cur_sampled_feat[18, ...] = fpn_feat['2'][i][pts_2d_l2[:, 1], pts_2d_l2[:, 0]]
        cur_sampled_feat[19, ...] = fpn_feat['3'][i][pts_2d_l3[:, 1], pts_2d_l3[:, 0]]
        # transpose from 512,704 to 1, 704,512
        cur_sampled_feat = fov_inds.long() * cur_sampled_feat
        cur_sampled_feat = cur_sampled_feat.permute(1, 0, 2).contiguous().view(num_pts,-1)  # (20, num_pts, 512) to (num_pts, 20, 512) to (num_pts, 20 * 512)
        fused_feat[i] = torch.cat([torch.t(pc_feat[i]), cur_sampled_feat], dim=-1)
    return fused_feat.permute(0, 2, 1)  ## now 4, 704, 21*512 to 4, 21*512, 704

# class CNN


class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.class_weights = DP.get_class_weights('SemanticKITTI')

        if self.config.img_require:
            self.cnn_fpn = resnet_fpn_backbone('resnet101', pretrained=True, norm_layer=None, trainable_layers=5)
            self.trans = TransformerEncoder(EncoderLayer(512, 4), 4)

        self.fc0 = pt_utils.Conv1d(3, 8, kernel_size=1, bn=True)

        self.dilated_res_blocks = nn.ModuleList()
        d_in = 8
        for i in range(self.config.num_layers):
            d_out = self.config.d_out[i]
            cur_drb = Dilated_res_block(d_in, d_out)
            for para in cur_drb.parameters():
                para.requires_grad = False
            self.dilated_res_blocks.append(cur_drb)
            d_in = 2 * d_out

        d_out = d_in
        # d_in = 512
        if self.config.img_require:
            self.decoder_ini = pt_utils.Conv2d(d_in * 2, d_out, kernel_size=(1, 1), bn=True)  # modify here 21 for cat, 2 for regular
        else:
            self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
            # self.decoder_1 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        self.decoder_blocks = nn.ModuleList()



        for j in range(self.config.num_layers):
            if j < 3:
                d_in = d_out + 2 * self.config.d_out[-j - 2]
                d_out = 2 * self.config.d_out[-j - 2]
            else:
                d_in = 4 * self.config.d_out[-4]
                d_out = 2 * self.config.d_out[-4]
            # if self.config.img_require:
            #     self.decoder_blocks_lv.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True))
            # else:
            self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True))  ## Modify here
        self.fc1 = pt_utils.Conv2d(d_out, 64, kernel_size=(1, 1), bn=True)
        self.fc2 = pt_utils.Conv2d(64, 32, kernel_size=(1, 1), bn=True)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = pt_utils.Conv2d(32, self.config.num_classes, kernel_size=(1, 1), bn=False, activation=None)

    def forward(self, end_points):

        features = end_points['features']  # Batch*channel*npoints
        features = self.fc0(features)

        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](features, end_points['xyz'][i], end_points['neigh_idx'][i])
            if self.config.img_require and i == 3:
                feat_fpn = self.cnn_fpn(end_points['img'])
                pts_3d = end_points['xyz'][3]
                # fused_feat = fuse_feat_cat(fpn_feat=feat_fpn, pts_3d=pts_3d, pc_feat=f_encoder_i[..., 0], calib=end_points['calib'])
                fused_feat = fuse_feat(fpn_feat=feat_fpn, pts_3d=pts_3d, pc_feat=f_encoder_i[..., 0],trans=self.trans,
                                           calib=end_points['calib'])
                f_encoder_i = fused_feat.unsqueeze(-1)
            f_sampled_i = self.random_sample(f_encoder_i, end_points['sub_idx'][i])
            features = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # TODO: Fusion here

        """
        inside f_encoder_list 4 levels of features, last two will(possibly) be fused with image (BS, 512, 176,1)
        and (BS, 256, 704, 1)
        """
        # ###########################Encoder############################
        if self.config.img_require:
            features = self.decoder_ini(f_encoder_list[-1])
        else:
            features = self.decoder_0(f_encoder_list[-1])
            # features = self.decoder_1(f_encoder_list[-1])

        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(features, end_points['interp_idx'][-j - 1])
            # if self.config.img_require:
            #     f_decoder_i = self.decoder_blocks_lv[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))
            # else:
            f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))

            features = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################

        features = self.fc1(features)
        features = self.fc2(features)
        features = self.dropout(features)
        features = self.fc3(features)
        f_out = features.squeeze(3)

        end_points['logits'] = f_out
        return end_points

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features


def compute_acc(end_points):
    logits = end_points['valid_logits']
    labels = end_points['valid_labels']
    logits = logits.max(dim=1)[1]
    acc = (logits == labels).sum().float() / float(labels.shape[0])
    end_points['acc'] = acc
    return acc, end_points


class IoUCalculator:
    def __init__(self, cfg):
        self.gt_classes = [0 for _ in range(cfg.num_classes)]
        self.positive_classes = [0 for _ in range(cfg.num_classes)]
        self.true_positive_classes = [0 for _ in range(cfg.num_classes)]
        self.cfg = cfg

    def add_data(self, end_points):
        logits = end_points['valid_logits']
        labels = end_points['valid_labels']
        pred = logits.max(dim=1)[1]
        pred_valid = pred.detach().cpu().numpy()
        labels_valid = labels.detach().cpu().numpy()

        val_total_correct = 0
        val_total_seen = 0

        correct = np.sum(pred_valid == labels_valid)
        val_total_correct += correct
        val_total_seen += len(labels_valid)

        conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.cfg.num_classes, 1))
        self.gt_classes += np.sum(conf_matrix, axis=1)
        self.positive_classes += np.sum(conf_matrix, axis=0)
        self.true_positive_classes += np.diagonal(conf_matrix)

    def compute_iou(self):
        iou_list = []
        for n in range(0, self.cfg.num_classes, 1):
            if float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n]) != 0:
                iou = self.true_positive_classes[n] / float(
                    self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n])
                iou_list.append(iou)
            else:
                iou_list.append(0.0)
        mean_iou = sum(iou_list) / float(self.cfg.num_classes)
        return mean_iou, iou_list


class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = pt_utils.Conv2d(d_in, d_out // 2, kernel_size=(1, 1), bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = pt_utils.Conv2d(d_out, d_out * 2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out * 2, kernel_size=(1, 1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # Batch*d_out*npoints*1
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc + shortcut, negative_slope=0.2)


class Building_block(nn.Module):
    def __init__(self, d_out):  # d_in = d_out//2
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(10, d_out // 2, kernel_size=(1, 1), bn=True)
        self.att_pooling_1 = Att_pooling(d_out, d_out // 2)

        self.mlp2 = pt_utils.Conv2d(d_out // 2, d_out // 2, kernel_size=(1, 1), bn=True)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):  # feature: Batch*channel*npoints*1
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*10
        f_xyz = f_xyz.permute((0, 3, 1, 2))  # batch*10*npoint*nsamples
        f_xyz = self.mlp1(f_xyz)
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)),
                                             neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_1(f_concat)  # Batch*channel*npoints*1

        f_xyz = self.mlp2(f_xyz)
        f_neighbours = self.gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)),
                                             neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # batch*npoint*nsamples*3

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3
        relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3
        relative_dis = torch.sqrt(
            torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))  # batch*npoint*nsamples*1
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz],
                                     dim=-1)  # batch*npoint*nsamples*10
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
        return features


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        # self.trans = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_in, 2), num_layers=2)
        # self.agg = nn.Sequential(nn.Linear(16, 16),
        #                          nn.ReLU(True),
        #                          nn.Linear(16, 1))

    def forward(self, feature_set):
        # feature_set: batch, channel, npoints, nsamples
        att_activation = self.fc(feature_set)
        ###############################################
        # BS = att_activation.size()[0]
        # att_activation = att_activation.permute(3, 0, 2, 1)  # nsamples, batch, npoints, channel
        # att_activation = att_activation.view(16, -1, self.d_in)  # nsamples, batch * npoints, channel
        # weighted_act = self.trans(att_activation)  # nsamples, batch * npoints, channel
        # weighted_act = weighted_act.permute(1, 2, 0)  # batch * npoints, channel, nsamples
        # weighted_act = weighted_act.view(BS, -1, self.d_in, 16)  # batch, npoints, channel, nsamples
        # weighted_act = self.agg(weighted_act)  # batch, npoints, channel, 1
        # f_agg = weighted_act.permute(BS, self.d_in, -1, 1)
        ################################################
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg


def compute_loss(end_points, cfg):
    logits = end_points['logits']
    labels = end_points['labels']

    logits = logits.transpose(1, 2).reshape(-1, cfg.num_classes)
    labels = labels.reshape(-1)

    # Boolean mask of points that should be ignored
    ignored_bool = labels == 0
    for ign_label in cfg.ignored_label_inds:
        ignored_bool = ignored_bool | (labels == ign_label)
    # Collect logits and labels that are not ignored
    valid_idx = ignored_bool == 0
    valid_logits = logits[valid_idx, :]
    valid_labels_init = labels[valid_idx]

    # Reduce label values in the range of logit shape
    reducing_list = torch.range(0, cfg.num_classes).long().cuda()
    # reducing_list = torch.range(0, cfg.num_classes).long()
    inserted_value = torch.zeros((1,)).long().cuda()
    # inserted_value = torch.zeros((1,)).long()
    for ign_label in cfg.ignored_label_inds:
        reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
    valid_labels = torch.gather(reducing_list, 0, valid_labels_init)
    loss = get_loss(valid_logits, valid_labels, cfg.class_weights)
    end_points['valid_logits'], end_points['valid_labels'] = valid_logits, valid_labels
    end_points['loss'] = loss
    return loss, end_points


def get_loss(logits, labels, pre_cal_weights):
    # calculate the weighted cross entropy according to the inverse frequency
    class_weights = torch.from_numpy(pre_cal_weights).float().cuda()
    # class_weights = torch.from_numpy(pre_cal_weights).float()
    # one_hot_labels = F.one_hot(labels, self.config.num_classes)

    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    output_loss = criterion(logits, labels)
    output_loss = output_loss.mean()
    return output_loss
