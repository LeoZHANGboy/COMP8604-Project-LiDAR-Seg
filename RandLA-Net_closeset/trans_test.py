import torch
import torch.nn as nn
import copy
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


# import torch.nn.functional as F


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
    pts_3d_hom = torch.cat([pts_3d, torch.ones(n, 1,device=pts_3d.device)], dim=-1)
    return pts_3d_hom

def l2I(pts_3d,calib):
    v2c = calib[0]
    p = calib[1]
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
    :param pc_feat: point_cloud feature (BS, num_points, 256)
    :param trans: transformer module for feature match
    :param calib: calib matrix (BS,2,4,3) Tr first, P2 second
    :return: sampled_feat (BS, num_points, 36, 256) 4 level * 9 each
    """
    BS = pts_3d.size()[0]
    num_pts = pts_3d.size()[1]
    fused_feat = torch.zeros(BS, num_pts, 512)
    for i in range(BS):
        cur_pts_3d = pts_3d[i]
        cur_sampled_feat = torch.zeros(20, num_pts, 256).to(pts_3d.device)
        cur_pts_2d = l2I(cur_pts_3d, calib[i])
        fov_inds = (
                (cur_pts_2d[:, 0] < 312)
                & (cur_pts_2d[:, 0] >= 0)
                & (cur_pts_2d[:, 1] < 1236)
                & (cur_pts_2d[:, 1] >= 0)
        )
        fov_inds = (fov_inds & (cur_pts_3d[:, 0] > 2.0)).unsqueeze(1)
        pts_2d_rect = fov_inds.long() * cur_pts_2d  # num_points, 2
        pts_2d_l0 = torch.ceil(torch.div(pts_2d_rect, 4.)).long()
        pts_2d_l1 = torch.ceil(torch.div(pts_2d_rect, 8.)).long()
        pts_2d_l2 = torch.ceil(torch.div(pts_2d_rect, 16.)).long()
        pts_2d_l3 = torch.ceil(torch.div(pts_2d_rect, 32.)).long()

        cur_sampled_feat[0, ...] = fpn_feat['0'][i][pts_2d_l0[:, 0], pts_2d_l0[:, 1], :]
        cur_sampled_feat[1, ...] = fpn_feat['0'][i][pts_2d_l0[:, 0] + 1, pts_2d_l0[:, 1], :]
        cur_sampled_feat[2, ...] = fpn_feat['0'][i][pts_2d_l0[:, 0] + 1, pts_2d_l0[:, 1] + 1, :]
        cur_sampled_feat[3, ...] = fpn_feat['0'][i][pts_2d_l0[:, 0] + 1, pts_2d_l0[:, 1] - 1, :]
        cur_sampled_feat[4, ...] = fpn_feat['0'][i][pts_2d_l0[:, 0], pts_2d_l0[:, 1] + 1, :]
        cur_sampled_feat[5, ...] = fpn_feat['0'][i][pts_2d_l0[:, 0], pts_2d_l0[:, 1] - 1, :]
        cur_sampled_feat[6, ...] = fpn_feat['0'][i][pts_2d_l0[:, 0] - 1, pts_2d_l0[:, 1], :]
        cur_sampled_feat[7, ...] = fpn_feat['0'][i][pts_2d_l0[:, 0] - 1, pts_2d_l0[:, 1] + 1, :]
        cur_sampled_feat[8, ...] = fpn_feat['0'][i][pts_2d_l0[:, 0] - 1, pts_2d_l0[:, 1] - 1, :]

        cur_sampled_feat[9, ...] = fpn_feat['1'][i][pts_2d_l1[:, 0], pts_2d_l1[:, 1], :]
        cur_sampled_feat[10, ...] = fpn_feat['1'][i][pts_2d_l1[:, 0] + 1, pts_2d_l1[:, 1], :]
        cur_sampled_feat[11, ...] = fpn_feat['1'][i][pts_2d_l1[:, 0] + 1, pts_2d_l1[:, 1] + 1, :]
        cur_sampled_feat[12, ...] = fpn_feat['1'][i][pts_2d_l1[:, 0] + 1, pts_2d_l1[:, 1] - 1, :]
        cur_sampled_feat[13, ...] = fpn_feat['1'][i][pts_2d_l1[:, 0], pts_2d_l1[:, 1] + 1, :]
        cur_sampled_feat[14, ...] = fpn_feat['1'][i][pts_2d_l1[:, 0], pts_2d_l1[:, 1] - 1, :]
        cur_sampled_feat[15, ...] = fpn_feat['1'][i][pts_2d_l1[:, 0] - 1, pts_2d_l1[:, 1], :]
        cur_sampled_feat[16, ...] = fpn_feat['1'][i][pts_2d_l1[:, 0] - 1, pts_2d_l1[:, 1] + 1, :]
        cur_sampled_feat[17, ...] = fpn_feat['1'][i][pts_2d_l1[:, 0] - 1, pts_2d_l1[:, 1] - 1, :]

        cur_sampled_feat[18, ...] = fpn_feat['2'][i][pts_2d_l2[:, 0], pts_2d_l2[:, 1]]
        cur_sampled_feat[19, ...] = fpn_feat['3'][i][pts_2d_l3[:, 0], pts_2d_l3[:, 1]]

        cur_img_feat = trans(pc_feat[i].unsqueeze(0), cur_sampled_feat)
        cur_img_feat = fov_inds.long() * cur_img_feat
        fused_feat[i] = torch.cat([pc_feat[i], cur_img_feat], dim=-1)
    return fused_feat


# q = torch.randn(1, 200, 256)
# kv = torch.randn(36, 200, 256)
# Trans = TransformerEncoder(EncoderLayer(256, 4), 4)
# weighted_v = Trans(q, kv)
# print(weighted_v.size())
BS = 4
device = 'cuda'
img_backbone = resnet_fpn_backbone('resnet101', pretrained=True, norm_layer=None, trainable_layers=5)
img_backbone = img_backbone.to(device)
# img_backbone.eval()
img = torch.randn(BS, 3, 312, 1236).to(device)
feat = img_backbone(img)
feat = {key: nn.ConstantPad2d(1, 0)(val).permute(0, 2, 3, 1) for key, val in feat.items()}
# for val in feat.values():
#     val = nn.ConstantPad2d(1, 0)(val)
pc_feat = torch.randn(BS, 270, 256).to(device)
pts_3d = torch.randint(-100, 500, (BS, 270, 3)).to(device)
trans = TransformerEncoder(EncoderLayer(256, 4), 4).to(device)
calib = torch.randn(BS, 2, 3, 4).to(device)
fused_feat = fuse_feat(feat, pts_3d, pc_feat, trans, calib)
print(fused_feat)
