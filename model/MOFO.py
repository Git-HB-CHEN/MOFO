import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from einops.layers.torch import Rearrange

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)

def cotp3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                              padding=dilation, groups=groups, bias=True, dilation=dilation,output_padding=1)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        H = W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x


class CSWinBlock(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim // 2, resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        H = W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        xm = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(xm)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)

        return x, xm

class Merge_Embed(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        return x

class backbone_encoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=96, depth=[2,2,6,2], split_size = [3,5,7],
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, use_chk=False):
        super(backbone_encoder, self).__init__()
        self.use_chk = use_chk
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads = num_heads

        self.stage0_conv_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim //2 , kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(embed_dim //2, eps=1e-5),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),

            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(embed_dim // 2, eps=1e-5),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )

        self.merge0 = Merge_Embed(embed_dim // 2, embed_dim)

        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule
        self.stage1 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[0], reso=img_size // 4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])

        self.merge1 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        self.stage2 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[1], reso=img_size // 8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1]) + i], norm_layer=norm_layer)
                for i in range(depth[1])])

        self.merge2 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        temp_stage3 = []
        temp_stage3.extend(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[2], reso=img_size // 16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2]) + i], norm_layer=norm_layer)
                for i in range(depth[2])])

        self.stage3 = nn.ModuleList(temp_stage3)

        self.merge3 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        self.stage4 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[3], reso=img_size // 32, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:-1]) + i], norm_layer=norm_layer, last_stage=True)
                for i in range(depth[-1])])

        self.norm = norm_layer(curr_dim)
        # Classifier head
        self.head = nn.Linear(curr_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.head.weight, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        xms = []
        x = self.stage0_conv_embed(x)
        # print(x.shape)
        xms.append(x)
        x = self.merge0(x)
        # print(x.shape)

        for blk in self.stage1:
            x = blk(x)

        for pre, blocks in zip([self.merge1, self.merge2, self.merge3],
                               [self.stage2, self.stage3, self.stage4]):
            x, xm = pre(x)
            xms.append(xm)
            for blk in blocks:
                x = blk(x)

        B, new_HW, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        xms.append(x)
        return xms

class BkConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(BkConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.in1   = nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True, momentum=0.1)
        self.act1  = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

    def forward(self, x):
        return self.act1(self.in1(self.conv1(x)))

class UpTransition(nn.Module):
    def __init__(self, in_channels, depth):
        super(UpTransition, self).__init__()
        self.unit_channel=32
        self.depth = depth
        self.up_conv = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.layer_ops1 = BkConv(in_channels, self.unit_channel * (2 ** depth))
        self.layer_ops2 = BkConv(self.unit_channel * (2 ** depth), self.unit_channel * (2 ** depth))

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv,skip_x),1)
        feature = self.layer_ops1(concat)
        output = self.layer_ops2(feature)
        return output, feature


class backbone_decoder(nn.Module):
    def __init__(self):
        super(backbone_decoder, self).__init__()

        self.up_tr256 = UpTransition(512, 3)
        self.up_tr128 = UpTransition(256, 2)
        self.up_tr64 = UpTransition(128, 1)
        self.up_tr32 = UpTransition(64,  0)
        self.encod_head = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, X_feats):

        self.out_up_256, self.feat_tr256 = self.up_tr256(X_feats[-1], X_feats[-2])
        self.out_up_128, self.feat_tr128 = self.up_tr128(self.out_up_256, X_feats[-3])
        self.out_up_64, self.feat_tr64 = self.up_tr64(self.out_up_128, X_feats[-4])
        self.out_up_32, self.feat_tr32 = self.up_tr32(self.out_up_64, X_feats[-5])

        return self.encod_head(self.out_up_32), [X_feats[-1],self.feat_tr256, self.feat_tr128, self.feat_tr64]


class prompt_decoder(nn.Module):
    def __init__(self, input_channel, reduction_factor, reduction_channel=128):
        super(prompt_decoder, self).__init__()

        self.reduction_layer = nn.Sequential(
            nn.Conv2d(input_channel, reduction_channel, kernel_size=2**reduction_factor, stride=2**reduction_factor, padding=0),
            Rearrange('b c h w -> b (h w) c', h=7, w=7))

        self.cross_attn_multihead_1 = nn.MultiheadAttention(reduction_channel, 4, batch_first=True)
        self.cross_attn_norm_1 = nn.LayerNorm(reduction_channel)
        self.ffn_linear1_1 = nn.Linear(reduction_channel, reduction_channel * 4)
        self.ffn_linear2_1 = nn.Linear(reduction_channel * 4, reduction_channel)
        self.ffn_norm_1 = nn.LayerNorm(reduction_channel)
        self.ffn_act_1 = F.relu


    def forward(self, pixel_feat, task_feat):

        pixel_feat_0 = self.reduction_layer(pixel_feat)

        # clust_feat_0 = torch.bmm(pixel_feat_0, einops.rearrange(task_feat, 'b n d -> b d n'))
        # clust_feat_1 = torch.bmm(torch.softmax(clust_feat_0,1),task_feat)+pixel_feat_0
        pixel_feat_1 = pixel_feat_0
        task_feat_1 = task_feat

        task_feat_cross_1,_ = self.cross_attn_multihead_1(task_feat_1, pixel_feat_1, pixel_feat_1)
        task_feat_1 = task_feat_1 + task_feat_cross_1
        task_feat_1 = self.cross_attn_norm_1(task_feat_1)
        task_feat_ffn_1 = self.ffn_linear1_1(task_feat_1)
        task_feat_ffn_1 = self.ffn_act_1(task_feat_ffn_1)
        task_feat_ffn_1 = self.ffn_linear2_1(task_feat_ffn_1)
        task_feat_1 = task_feat_1 + task_feat_ffn_1
        task_feat_final = self.ffn_norm_1(task_feat_1)

        return task_feat_final

class MOFO(nn.Module):
    def __init__(self, class_num, task_prompt = 'rand_embedding', prompt_dim=256):
        super().__init__()
        self.class_num = class_num
        self.out_channels = class_num
        self.task_prompt = task_prompt
        self.prompt_dim = prompt_dim

        self._backbone_encoder = backbone_encoder(patch_size=4, embed_dim=64, depth=[2, 4, 32, 2], split_size=[1, 2, 7, 7], num_heads=[2, 4, 8, 16], mlp_ratio=4.)
        self._backbone_decoder = backbone_decoder()

        if self.task_prompt == 'rand_embedding':
            self.organ_embedding = nn.Embedding(self.class_num, self.prompt_dim)
        elif self.task_prompt == 'word_embedding':
            self.register_buffer('organ_embedding', torch.randn(self.class_num, 512))
            self.text_to_vision = nn.Linear(512, self.prompt_dim)
            self.pos_embed = nn.Parameter(torch.randn(self.class_num, self.prompt_dim) * .05)

        self._prompt_decoder_1 = prompt_decoder(input_channel=64,  reduction_factor=3, reduction_channel=self.prompt_dim)
        self._prompt_decoder_2 = prompt_decoder(input_channel=128, reduction_factor=2, reduction_channel=self.prompt_dim)
        self._prompt_decoder_3 = prompt_decoder(input_channel=256, reduction_factor=1, reduction_channel=self.prompt_dim)
        self._prompt_decoder_4 = prompt_decoder(input_channel=512, reduction_factor=0, reduction_channel=self.prompt_dim)

        self._classifier_norm = nn.LayerNorm(self.prompt_dim)
        self._classifier_head = nn.Sequential(nn.Linear(self.prompt_dim, self.prompt_dim // 4),
                                              nn.ReLU(True),
                                              nn.Linear(self.prompt_dim // 4, self.class_num))

        self.precls_conv = nn.Sequential(nn.GroupNorm(8, 32),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(32, 8, kernel_size=1))


        weight_nums, bias_nums = [], []
        weight_nums.append(8 * 8)
        weight_nums.append(8 * 8)
        weight_nums.append(8 * 1)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(1)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.controller = nn.Linear(self.prompt_dim, sum(weight_nums + bias_nums))

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 1)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            # print(i, x.shape, w.shape, b.shape)
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def forward(self, x):
        B = x.shape[0]
        H,W = x.shape[2],x.shape[3]

        self.x_feats = self._backbone_encoder(x)
        self.out_feats, self.scale_feats = self._backbone_decoder(self.x_feats)

        if self.task_prompt == 'rand_embedding':
            self.prompt_encoding_raw = self.organ_embedding.weight.unsqueeze(0).repeat(B,1,1)
        elif self.task_prompt == 'word_embedding':
            self.task_encoding = F.relu(self.text_to_vision(self.organ_embedding))
            self.task_encoding_pos_embed = self.task_encoding + self.pos_embed
            self.prompt_encoding_raw = self.task_encoding_pos_embed.unsqueeze(0).repeat(B,1,1)

        self.prompt_encoding = self._prompt_decoder_1(self.x_feats[1], self.prompt_encoding_raw)
        self.prompt_encoding = self._prompt_decoder_2(self.x_feats[2], self.prompt_encoding)
        self.prompt_encoding = self._prompt_decoder_3(self.x_feats[3], self.prompt_encoding)
        self.prompt_encoding = self._prompt_decoder_4(self.x_feats[4], self.prompt_encoding)

        self.classifier_feat = self._classifier_norm(self.prompt_encoding)
        self.classifier_feat = self.classifier_feat.mean(dim=1)
        self.classification_prob_maps = self._classifier_head(self.classifier_feat)

        params = self.controller(self.prompt_encoding)

        logits_array = []
        for i in range(B):
            head_inputs = self.precls_conv(self.out_feats[i].unsqueeze(0))
            head_inputs = head_inputs.repeat(self.class_num, 1, 1, 1)
            N, _, H, W = head_inputs.size()
            head_inputs = head_inputs.reshape(1, -1, H, W)
            weights, biases = self.parse_dynamic_params(params[i], 8, self.weight_nums, self.bias_nums)
            logits = self.heads_forward(head_inputs, weights, biases, N)
            logits_array.append(logits.reshape(1, -1, H, W))

        self.mask_prob_maps = torch.cat(logits_array, dim=0)

        return self.mask_prob_maps, self.classification_prob_maps

    def load_from(self, pretrained_path):
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            model_dict = self._backbone_encoder.state_dict()
            pretrained_dict = pretrained_dict['state_dict_ema']
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "stage" in k:
                    current_layer_num = k[5:6]
                    current_k = "stage_up" + str(current_layer_num) + k[6:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        # print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]
            msg = self._backbone_encoder.load_state_dict(full_dict, strict=False)
            print("---finish load pretrained weights for encoding module---")
            # print(msg)
        else:
            print("none pretrain")
