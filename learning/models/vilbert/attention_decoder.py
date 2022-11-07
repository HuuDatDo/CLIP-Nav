from tkinter import S
from numpy import inner
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.upsampling import Upsample

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from functools import partial
from mmcv.cnn import build_norm_layer
import math
import warnings
from utils.dict_tools import objectview


"""
Compute attention scores across pixels
"""

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM,self).__init__()
        self.lstm = nn.LSTM(input_size = 768, hidden_size = 512, num_layers = 1, dropout = 0.1, bidirectional=False)
    
    def forward(self, sequence_output_t):
        outputs, states = self.lstm(sequence_output_t.unsqueeze(1))
        sentence_embedding = outputs[-1]
        return sentence_embedding


class DoubleConv(torch.nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0, cmid=None, stride2=1):
        super(DoubleConv, self).__init__()
        if cmid is None:
            cmid = cin
        self.conv1 = nn.Conv2d(cin, cmid, k, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(cmid, cout, k, stride=stride2, padding=padding)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)

    def forward(self, img):
        x = self.conv1(img)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        return x

class DoubleDeconv(torch.nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0):
        super(DoubleDeconv, self).__init__()
        self.conv1 = nn.ConvTranspose2d(cin, cout, k, stride=1, padding=padding)
        self.conv2 = nn.ConvTranspose2d(cout, cout, k, stride=stride, padding=padding)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)

    def forward(self, img, output_size):
        # TODO: 2 is stride
        osize1 = [int(i/2) for i in output_size]
        x = self.conv1(img, output_size=osize1)
        x = F.leaky_relu(x)
        x = self.conv2(x, output_size=output_size)
        return x
    
class UpscaleDoubleConv(torch.nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0):
        super(UpscaleDoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(cin, cout, k, stride=1, padding=padding)
        #self.upsample1 = Upsample(scale_factor=2, mode="nearest")
        self.conv2 = nn.Conv2d(cout, cout, k, stride=1, padding=padding)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)

    def forward(self, img, output_size):
        x = self.conv1(img)
        x = F.leaky_relu(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        #x = self.upsample1(x)
        x = self.conv2(x)
        return x

class UpscaleConv(torch.nn.Module):
    def __init__(self, cin, cout, k, padding=0):
        super(UpscaleConv, self).__init__()
        self.upsample1 = Upsample(scale_factor=2, mode="nearest")
        self.conv2 = nn.Conv2d(cin, cout, k, stride=1, padding=padding)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)

    def forward(self, img):
        x = F.leaky_relu(img)
        x = self.upsample1(x)
        x = self.conv2(x)
        return x
    
    
class AttentionDecoder(nn.Module):
    def __init__(self, params, out_channel = 2, img_size=32, embed_dim=1024,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_cfg=None,
                 num_conv=2, upsampling_method='bilinear', num_upsampe_layer=4, conv3x3_conv1x1=True, **kwargs):
        super().__init__()
        #Init for computing self-attention scores
        self.num_attention_heads = 8
        self.attention_head_size = 128
        self.all_head_size = 1024
        self.v_hidden_size = 1024
        self.t_hidden_size = 768
        
        # self.resize16 = Resize((16,16), InterpolationMode.BILINEAR)
        # self.resize32 = Resize((32,32), InterpolationMode.BILINEAR)

        self.query = nn.Linear(512, self.all_head_size)
        self.key = nn.Linear(self.t_hidden_size, self.all_head_size)
        self.value = nn.Linear(self.t_hidden_size, self.all_head_size)
        
        self.query1 = nn.Linear(128, self.all_head_size)
        self.key1 = nn.Linear(self.t_hidden_size, self.all_head_size)
        self.value1 = nn.Linear(self.t_hidden_size, self.all_head_size)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 8*8, 512))
        self.pos_embedding1 = nn.Parameter(torch.randn(1, 16*16, 128))
        self.pos_embedding2 = nn.Parameter(torch.randn(1, 32*32, 48))
        self.softmax = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(p=0.)

        self.linear8 = nn.Linear(64,256)
        self.linear16 = nn.Linear(256, 1024)

        self.conv8 = nn.Conv2d(1024, 128, 3, 1, 1)
        self.conv16 = nn.Conv2d(1024, 48, 3, 1, 1)
        
        #Init for upsampling features
        self.img_size = img_size
        self.norm_cfg = dict(type='BN2d', requires_grad=True)
        self.num_conv = num_conv
        self.norm = norm_layer(embed_dim)
        self.upsampling_method = upsampling_method
        self.num_upsampe_layer = num_upsampe_layer
        self.align_corners = False
        self.conv3x3_conv1x1 = conv3x3_conv1x1
        self.in_index = -1
        out_channel = out_channel # self.num_classes
        
        
        if self.num_conv == 2:
            if self.conv3x3_conv1x1:
                self.conv_0 = nn.Conv2d(
                    512, 128, kernel_size=3, stride=1, padding=1)
            else:
                self.conv_0 = nn.Conv2d(512, 128, 3, 1, 1)
            self.conv_1 = nn.Conv2d(256, 48, 1, 1)
            _, self.syncbn_fc_0 = build_norm_layer(self.norm_cfg, 128)
            
        #Init for computing the kernel from the linguistic outputs
        self.lstm = LSTM()
        self.conv = nn.Conv2d(1024,512,3,1,1)

        self.p = objectview(params)
        if self.p.split_embedding:
            self.emb_block_size = int(self.p.embedding_size / 5)
        else:
            self.emb_block_size = self.p.embedding_size
        
        if self.p.upscale_conv:
            if self.p.double_up:
                DeconvOp = UpscaleDoubleConv
            else:
                DeconvOp = UpscaleConv
        else:
            DeconvOp = DoubleDeconv
        
        if self.p.upscale_conv:
            self.deconv5 = DeconvOp(96, self.p.out_channels, 3, stride=self.p.stride, padding=1)
        else:
            self.deconv5 = nn.ConvTranspose2d(96, self.p.out_channels, 3, stride=self.p.stride,
                                              padding=1)
            
        self.convoob = DoubleConv(96, 2, 3, stride=2, padding=1, stride2=2, cmid=16)

        self.conv128 = nn.Conv2d(256, 128, 3, 1, 1)

        self.lang37 = nn.Linear(512, 48*48)
        self.lang46 = nn.Linear(512, 128*128)
        self.lang55 = nn.Linear(512, 512*512)

        self.fnorm1 = nn.InstanceNorm2d(48)
        self.fnorm2 = nn.InstanceNorm2d(128)
        self.fnorm3 = nn.InstanceNorm2d(512)
        
    def transpose_for_scores(self,x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def compute_scores(self, images, t_sequence_output):
        b, c, h,w = images.size(0), images.size(1), images.size(2), images.size(3)
        images = images.reshape(b, c, h*w) #[B,C,8,8] -> [B,C,64]
        images = self.dropout(images)
        images = images.permute(0,2,1)

        if h == 8:
            images += self.pos_embedding
            q = self.query(images)
            k = self.key(t_sequence_output)
            v = self.value(t_sequence_output)

        if h == 16:
            # print(f"IMAGES: {images.size()}")
            images += self.pos_embedding1
            q = self.query1(images)
            k = self.key1(t_sequence_output)
            v = self.value1(t_sequence_output)
            
        else:
            assert "Wrong size for k,q,v"

        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)
        
        dots = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        dots = dots/math.sqrt(self.attention_head_size)
        
        attn = nn.Softmax(dim=-1)(dots)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, value_layer)
        out = out.permute(0, 2, 1, 3).contiguous()
        new_out_layer_shape = out.size()[:-2] + (self.all_head_size,)
        out = out.view(*new_out_layer_shape)
        
        # Out should be 16x16 or 32x32

        if h == 8:
            out = self.linear8(out.permute(0,2,1))
            out = out.reshape(b,1024,2*h,2*w)
            out = self.conv8(out)
        if h == 16:
            out = self.linear16(out.permute(0,2,1))
            out = out.reshape(b,1024, 2*h, 2*w)
            out = self.conv16(out)
        out = nn.Sigmoid()(out)
        return out
        
    def forward(self, v_sequence_output, t_sequence_output, map16, map32):
        """_summary_

        Args:
            v_sequence_output (_type_): Visual sequence output of the VilBert
            t_sequence_output (_type_): Linguistics sequence output of the VilBert
            map16 (_type_): 128x16x16 features map from the ResNet
            map32 (_type_): 48x32x32 features map from the ResNet

        Returns:
            _type_: return inner_scores and outer_scores for PartialDistribution
        """
        if self.upsampling_method == 'bilinear':
            if v_sequence_output.dim() == 3:
                n, hw, c = v_sequence_output.shape
                h = w = int(math.sqrt(hw))
                x = v_sequence_output.transpose(1, 2).reshape(n, c, h, w)
        # print(v_sequence_output.size()) 
        # print(t_sequence_output.size())
        batch_size = v_sequence_output.size(0)
        embedding = torch.zeros(size = (batch_size,512))
        for i in range(batch_size):
            embedding[i:i+1] = self.lstm(t_sequence_output[i])
        embedding = embedding.cuda(0)
        if self.p.split_embedding:
            block_size = self.emb_block_size
            emb1 = embedding[:, 0*block_size:1*block_size]
            emb2 = embedding[:, 1*block_size:2*block_size]
            emb3 = embedding[:, 2*block_size:3*block_size]
            # emb4 = t_sequence_output[:, 3*block_size:4*block_size]
            # emb5 = t_sequence_output[:, 4*block_size:5*block_size]
        else:
            emb1 = emb2 = emb3 = emb4 = emb5 = embedding
        
        x =self.conv(x)

        batch_size = v_sequence_output.size(0)
        x1f = Variable(torch.zeros_like(map32[:,0:48,:,:].data))
        x2f = Variable(torch.zeros_like(map16[:,0:128,:,:].data))
        x3f = Variable(torch.zeros_like(x.data))
        
        # map16 = self.resize16(map16)
        # map32 = self.resize32(map32)
        for i in range(batch_size):
            emb_idx = i if t_sequence_output.shape[0] == batch_size else 0
            lf3 = F.normalize(self.lang55(emb3[emb_idx:emb_idx + 1])).view([512, 512, 1, 1])
            x3f[i:i+1] = F.conv2d(x[i:i+1], lf3)
        
        #print(f"X BEFORE INSTACNCE NORM: {x3f.size()}")
        x3 = self.fnorm3(x3f)
        
        x = self.conv_0(x3.clone())
        x = self.syncbn_fc_0(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(
            x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
        scores2 = self.compute_scores(x3, t_sequence_output)
        #print(f"MAP16 BEFORE MUL: {map16.size()}")
        #print(f"SCORES2: {scores2.size()}")
        map16 = torch.mul(map16, scores2)
        #print(f"MAP16 AFTER MUL: {map16.size()}")
        
        for i in range(batch_size):
            emb_idx = i if t_sequence_output.shape[0] == batch_size else 0
            #print(self.lang46(emb2[emb_idx:emb_idx + 1]).size())
            lf2 = F.normalize(self.lang46(emb2[emb_idx:emb_idx + 1])).view([128, 128, 1, 1])
            x2f[i:i+1] = F.conv2d(map16[i:i+1], lf2)
        #print(f"X BEFORE INSTACNCE NORM: {x2f.size()}")
        x2 = self.fnorm2(x2f)
        #print(f"X2 AFTER LANGUAGE FILTER: {x2.size()}")
        #print(f"X BEFORE CONCATENATING: {x.size()}")
        x = torch.cat([x,x2], dim=1)
        
        x2 = self.conv128(x)
        x = self.conv_1(x)
        #print(f"X BEFORE INTERPOLATE 32: {x2.size()}")
        x = F.interpolate(
            x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)
        scores1 = self.compute_scores(x2, t_sequence_output)
        map32 = torch.mul(map32, scores1)
        for i in range(batch_size):
            emb_idx = i if embedding.shape[0] == batch_size else 0
            lf1 = F.normalize(self.lang37(emb1[emb_idx:emb_idx + 1])).view([48, 48, 1, 1])
            x1f[i:i+1] = F.conv2d(map32[i:i+1], lf1)
        #print(f"X BEFORE INSTACNCE NORM: {x2f.size()}")
        x1 = self.fnorm1(x1f)
        #print(f"X1 BEFORE CONCATENATING: {x1.size()}")
        #print(f"X BEFORE CONCATENING: {x.size()}")
        x = torch.cat([x,x1],dim=1)
        #print(f"X BEFORE COMPUTE SCORES: {x.size()}")
        
        inner_scores = self.deconv5(x, output_size=[batch_size, 35, 64, 64])
        o = self.convoob(x)
        outer_scores = F.avg_pool2d(o, o.shape[2]).view([batch_size, 2])
        return inner_scores, outer_scores