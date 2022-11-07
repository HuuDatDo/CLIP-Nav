import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.upsampling import Upsample

from utils.dict_tools import objectview
from learning.inputs.partial_2d_distribution import Partial2DDistribution

import math
from mmcv.cnn import build_norm_layer

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM,self).__init__()
        self.lstm = nn.LSTM(input_size = 1024, hidden_size = 50, num_layers = 1, dropout = 0.1, bidirectional=False)
# INSTRUCTIONS[0:1]: torch.Size([1, 5])
# SENT_EMBEDDINGS: torch.Size([1, 50])
# INSTR_LENGTH: tensor([4])
    def forward(self, sequence_output_t):
        # print(f"SEQUENCE_OUTPUT_T: {sequence_output_t.size()}")
        outputs, states = self.lstm(sequence_output_t[0].unsqueeze(1))
        # print(f"OUTPUT: {outputs.size()}")
        sentence_embedding = outputs[-1] #.squeeze()
        # print(f"SENTENCE_EMBEDDING: {sentence_embedding.size()}")
        return sentence_embedding


class Conv_MLA(nn.Module):  #in channels was 1024
    def __init__(self, in_channels = 48, mla_channels = 256,norm_cfg = None):
        super(Conv_MLA,self).__init__()
        norm_cfg = dict(type='BN2d', requires_grad=True)
        self.mla_p2_1x1 = nn.Sequential(nn.Conv2d(
            in_channels, mla_channels, 1, bias= False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p3_1x1 = nn.Sequential(nn.Conv2d(
            in_channels, mla_channels, 1, bias= False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p4_1x1 = nn.Sequential(nn.Conv2d(
            in_channels, mla_channels, 1, bias= False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p5_1x1 = nn.Sequential(nn.Conv2d(
            in_channels, mla_channels, 1, bias= False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        
        self.mla_p2 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
                                    build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p3 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
                                    build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p4 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
                                    build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p5 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
                                    build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())

    def to_2D(self,x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1,2).reshape(n, c, h, w)
        return x
    
    def forward(self, res2, res3, res4, res5):
        res2 = self.to_2D(res2)
        res3 = self.to_2D(res3)
        res4 = self.to_2D(res4)
        res5 = self.to_2D(res5)


        #print(f"RES5: {res5.size()}")
        mla_p5_1x1 = self.mla_p5_1x1(res5)
        mla_p4_1x1 = self.mla_p4_1x1(res4)
        mla_p3_1x1 = self.mla_p3_1x1(res3)
        mla_p2_1x1 = self.mla_p2_1x1(res2)

        mla_p4_plus = mla_p5_1x1 + mla_p4_1x1
        mla_p3_plus = mla_p4_plus + mla_p3_1x1
        mla_p2_plus = mla_p3_plus + mla_p2_1x1

        mla_p5 = self.mla_p5(mla_p5_1x1)
        mla_p4 = self.mla_p4(mla_p4_plus)
        mla_p3 = self.mla_p3(mla_p3_plus)
        mla_p2 = self.mla_p2(mla_p2_plus)

        return mla_p2, mla_p3, mla_p4, mla_p5
    
    
class MLAHead(nn.Module):
    def __init__(self, mla_channels=256, mlahead_channels=128, norm_cfg=None):
        super(MLAHead, self).__init__()
        norm_cfg = dict(type='BN2d', requires_grad=True)
        self.head2 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
            1], nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())
        self.head3 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
            1], nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())
        self.head4 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
            1], nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())
        self.head5 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
            1], nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())

    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5):
        # head2 = self.head2(mla_p2)
        head2 = F.interpolate(self.head2(
            mla_p2), 4*mla_p2.shape[-1], mode='bilinear', align_corners=True)
        head3 = F.interpolate(self.head3(
            mla_p3), 4*mla_p3.shape[-1], mode='bilinear', align_corners=True)
        head4 = F.interpolate(self.head4(
            mla_p4), 4*mla_p4.shape[-1], mode='bilinear', align_corners=True)
        head5 = F.interpolate(self.head5(
            mla_p5), 4*mla_p5.shape[-1], mode='bilinear', align_corners=True)
        #print(f"PROCESSED_HEAD2: {head2.size()}")
        return torch.cat([head2, head3, head4, head5], dim=1)

class VIT_MLAHead(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size, out_channels, mla_channels=256, mlahead_channels=128,
                 norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(VIT_MLAHead, self).__init__(**kwargs)
        self.img_size = img_size
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.mla_channels = mla_channels
        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels

        self.mlahead = MLAHead(mla_channels=self.mla_channels,
                               mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        self.cls = nn.Conv2d(mla_channels*2,
                             self.out_channels, 3, padding=1)

    def forward(self, inputs):
        x = self.mlahead(inputs[0], inputs[1], inputs[2], inputs[3])
        #print(f"X: {x.size()}")
        x = self.cls(x)
        # x_for_outer = F.interpolate(x, size = 8, mode='bilinear',#[4, 2, 8, 8]
        #                 align_corners=None)
        # #print(f"X FOR OUTER: {x_for_outer.size()}")
        # x = F.interpolate(x, size=self.img_size, mode='bilinear',
        #                   align_corners=None)# Channels should be 2
        # return x, x_for_outer
        # if self.img_size == 2:
        #     return x
        x = F.interpolate(x, size= self.img_size, mode = 'bilinear', align_corners=None)
        return x
        
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

class Decoder(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.conv_mla = Conv_MLA()
        self.vit_mla2 = VIT_MLAHead(img_size=2, out_channels=48)
        self.vit_mla4 = VIT_MLAHead(img_size=4, out_channels=48)
        self.vit_mla8 = VIT_MLAHead(img_size=8, out_channels=48)
        self.vit_mla16 = VIT_MLAHead(img_size=16, out_channels=48)
        self.vit_mla32 = VIT_MLAHead(img_size=32, out_channels=48)
        self.lstm = LSTM()
        self.linear = nn.Linear(768, 192)
        
        self.p = objectview(params)
        self.emb_block_size = int(self.p.embedding_size / 5)
        
        if self.p.upscale_conv:
            if self.p.double_up:
                DeconvOp = UpscaleDoubleConv
            else:
                DeconvOp = UpscaleConv
        else:
            DeconvOp = DoubleDeconv
        
        self.deconv1 = DeconvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv2 = DeconvOp(self.p.hc1 + self.p.hb1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv3 = DeconvOp(self.p.hc1 + self.p.hb1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv4 = DeconvOp(self.p.hc1 + self.p.hb1, self.p.hc2, 3, stride=self.p.stride, padding=1)
        if self.p.upscale_conv:
            self.deconv5 = DeconvOp(self.p.hb1 + self.p.hc2, self.p.out_channels, 3, stride=self.p.stride, padding=1)
        else:
            self.deconv5 = nn.ConvTranspose2d(self.p.hb1 + self.p.hc2, self.p.out_channels, 3, stride=self.p.stride,
                                              padding=1)
        
        self.act = nn.LeakyReLU()
        self.convoob = DoubleConv(self.p.hb1 + self.p.hc2, 2, 3, stride=2, padding=1, stride2=2, cmid=16)


        self.dnorm2 = nn.InstanceNorm2d(48)
        self.dnorm3 = nn.InstanceNorm2d(48)
        self.dnorm4 = nn.InstanceNorm2d(48)
        self.dnorm5 = nn.InstanceNorm2d(self.p.hc2)
        
        self.fnorm1 = nn.InstanceNorm2d(24)
        self.fnorm2 = nn.InstanceNorm2d(24)
        self.fnorm3 = nn.InstanceNorm2d(24)
        self.fnorm4 = nn.InstanceNorm2d(24)
        
        self.lang19 = nn.Linear(50,self.p.hc1 * self.p.hb1)
        self.lang28 = nn.Linear(50,self.p.hc1 * self.p.hb1)
        self.lang37 = nn.Linear(50,self.p.hc1 * self.p.hb1)
        self.lang46 = nn.Linear(50,self.p.hc1 * self.p.hb1)
        self.lang55 = nn.Linear(50,self.p.hc1 * self.p.hc1)
        
    def forward(self, sequence_output_v, sequence_output_t, tensor_store=None):
        batch_size = sequence_output_t.size(0)

        ###TODO: CHANGE THE INDEX
        # print(f"SEQUENCE_OUTPUT_V: {sequence_output_v.size()}")
        sequence_output_v = self.linear(sequence_output_v)
        res2 = sequence_output_v[:,1,:]
        res2 = res2.reshape(batch_size,4,48)
        res3 = sequence_output_v[:,2,:].reshape(batch_size,4,48)
        if sequence_output_v.size()[1] <3:
            res4 = sequence_output_v[:,2,:].reshape(batch_size,4,48)
            if sequence_output_v.size()[1] <4 and res5 :
                res5 = sequence_output_v[:,3,:].reshape(batch_size,4,48)
            else:
                res5 = sequence_output_v[:,4,:].reshape(batch_size,4,48)

        else:
            res4 = sequence_output_v[:,3,:].reshape(batch_size,4,48)
            res5 = res4
        # res4 = sequence_output_v[:,3,:].reshape(batch_size,4,48)
        
        mla_p2, mla_p3, mla_p4, mla_p5 = self.conv_mla(res2, res3, res4, res5)
        embedding = self.lstm(sequence_output_t)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        x1 = self.vit_mla32([mla_p2, mla_p3, mla_p4, mla_p5])
        x2 = self.vit_mla16([mla_p2, mla_p3, mla_p4, mla_p5])
        x3 = self.vit_mla8([mla_p2, mla_p3, mla_p4, mla_p5])
        x4 = self.vit_mla4([mla_p2, mla_p3, mla_p4, mla_p5])
        x5 = self.vit_mla2([mla_p2, mla_p3, mla_p4, mla_p5])

# X1: torch.Size([4, 48, 32, 32])
# X2: torch.Size([4, 48, 16, 16])
# X3: torch.Size([4, 48, 8, 8])
# X4: torch.Size([4, 48, 4, 4])
# X5: torch.Size([4, 48, 2, 2])

        # print(f"X1: {x1.size()}")
        # print(f"X2: {x2.size()}")
        # print(f"X3: {x3.size()}")
        # print(f"X4: {x4.size()}")
        # print(f"X5: {x5.size()}")

        if self.p.split_embedding:
            block_size = self.emb_block_size
            emb1 = embedding[:, 0*block_size:1*block_size]
            emb2 = embedding[:, 1*block_size:2*block_size]
            emb3 = embedding[:, 2*block_size:3*block_size]
            emb4 = embedding[:, 3*block_size:4*block_size]
            emb5 = embedding[:, 4*block_size:5*block_size]
        else:
            emb1 = emb2 = emb3 = emb4 = emb5 = embedding
        
        # print(f"EMB1: {emb1.size()}")
        # print(f"EMB2: {emb2.size()}")
        # print(f"EMB3: {emb3.size()}")
        x1f = Variable(torch.zeros_like(x1[:,0:self.p.hb1,:,:].data))
        x2f = Variable(torch.zeros_like(x2[:,0:self.p.hb1,:,:].data))
        x3f = Variable(torch.zeros_like(x3[:,0:self.p.hb1,:,:].data))
        x4f = Variable(torch.zeros_like(x4[:,0:self.p.hb1,:,:].data))
        x5f = Variable(torch.zeros_like(x5.data))
        # print(f"X1 before loop:{x1.size()}")
        for i in range(batch_size):
            # print(f"X1 in loop: {x1.size()}")
            emb_idx = i if embedding.shape[0] == batch_size else 0
            # print(f"WTF IS GOING ON: {self.lang55(emb5[emb_idx:emb_idx + 1]).size()}")
            # EMBEDDING_SIZE: torch.Size([1, 50])
            # WTF IS GOING ON: torch.Size([1, 1152])
            # EMBEDDING_SIZE: torch.Size([6, 16, 100])
            # WTF IS GOING ON: torch.Size([1, 16, 1152])
            lf1 = F.normalize(self.lang19(emb1[emb_idx:emb_idx + 1])).view([self.p.hb1, self.p.hc1, 1, 1])
            lf2 = F.normalize(self.lang28(emb2[emb_idx:emb_idx + 1])).view([self.p.hb1, self.p.hc1, 1, 1])
            lf3 = F.normalize(self.lang37(emb3[emb_idx:emb_idx + 1])).view([self.p.hb1, self.p.hc1, 1, 1])
            lf4 = F.normalize(self.lang46(emb4[emb_idx:emb_idx + 1])).view([self.p.hb1, self.p.hc1, 1, 1])
            lf5 = F.normalize(self.lang55(emb5[emb_idx:emb_idx + 1])).view([self.p.hc1, self.p.hc1, 1, 1])

            # print(f"LF1: {lf1.size()}")
            # print(f"LF2: {lf2.size()}")
            # print(f"LF3: {lf3.size()}")
            # print(f"LF4: {lf4.size()}")
            # print(f"LF5: {lf5.size()}")

            # LF1: torch.Size([24, 48, 1, 1])
            # LF2: torch.Size([24, 48, 1, 1])
            # LF3: torch.Size([24, 48, 1, 1])
            # LF4: torch.Size([24, 48, 1, 1])
            # LF5: torch.Size([48, 48, 1, 1])
            # X1:torch.Size([4, 48, 32, 32])
            # X1[i:i+1]: torch.Size([1, 48, 32, 32])
            # LF1:torch.Size([24, 48, 1, 1])

            #print(f"X1: {x1.size()}\nX1[i:i+1]: {x1[i:i+1].size()}\nLF1:{lf1.size()}\n I: {i}")
            x1f[i:i+1] = F.conv2d(x1[i:i+1], lf1)
            x2f[i:i+1] = F.conv2d(x2[i:i+1], lf2)
            x3f[i:i+1] = F.conv2d(x3[i:i+1], lf3)
            x4f[i:i+1] = F.conv2d(x4[i:i+1], lf4)
            x5f[i:i+1] = F.conv2d(x5[i:i+1], lf5)

        x1 = self.fnorm1(x1f)
        x2 = self.fnorm2(x2f)
        x3 = self.fnorm3(x3f)
        x4 = self.fnorm4(x4f)
        x5 = x5f
            
        if tensor_store is not None:
            tensor_store.keep_inputs("lingunet_g1", x1)
            tensor_store.keep_inputs("lingunet_g2", x2)
            tensor_store.keep_inputs("lingunet_g3", x3)
            tensor_store.keep_inputs("lingunet_g4", x4)
            tensor_store.keep_inputs("lingunet_g5", x5)
            
        x6 = self.act(self.deconv1(x5, output_size=x4.size()))
        x46 = torch.cat([x4, x6], 1)
        x7 = self.dnorm3(self.act(self.deconv2(x46, output_size=x3.size())))
        x37 = torch.cat([x3, x7], 1)
        x8 = self.dnorm4(self.act(self.deconv3(x37, output_size=x2.size())))
        x28 = torch.cat([x2, x8], 1)
        x9 = self.dnorm5(self.act(self.deconv4(x28, output_size=x1.size())))
        x19 = torch.cat([x1, x9], 1)
        inner_scores = self.deconv5(x19, output_size=[batch_size, 35, 64, 64])
        
        
        o = self.convoob(x19)
        outer_scores = F.avg_pool2d(o, o.shape[2]).view([batch_size, 2])

        # inner_scores.cuda(0)
        # outer_scores.cuda(0)

        # print(f"INNER_DEVICE: {inner_scores.device} \n OUTER_DEVICE: {outer_scores.device}")
        
        # both_dist_scores = Partial2DDistribution(inner_scores, outer_scores)
        
        return inner_scores, outer_scores #both_dist_scores