import torch
from torch.autograd import Variable
from torch import nn as nn
import torch.nn.functional as F
from torch.nn.modules.upsampling import Upsample
import clip
import numpy as np
from learning.modules.lseg.lseg_blocks import _make_encoder

from learning.inputs.partial_2d_distribution import Partial2DDistribution

from utils.dict_tools import objectview

class depthwise_clipseg_conv(nn.Module):
    def __init__(self):
        super(depthwise_clipseg_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    
    def depthwise_clipseg(self, x, channels):
        x = torch.cat([self.depthwise(x[:, i].unsqueeze(1)) for i in range(channels)], dim=1)
        return x

    def forward(self, x):
        channels = x.shape[1]
        out = self.depthwise_clipseg(x, channels)
        return out


class depthwise_conv(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # support for 4D tensor with NCHW
        C, H, W = x.shape[1:]
        x = x.reshape(-1, 1, H, W)
        x = self.depthwise(x)
        x = x.view(-1, C, H, W)
        return x


class depthwise_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(depthwise_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x, act=True):
        x = self.depthwise(x)
        if act:
            x = self.activation(x)
        return x


class bottleneck_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(bottleneck_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()


    def forward(self, x, act=True):
        sum_layer = x.max(dim=1, keepdim=True)[0]
        x = self.depthwise(x)
        x = x + sum_layer
        if act:
            x = self.activation(x)
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
    def __init__(self, cin, cout, k, stride=1, padding=0):
        super(UpscaleConv, self).__init__()
        self.upsample1 = Upsample(scale_factor=2, mode="nearest")
        self.conv2 = nn.Conv2d(cin, cout, k, stride=1, padding=padding)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)

    def forward(self, img, output_size):
        x = F.leaky_relu(img)
        x = self.upsample1(x)
        x = self.conv2(x)
        return x

class CLIPUnet(torch.nn.Module):
    def __init__(self, params):
                 #in_channels, out_channels, embedding_size, hc1=32, hb1=16, hc2=256, stride=2, split_embedding=False):
        super(CLIPUnet, self).__init__()

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
            
        self.clip_pretrained, _ =  clip.load("ViT-B/32", device='cuda', jit=False)


        # inchannels, outchannels, kernel size
        self.conv1 = DoubleConv(self.p.in_channels, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv2 = DoubleConv(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv3 = DoubleConv(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv4 = DoubleConv(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv5 = DoubleConv(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)

        self.deconv1 = DeconvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv2 = DeconvOp(self.p.hc1 + self.p.hb1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv3 = DeconvOp(self.p.hc1 + self.p.hb1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv4 = DeconvOp(self.p.hc1 + self.p.hb1, self.p.hc2, 3, stride=self.p.stride, padding=1)
        if self.p.upscale_conv:
            self.deconv5 = DeconvOp(self.p.hb1 + self.p.hc2, self.p.out_channels, 3, stride=self.p.stride, padding=1)
        else:
            self.deconv5 = nn.ConvTranspose2d(self.p.hb1 + self.p.hc2, self.p.out_channels, 3, stride=self.p.stride,
                                              padding=1)

        self.convoob = DoubleConv(self.p.hb1 + self.p.hc2, 2, 3, stride=2, padding=1, stride2=2, cmid=16)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        self.out_c = 48
        self.text_linear1 = nn.Linear(512, self.out_c)
        self.text_linear2 = nn.Linear(48, 256)
        self.bottleneck1 = nn.Sequential(bottleneck_block(activation="relu"),
                                         bottleneck_block(activation="relu"))
        self.bottleneck2 = nn.Sequential(bottleneck_block(activation="relu"),
                                         bottleneck_block(activation="relu"))
        self.bottleneck3 = nn.Sequential(bottleneck_block(activation="relu"),
                                         bottleneck_block(activation="relu"))
        self.bottleneck4 = nn.Sequential(bottleneck_block(activation="relu"),
                                         bottleneck_block(activation="relu"))
        
        self.clipconv1 = nn.Conv2d(1, 232, 3, padding="same")
        self.clipconv2 = nn.Conv2d(1, 24, 3, padding="same")
        self.clipconv3 = nn.Conv2d(1, 24, 3, padding="same")
        self.clipconv4 = nn.Conv2d(1, 24, 3, padding="same")


        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.norm2 = nn.InstanceNorm2d(self.p.hc1)
        self.norm3 = nn.InstanceNorm2d(self.p.hc1)
        self.norm4 = nn.InstanceNorm2d(self.p.hc1)
        self.norm5 = nn.InstanceNorm2d(self.p.hc1)
        # self.dnorm1 = nn.InstanceNorm2d(in_channels * 4)
        self.dnorm2 = nn.InstanceNorm2d(self.p.hc1)
        self.dnorm3 = nn.InstanceNorm2d(self.p.hc1)
        self.dnorm4 = nn.InstanceNorm2d(self.p.hc1)
        self.dnorm5 = nn.InstanceNorm2d(self.p.hc2)

        self.fnorm1 = nn.InstanceNorm2d(self.p.hb1)
        self.fnorm2 = nn.InstanceNorm2d(self.p.hb1)
        self.fnorm3 = nn.InstanceNorm2d(self.p.hb1)
        self.fnorm4 = nn.InstanceNorm2d(self.p.hb1)

        self.lang19 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hb1)
        self.lang28 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hb1)
        self.lang37 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hb1)
        self.lang46 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hb1)
        self.lang55 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hc1)

    def init_weights(self):
        self.conv1.init_weights()
        self.conv2.init_weights()
        self.conv3.init_weights()
        self.conv4.init_weights()
        self.conv5.init_weights()
        self.deconv1.init_weights()
        self.deconv2.init_weights()
        self.deconv3.init_weights()
        self.deconv4.init_weights()
        #self.deconv5.init_weights()

    def compute_scores(self, image_features, text_features, out_c =None):
        if out_c == None:
            out_c = self.out_c
        
        imshape = image_features.shape
        image_features = image_features.permute(0,2,3,1).reshape(-1, out_c)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # text_features = text_features.repeat(256,1)
        logits_per_image = self.logit_scale * image_features @ text_features.t()

        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2)

        return out
    
    def forward(self, input, embedding, save_img, count, domain, tensor_store=None):
        text = clip.tokenize(embedding).cuda(input.device)
        self.logit_scale = self.logit_scale.to(input.device)
        text_features = self.clip_pretrained.encode_text(text)
        text_features = self.text_linear1(text_features)
        
        x1 = self.norm2(self.act(self.conv1(input)))
        x2 = self.norm3(self.act(self.conv2(x1)))
        x3 = self.norm4(self.act(self.conv3(x2)))
        x4 = self.norm5(self.act(self.conv4(x3)))
        x5 = self.act(self.conv5(x4))

        if tensor_store is not None:
            tensor_store.keep_inputs("lingunet_f1", x1)
            tensor_store.keep_inputs("lingunet_f2", x2)
            tensor_store.keep_inputs("lingunet_f3", x3)
            tensor_store.keep_inputs("lingunet_f4", x4)
            tensor_store.keep_inputs("lingunet_f5", x5)

        batch_size = input.shape[0]
        
        #TODO: WRONG DIMENSION
        x6 = self.act(self.deconv1(x5, output_size=x4.size()))
        x6 = self.compute_scores(x6, text_features)
        x6 = self.clipconv4(self.bottleneck4(x6))
        # x6 = self.act(self.deconv1(x5, output_size=x4.size()))
        x46 = torch.cat([x4, x6], 1)
        
        x7 = self.dnorm3(self.act(self.deconv2(x46, output_size=x3.size())))
        x7 = self.compute_scores(x7, text_features)
        x7 = self.clipconv3(self.bottleneck3(x7))
        x37 = torch.cat([x3, x7], 1)
        
        x8 = self.dnorm4(self.act(self.deconv3(x37, output_size=x2.size())))
        x8 = self.compute_scores(x8, text_features)
        x8 = self.clipconv2(self.bottleneck2(x8))
        x28 = torch.cat([x2, x8], 1)
        
        x9 = self.dnorm5(self.act(self.deconv4(x28, output_size=x1.size())))
        x9 = self.compute_scores(x9, self.text_linear2(text_features), 256)
        x9 = self.clipconv1(self.bottleneck1(x9))
        x19 = torch.cat([x1, x9], 1)
        
        # print("CHECK NAN", x19.isnan().any())
        inner_scores = self.deconv5(x19, output_size=input.size())

        # Predict probability masses / scores for the goal or trajectory traveling outside the observed part of the map
        o = self.convoob(x19)
        outer_scores = F.avg_pool2d(o, o.shape[2]).view([batch_size, 2])


# x19: torch.Size([6, 280, 32, 32])
# Inner_Scores torch.Size([6, 2, 64, 64])
# Outer_Scores torch.Size([6, 2]

        both_dist_scores = Partial2DDistribution(inner_scores, outer_scores)

        return both_dist_scores


class CLIPUnetv2(torch.nn.Module):
    def __init__(self, params):
                 #in_channels, out_channels, embedding_size, hc1=32, hb1=16, hc2=256, stride=2, split_embedding=False):
        super(CLIPUnetv2, self).__init__()

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
            
        self.clip_pretrained, _ =  clip.load("ViT-B/32", device='cuda', jit=False)


        # inchannels, outchannels, kernel size
        self.conv1 = DoubleConv(self.p.in_channels, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv2 = DoubleConv(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv3 = DoubleConv(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv4 = DoubleConv(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv5 = DoubleConv(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)

        self.deconv1 = DeconvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv2 = DeconvOp(self.p.hc1 + self.p.hb1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv3 = DeconvOp(self.p.hc1 + self.p.hb1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv4 = DeconvOp(self.p.hc1 + self.p.hb1, self.p.hc2, 3, stride=self.p.stride, padding=1)
        if self.p.upscale_conv:
            self.deconv5 = DeconvOp(self.p.hb1 + self.p.hc2, self.p.out_channels, 3, stride=self.p.stride, padding=1)
        else:
            self.deconv5 = nn.ConvTranspose2d(self.p.hb1 + self.p.hc2, self.p.out_channels, 3, stride=self.p.stride,
                                              padding=1)

        self.convoob = DoubleConv(self.p.hb1 + self.p.hc2, 2, 3, stride=2, padding=1, stride2=2, cmid=16)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        self.out_c = 48
        self.text_linear1 = nn.Linear(512, self.out_c)
        self.text_linear2 = nn.Linear(48, 256)
        self.bottleneck1 = nn.Sequential(bottleneck_block(activation="relu"),
                                         bottleneck_block(activation="relu"))
        self.bottleneck2 = nn.Sequential(bottleneck_block(activation="relu"),
                                         bottleneck_block(activation="relu"))
        self.bottleneck3 = nn.Sequential(bottleneck_block(activation="relu"),
                                         bottleneck_block(activation="relu"))
        self.bottleneck4 = nn.Sequential(bottleneck_block(activation="relu"),
                                         bottleneck_block(activation="relu"))
        
        self.clipconv1 = nn.Conv2d(1, 24, 3, padding="same")
        self.clipconv2 = nn.Conv2d(1, 24, 3, padding="same")
        self.clipconv3 = nn.Conv2d(1, 24, 3, padding="same")
        self.clipconv4 = nn.Conv2d(1, 24, 3, padding="same")


        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.norm2 = nn.InstanceNorm2d(self.p.hc1)
        self.norm3 = nn.InstanceNorm2d(self.p.hc1)
        self.norm4 = nn.InstanceNorm2d(self.p.hc1)
        self.norm5 = nn.InstanceNorm2d(self.p.hc1)
        # self.dnorm1 = nn.InstanceNorm2d(in_channels * 4)
        self.dnorm2 = nn.InstanceNorm2d(self.p.hc1)
        self.dnorm3 = nn.InstanceNorm2d(self.p.hc1)
        self.dnorm4 = nn.InstanceNorm2d(self.p.hc1)
        self.dnorm5 = nn.InstanceNorm2d(self.p.hc2)

        self.fnorm1 = nn.InstanceNorm2d(self.p.hb1)
        self.fnorm2 = nn.InstanceNorm2d(self.p.hb1)
        self.fnorm3 = nn.InstanceNorm2d(self.p.hb1)
        self.fnorm4 = nn.InstanceNorm2d(self.p.hb1)

        self.lang19 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hb1)
        self.lang28 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hb1)
        self.lang37 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hb1)
        self.lang46 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hb1)
        self.lang55 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hc1)

    def init_weights(self):
        self.conv1.init_weights()
        self.conv2.init_weights()
        self.conv3.init_weights()
        self.conv4.init_weights()
        self.conv5.init_weights()
        self.deconv1.init_weights()
        self.deconv2.init_weights()
        self.deconv3.init_weights()
        self.deconv4.init_weights()
        #self.deconv5.init_weights()

    def compute_scores(self, image_features, text_features, out_c =None):
        if out_c == None:
            out_c = self.out_c
        
        imshape = image_features.shape
        image_features = image_features.permute(0,2,3,1).reshape(-1, out_c)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # text_features = text_features.repeat(256,1)
        logits_per_image = self.logit_scale * image_features @ text_features.t()

        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2)

        return out
    
    def forward(self, input, embedding, save_img, count, domain, tensor_store=None):
        text = clip.tokenize(embedding).cuda(input.device)
        self.logit_scale = self.logit_scale.to(input.device)
        text_features = self.clip_pretrained.encode_text(text)
        text_features = self.text_linear1(text_features)
        
        x1 = self.norm2(self.act(self.conv1(input)))
        x2 = self.norm3(self.act(self.conv2(x1)))
        x3 = self.norm4(self.act(self.conv3(x2)))
        x4 = self.norm5(self.act(self.conv4(x3)))
        x5 = self.act(self.conv5(x4))

        if tensor_store is not None:
            tensor_store.keep_inputs("lingunet_f1", x1)
            tensor_store.keep_inputs("lingunet_f2", x2)
            tensor_store.keep_inputs("lingunet_f3", x3)
            tensor_store.keep_inputs("lingunet_f4", x4)
            tensor_store.keep_inputs("lingunet_f5", x5)

        batch_size = input.shape[0]
        
        #TODO: WRONG DIMENSION
        x6 = self.act(self.deconv1(x5, output_size=x4.size()))
        x4 = self.compute_scores(x4, text_features)
        x4 = self.clipconv4(self.bottleneck4(x4))
        # x6 = self.act(self.deconv1(x5, output_size=x4.size()))
        x46 = torch.cat([x4, x6], 1)
        
        x7 = self.dnorm3(self.act(self.deconv2(x46, output_size=x3.size())))
        x3 = self.compute_scores(x3, text_features)
        x3 = self.clipconv3(self.bottleneck3(x3))
        x37 = torch.cat([x3, x7], 1)
        
        x8 = self.dnorm4(self.act(self.deconv3(x37, output_size=x2.size())))
        x2 = self.compute_scores(x2, text_features)
        x2 = self.clipconv2(self.bottleneck2(x2))
        x28 = torch.cat([x2, x8], 1)
        
        x9 = self.dnorm5(self.act(self.deconv4(x28, output_size=x1.size())))
        print(x1.size())
        x1 = self.compute_scores(x1, text_features)
        x1 = self.clipconv1(self.bottleneck1(x1))
        x19 = torch.cat([x1, x9], 1)
        
        # print("CHECK NAN", x19.isnan().any())
        inner_scores = self.deconv5(x19, output_size=input.size())

        # Predict probability masses / scores for the goal or trajectory traveling outside the observed part of the map
        o = self.convoob(x19)
        outer_scores = F.avg_pool2d(o, o.shape[2]).view([batch_size, 2])


# x19: torch.Size([6, 280, 32, 32])
# Inner_Scores torch.Size([6, 2, 64, 64])
# Outer_Scores torch.Size([6, 2]

        both_dist_scores = Partial2DDistribution(inner_scores, outer_scores)

        return both_dist_scores
