import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import clip
# import cliport.utils.utils as utils
# from cliport.cliport.models.resnet import IdentityBlock, ConvBlock
from learning.modules.clipunet.unet import Up
# from learning.modules.clipunet.clip import build_model, load_clip, tokenize
from learning.modules.lseg.lseg_blocks import FeatureFusionBlock, Interpolate, _make_encoder, FeatureFusionBlock_custom, forward_vit

from learning.modules.map_to_map.leaky_integrator_w import LeakyIntegratorGlobalMap
from learning.modules.img_to_map.clip_fpv_to_global_map import ClipFPVToGlobalMap
from learning.modules.clipunet import fusion
from learning.modules.clipunet.fusion import FusionConvLat
from learning.inputs.partial_2d_distribution import Partial2DDistribution

from parameters.parameter_server import get_current_parameters
import parameters.parameter_server as P


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

class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.
        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)

def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        activation=nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

# class depthwise_clipseg_conv(nn.Module):
#     def __init__(self):
#         super(depthwise_clipseg_conv, self).__init__()
#         self.depthwise = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    
#     def depthwise_clipseg(self, x, channels):
#         x = torch.cat([self.depthwise(x[:, i].unsqueeze(1)) for i in range(channels)], dim=1)
#         return x

#     def forward(self, x):
#         channels = x.shape[1]
#         out = self.depthwise_clipseg(x, channels)
#         return out


# class depthwise_conv(nn.Module):
#     def __init__(self, kernel_size=3, stride=1, padding=1):
#         super(depthwise_conv, self).__init__()
#         self.depthwise = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

#     def forward(self, x):
#         # support for 4D tensor with NCHW
#         C, H, W = x.shape[1:]
#         x = x.reshape(-1, 1, H, W)
#         x = self.depthwise(x)
#         x = x.view(-1, C, H, W)
#         return x


# class depthwise_block(nn.Module):
#     def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
#         super(depthwise_block, self).__init__()
#         self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
#         if activation == 'relu':
#             self.activation = nn.ReLU()
#         elif activation == 'lrelu':
#             self.activation = nn.LeakyReLU()
#         elif activation == 'tanh':
#             self.activation = nn.Tanh()

#     def forward(self, x, act=True):
#         x = self.depthwise(x)
#         if act:
#             x = self.activation(x)
#         return x


# class bottleneck_block(nn.Module):
#     def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
#         super(bottleneck_block, self).__init__()
#         self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
#         if activation == 'relu':
#             self.activation = nn.ReLU()
#         elif activation == 'lrelu':
#             self.activation = nn.LeakyReLU()
#         elif activation == 'tanh':
#             self.activation = nn.Tanh()


#     def forward(self, x, act=True):
#         sum_layer = x.max(dim=1, keepdim=True)[0]
#         x = self.depthwise(x)
#         x = x + sum_layer
#         if act:
#             x = self.activation(x)
#         return x
    

class CLIPLingUNet(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="clip_vitb32_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs,
    ):
        super(CLIPLingUNet, self).__init__()

        self.root_params = P.get_current_parameters()["ModelPVN"]
        self.params = self.root_params["Stage1"]
        self.lingunet_params = self.params["lingunet"]
        
        self.channels_last = channels_last

        hooks = {
            "clip_vitl16_384": [5, 11, 17, 23],
            "clipRN50x16_vitl16_384": [5, 11, 17, 23],
            "clip_vitb32_384": [2, 5, 8, 11],
            "clip_vitb32_rn50_384": [0, 1, 8, 11]
        }

        # Instantiate backbone and reassemble blocks
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )

        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # self.bert = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        self.linear1 = nn.Linear(512,256)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        if backbone in ["clipRN50x16_vitl16_384"]:
            self.out_c = 768
        else:
            self.out_c = 256 #512
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        #Change this to change the decoder architecture
        self.arch_option = 1#kwargs["arch_option"]
        if self.arch_option == 1:
            self.scratch.head_block = bottleneck_block(activation="relu")
            self.block_depth = 3#kwargs['block_depth']
        elif self.arch_option == 2:
            self.scratch.head_block = depthwise_block(activation="relu")
            self.block_depth = 3#kwargs['block_depth']
            
        self.bottleneck1 = nn.Sequential(bottleneck_block(activation="relu"),
                                         bottleneck_block(activation="relu"))
        self.bottleneck2 = nn.Sequential(bottleneck_block(activation="relu"),
                                         bottleneck_block(activation="relu"))
        self.bottleneck3 = nn.Sequential(bottleneck_block(activation="relu"),
                                         bottleneck_block(activation="relu"))
        self.bottleneck4 = nn.Sequential(bottleneck_block(activation="relu"),
                                         bottleneck_block(activation="relu"))
        
        self.conv1 = nn.Conv2d(1, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(1, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(1, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(1, 256, 3, padding=1)

        self.scratch.output_conv = head

        lingunet_params = self.params["lingunet"]
        # self.text = clip.tokenize(self.labels)
        # self.lingunet_convoob = DoubleConv(
        #     lingunet_params["hb1"] + lingunet_params["hc2"],
        #     lingunet_params["out_channels"],
        #     3, stride=2, padding=1, stride2=2, cmid=16)

        self.deconv5 = nn.ConvTranspose2d(lingunet_params["hb1"] + lingunet_params["hc2"],
                                        lingunet_params["out_channels"], 3, stride=lingunet_params["stride"],
                                        padding=1)

        self.conv2d_inner = nn.Conv2d(256,2, kernel_size = 3, padding = 1)
        self.conv2d_outer = nn.Conv2d(256,2, kernel_size = 3, padding = 1)

    def encode_image(self, img):
        with torch.no_grad():
            img_encoding, img_im = self.clip_rn50.visual.prepool_im(img)
        return img_encoding, img_im

    def encode_text(self, x):
        with torch.no_grad():
            tokens = clip.tokenize([x]).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens==0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask
    
    def compute_scores(self, image_features, text_features):
        imshape = image_features.shape

        image_features = image_features.permute(0,2,3,1).reshape(-1, self.out_c)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # text_features = text_features.repeat(256,1)
        logits_per_image = self.logit_scale * image_features @ text_features.t()

        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2)
                
        return out
    
    # def bottleneck_block(self,x):
    #     for _ in range(self.block_depth - 1):
    #         out = self.scratch.head_block(x)
    #             # print("decoder out {_}:", out)
    #     out = self.scratch.head_block(out, False)
    #     out = self.scratch.output_conv(out)
    #     return out
    
    def forward(self,x, text, cam_poses, images_rl, cam_poses_rl):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)
            
        x = x.cuda(3)
        images_rl = images_rl.cuda(3)
        batch_size = x.size(0)

        layer_1, layer_2, layer_3, layer_4, select_S_W, select_SW_M, SM_W = forward_vit(self.pretrained, x, cam_poses, images_rl, cam_poses_rl)
        

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        # print("layer_1 size", layer_1_rn.size())
        # print("layer_2 size", layer_2_rn.size())
        # print("layer_3 size", layer_3_rn.size())
        # print("layer_4 size", layer_4_rn.size())
        
        text = clip.tokenize(text).cuda(3)
        self.logit_scale = self.logit_scale.to(x.device)
        text_features = self.clip_pretrained.encode_text(text)

        #TODO:Add the text-multiplication and depthwise block
        path_4 = self.scratch.refinenet4(layer_4_rn)
        text_4 = self.linear1(text_features) #.view()
        out4 = self.compute_scores(path_4, text_4)
        out4 = self.bottleneck4(out4)
        out4 = self.conv4(out4)
        
        path_3 = self.scratch.refinenet3(out4, layer_3_rn)
        # text_3 = self.linear2(text_4).view()
        out3 = self.compute_scores(path_3, text_4)
        out3 = self.bottleneck3(out3)
        out3 = self.conv3(out3)
        
        path_2 = self.scratch.refinenet2(out3, layer_2_rn)
        # text_2 = self.linear3(text_4).view()
        out2 = self.compute_scores(path_2, text_4)
        out2 = self.bottleneck2(out2)
        out2 = self.conv2(out2)
        
        path_1 = self.scratch.refinenet1(out2, layer_1_rn)
        out = self.compute_scores(path_1, text_4)
        out = self.bottleneck1(out)
        out = self.conv1(out)
        out = self.scratch.head(self.scratch.head1(out))
        
        inner_scores = self.conv2d_inner(out)
        outer_scores = F.avg_pool2d(self.conv2d_outer(out), out.shape[2]).view([batch_size, 2])
        
        inner_scores, outer_scores = inner_scores.cuda(0), outer_scores.cuda(0)
        
        both_dist_scores = Partial2DDistribution(inner_scores, outer_scores)
            
        return both_dist_scores, select_S_W, select_SW_M, SM_W
    
    # def forward(self,x, text, cam_poses):
    #     if self.channels_last == True:
    #         x.contiguous(memory_format=torch.channels_last)
            
    #     x = x.cuda(1)
    #     batch_size = x.size(0)

    #     layer_1, layer_2, layer_3, layer_4, S_W, SW_M = forward_vit(self.pretrained, x, cam_poses)
        

    #     layer_1_rn = self.scratch.layer1_rn(layer_1)
    #     layer_2_rn = self.scratch.layer2_rn(layer_2)
    #     layer_3_rn = self.scratch.layer3_rn(layer_3)
    #     layer_4_rn = self.scratch.layer4_rn(layer_4)
        
    #     # print("layer_1 size", layer_1_rn.size())
    #     # print("layer_2 size", layer_2_rn.size())
    #     # print("layer_3 size", layer_3_rn.size())
    #     # print("layer_4 size", layer_4_rn.size())
        
    #     text = clip.tokenize(text).cuda(1)
    #     self.logit_scale = self.logit_scale.to(x.device)
    #     text_features = self.clip_pretrained.encode_text(text)

    #     #TODO:Add the text-multiplication and depthwise block
    #     path_4 = self.scratch.refinenet4(layer_4_rn)
    #     text_4 = self.linear1(text_features) #.view()
    #     out4 = self.compute_scores(path_4, text_4)
    #     out4 = self.bottleneck4(out4)
    #     # out4 = self.conv4(out4)
        
    #     path_3 = self.scratch.refinenet3(out4, layer_3_rn)
    #     # text_3 = self.linear2(text_4).view()
    #     out3 = self.compute_scores(path_3, text_4)
    #     out3 = self.bottleneck3(out3)
    #     # out3 = self.conv3(out3)
        
    #     path_2 = self.scratch.refinenet2(out3, layer_2_rn)
    #     # text_2 = self.linear3(text_4).view()
    #     out2 = self.compute_scores(path_2, text_4)
    #     out2 = self.bottleneck2(out2)
    #     # out2 = self.conv2(out2)
        
    #     path_1 = self.scratch.refinenet1(out2, layer_1_rn)
    #     out = self.compute_scores(path_1, text_4)
    #     out = self.bottleneck1(out)
    #     # out = self.conv1(out)
    #     out = self.scratch.head1(path_1)
        
    #     inner_scores = self.conv2d_inner(out)
    #     outer_scores = F.avg_pool2d(self.conv2d_outer(out), out.shape[2]).view([batch_size, 2])
        
    #     inner_scores, outer_scores = inner_scores.cuda(0), outer_scores.cuda(0)
        
    #     both_dist_scores = Partial2DDistribution(inner_scores, outer_scores)
            
    #     return both_dist_scores, S_W, SW_M