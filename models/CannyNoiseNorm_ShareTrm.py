import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch
import numpy as np

import math
from einops import rearrange


class SRMLayer(nn.Module):
    def __init__(self):
        super(SRMLayer, self).__init__()
        q = [4.0, 12.0, 2.0]
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]
        filter1 = np.asarray(filter1, dtype=float) / q[0]
        filter2 = np.asarray(filter2, dtype=float) / q[1]
        filter3 = np.asarray(filter3, dtype=float) / q[2]
        filters = np.asarray(
            [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]])  # shape=(3,3,5,5)
        # filters = np.transpose(filters, (2,3,1,0)) #shape=(5,5,3,3)

        initializer_srm = torch.tensor(filters).float()
        self.srm = nn.Conv2d(3, 3, (5, 5), padding=2)
        self.srm.weight = torch.nn.Parameter(initializer_srm)
        self.downsample = nn.AdaptiveMaxPool2d(output_size=10)
        # print(self.srm.weight)

    def forward(self, x):
        output = self.srm(x)
        output = torch.max(output, 1)[0]
        output = self.downsample(output)
        output = torch.sigmoid(output)
        return output


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)



class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        print("noise  not add1 only div sum")
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.ln = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, canny, noise, mask=None):
        b, n, dim, h = *x.shape, self.heads
        x_norm = self.ln(x)
        # print (x.shape, x_norm.shape)
        qkv = self.to_qkv(x_norm)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)
        # print (q.shape)
        # q = q[:, :, 0, :].unsqueeze(2)
        # print(q.shape)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        
        attn_map = dots[:, :, 0, 1:]  # batch *4*49
        attn_map_norm = attn_map.clone().softmax(dim=-1)  # .detach()
        
        #add canny
        canny = rearrange(torch.squeeze(canny, dim=1),'b p1 p2-> b (p1 p2)')+1
        canny_sum = torch.sum(canny, 1, keepdim=True).repeat(1, canny.shape[-1])
        canny = torch.div(canny, canny_sum).unsqueeze(1)
        #print("canny",attn_map_norm.shape,canny.shape)
        attn_map_norm = canny + attn_map_norm
        
        #add noise
        noise = rearrange(noise, 'b p1 p2-> b (p1 p2)')
        noise_sum = torch.sum(noise, 1, keepdim=True).repeat(1, noise.shape[-1])
        noise = torch.div(noise, noise_sum).unsqueeze(1)
        #print (attn_map_norm.shape,noise.shape)
        attn_map_norm= noise + attn_map_norm

        dots[:, :, 0, 1:] = attn_map_norm
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        print(depth)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads),
                Residual(nn.LayerNorm(dim)),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))

    def forward(self, x, canny, noise, mask=None):

        for attn, ln, ff in self.layers:
            x = attn(x, canny, noise)
            x = ln(x)
            x = ff(x)

        return x


class ViT_pre_clstoken(nn.Module):
    def __init__(self, dim=512, depth=1, heads=4, mlp_dim=2048):
        super().__init__()

        # self.patch_size = patch_size

        self.transformer = Transformer(dim, depth, heads, mlp_dim)
        self.transform_conv = nn.Conv2d(128, 32, kernel_size=1, stride=1)
        self.transform_norm = nn.InstanceNorm2d(32)
        self.transform_conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        self.transform_norm2 = nn.InstanceNorm2d(128)
        self.to_cls_token = nn.Identity()

    def forward(self, features, cls_token, canny, noise, patch_size, mask=None):
        p = patch_size
        # print ("patch",p)
        if p == 4:
            features = self.transform_conv(features)
            features = self.transform_norm(features)
        elif p == 2:
            features = self.transform_conv2(features)
            features = self.transform_norm2(features)
        y = rearrange(features, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)  # 2*49*512

        cls_tokens = cls_token.unsqueeze(1)  # torch.max(y,1)[0].

        # print (cls_tokens.__class__)
        # print (cls_tokens.shape)#2*1*512
        # print(y.shape, cls_tokens.shape)
        x = torch.cat((cls_tokens, y), 1)
        x = self.transformer(x, canny, noise, mask)
        cls_token = self.to_cls_token(x[:, 0])

        return cls_token


class Clstoken_from_featues(nn.Module):

    def __init__(self, channel):
        super(Clstoken_from_featues, self).__init__()
        max_feature = (256 // channel)
        if channel == 128:
            self.nn = nn.Sequential(
                nn.AdaptiveMaxPool2d(output_size=max_feature),
                nn.Flatten(),

            )
        else:
            self.nn = nn.Sequential(
                nn.AdaptiveMaxPool2d(output_size=max_feature),
                nn.Flatten(),
                nn.Linear(max_feature ** 2 * channel, 512)
            )

    def forward(self, x):
        x = self.nn(x)
        # print (x.shape)#batch*dim
        return x


class CannyNoiseNorm_ShareTrm(nn.Module):  # 224x224x3

    def __init__(self, pretrained=True, num_classes=2):
        super(CannyNoiseNorm_ShareTrm, self).__init__()
        # model = models.resnet34(pretrained)
        model = torch.hub.load('cfzd/FcaNet', 'fca34', pretrained=False)
        print("  div sum norm noise and canny added share trm layer1 feature map no back to backbone")
        self.pre = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.C2 = model.layer1
        self.C3 = model.layer2
        self.C4 = model.layer3
        self.C5 = model.layer4

        # self.transformer_C3 = ViT_pre_clstoken(patch_size=4, dim=2048)  # numpatch=49
        # self.transformer_C4 = ViT_pre_clstoken(patch_size=2, dim=1024)  # numpatch=49
        # self.transformer_C5 = ViT_pre_clstoken(patch_size=1, dim=512)  # numpatch=49
        self.transformer = ViT_pre_clstoken(dim=512)  # numpatch=49
        self.get_clstoken2 = Clstoken_from_featues(channel=64)
        self.get_clstoken3 = Clstoken_from_featues(channel=128)
        self.get_clstoken4 = Clstoken_from_featues(channel=256)
        self.downsample = nn.AdaptiveAvgPool2d(output_size=10)
        self.srm = SRMLayer()


        self.classifier = nn.Linear(512, 2)

    def forward(self, x, canny):  # 224x224x3
        canny = self.downsample(canny)
        noise = self.srm(x)
        # print(noise.shape, canny.shape)
        x = self.pre(x)  # 56x56x64
        feature2 = self.C2(x)  # 64*56*56

        clstoken2 = self.get_clstoken2(feature2)
        feature3 = self.C3(feature2)

        canny = self.downsample(canny)
        cls_embding_1 = self.transformer(feature3, clstoken2, canny, noise, 4)

        clstoken3 = self.get_clstoken3(feature3)
        feature4 = self.C4(feature3)
        cls_embding_2 = self.transformer(feature4, clstoken3, canny, noise, 2)

        clstoken4 = self.get_clstoken4(feature4)
        feature5 = self.C5(feature4)
        cls_embding_3 = self.transformer(feature5, clstoken4, canny, noise, 1)

        # cls_embding_1 = self.embedding1_512(cls_embding_1)
        # cls_embding_2 = self.embedding2_512(cls_embding_2)
        # cls_embding_3 = self.embedding3_512(cls_embding_3)

        output = cls_embding_1 + cls_embding_2 + cls_embding_3
        cls = self.classifier(output)
        #print (cls)
        return cls

#
# from torchsummary import summary
# model = CannyNoiseNorm_ShareTrm()#ResVit_SidewayFusion() #ResVit()##
# summary(model,input_size=([(3, 320, 320),(1,320,320)]))