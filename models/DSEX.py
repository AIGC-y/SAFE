#detials expend model
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import *
from transformers import CLIPProcessor, CLIPModel


class DSEX(nn.Module):
    def __init__(self):
        super(DSEX, self).__init__()
        #伪影用resnet,其他用其他表征吗?这个可以慢慢式,先把整体结构写出来.
        #用四个还是三个也没想好
        # self.resnet = resnet50(num_classes=512)
        self.resnet1 = resnet50(num_classes=256)
        self.resnet2 = resnet50(num_classes=256)
        self.resnet3 = resnet50(num_classes=256)
        self.mlp1 = nn.Linear(768, 768)

        self.clipvit = CLIPModel.from_pretrained("/home/yiruolei/ALLMODEL/openai/clip-vit-large-patch14").vision_model
        for param in self.clipvit.parameters():
            param.requires_grad = False
        self.mlp2 = nn.Linear(1024, 768)
        # self.resnet4 = resnet50(num_classes=512)
        self.fc = nn.Linear(768*2, 2)
        # self.re = nn.Linear(input_size, 2)

    # def reweight(self,refactor,x,y):#这个结构能否根据refactor来增加比例呢?
    #     matrix = self.re(refactor)
    #     x = x * matrix
    #     y = y * (1-matrix)
    #     return x + y
        
    def forward(self, x):
        x1,x2,x3,x4 = x[0],x[1],x[2],x[3]
        # print("x1:",x1.shape,"x2:",x2.shape,"x3:",x3.shape)#*拼接图象消除了对象的影响,但是后面也没获得什么效果
        
        #?这里目前是设置的不同部分分开,后期可以采用不同的分开方式
        #*分开通道,空间,分开操作方式,分开幅度和相位,而其他组合来用.
        # x = self.resnet(x)
        x1 = self.resnet1(x1)
        x2 = self.resnet2(x2)
        x3 = self.resnet3(x3)
        x_res = torch.cat((x1, x2, x3), dim=1) #[B,256*3]
        x_res = self.mlp1(x_res) #[B,768]

        x_llm = self.clipvit(x4).pooler_output #[B,1024]
        x_llm = self.mlp2(x_llm)
        # print(x4.shape)

        

        #重加权方法再议,太多都能尝试了.
        # x_reweight1 = self.reweight(img_ori,low_freq,high_freq)#加强语义
        # x_reweight2 = self.reweight(img_patch,low_freq,high_freq)
        # x3 = self.resnet3(x_reweight1)
        # x4 = self.resnet4(x_reweight2)

        ax = torch.cat((x_res, x_llm), dim=1) #[B,768*2]
        a_x = self.fc(ax)
        # print(ax.shape)
        
        return a_x, x1,x2,x3,x4



