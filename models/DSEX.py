#detials expend model
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import *


class DSEX(nn.Module):
    def __init__(self,input_size=128):
        super(DSEX, self).__init__()
        #伪影用resnet,其他用其他表征吗?这个可以慢慢式,先把整体结构写出来.
        #用四个还是三个也没想好
        # self.resnet = resnet50(num_classes=512)
        self.resnet1 = resnet50(num_classes=256)
        self.resnet2 = resnet50(num_classes=256)
        self.resnet3 = resnet50(num_classes=256)
        # self.resnet4 = resnet50(num_classes=512)
        self.fc = nn.Linear(768, 2)
        # self.re = nn.Linear(input_size, 2)

    # def reweight(self,refactor,x,y):#这个结构能否根据refactor来增加比例呢?
    #     matrix = self.re(refactor)
    #     x = x * matrix
    #     y = y * (1-matrix)
    #     return x + y
        
    def forward(self, x):
        x1,x2,x3 = x[0],x[1],x[2]
        # print("x1:",x1.shape,"x2:",x2.shape,"x3:",x3.shape)#*拼接图象消除了对象的影响,但是后面也没获得什么效果
        
        #?这里目前是设置的不同部分分开,后期可以采用不同的分开方式
        #*分开通道,空间,分开操作方式,分开幅度和相位,而其他组合来用.
        # x = self.resnet(x)
        x1 = self.resnet1(x1)
        x2 = self.resnet2(x2)
        x3 = self.resnet3(x3)

        #重加权方法再议,太多都能尝试了.
        # x_reweight1 = self.reweight(img_ori,low_freq,high_freq)#加强语义
        # x_reweight2 = self.reweight(img_patch,low_freq,high_freq)
        # x3 = self.resnet3(x_reweight1)
        # x4 = self.resnet4(x_reweight2)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.fc(x)

        return x,x1,x2,x3



