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
        self.resnet1 = resnet50(class_num=512)
        self.resnet2 = resnet50(class_num=512)
        self.resnet3 = resnet50(class_num=512)
        self.resnet4 = resnet50(class_num=512)
        self.fc = nn.Linear(2048, 2)
        self.re = nn.Linear(input_size, 2)

    def reweight(self,refactor,x,y):#这个结构能否根据refactor来增加比例呢?
        matrix = self.re(refactor)
        x = x * matrix
        y = y * (1-matrix)
        return x + y
        
    def forward(self, img_ori,img_patch,low_freq,high_freq):
        x1 = self.resnet1(img_ori)
        x2 = self.resnet2(img_patch)
        x_reweight1 = self.reweight(img_ori,low_freq,high_freq)#加强语义
        x_reweight2 = self.reweight(img_patch,low_freq,high_freq)
        x3 = self.resnet3(x_reweight1)
        x4 = self.resnet4(x_reweight2)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.fc(x)
        return x,x1,x2,x3,x4



