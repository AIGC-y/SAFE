#detials expend model
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import *
from transformers import CLIPProcessor, CLIPModel


class DSEX(nn.Module):
    def __init__(self):
        super(DSEX, self).__init__()
        ####resnet####
        # self.resnet0 = resnet50(num_classes=2)

        # self.resnet1 = resnet50(num_classes=512)
        # self.resnet2 = resnet50(num_classes=512)
        # self.mlp1 = nn.Linear(512*2, 512)
        # self.fc = nn.Linear(512, 2)



        ######vit#####
        self.clipvit = CLIPModel.from_pretrained("/home/yiruolei/ALLMODEL/openai/clip-vit-large-patch14").vision_model
        for param in self.clipvit.parameters():
            param.requires_grad = False

        self.mlp1 = nn.Linear(1024, 512)
        # # self.mlp2 = nn.Linear(1024, 512)
        # # self.mlp3 = nn.Linear(1024, 512)
        # # self.fc = nn.Linear(512*2, 2)
        self.fc1 = nn.Linear(512,2)
        # self.fc2 = nn.Linear(768, 2)
        # self.fc3 = nn.Linear(256, 2)

        #为了处理最后分类器的变化:不是直接使用归一化?
        self.s = 30
        self.m = 0.3
        self.weight = nn.Parameter(torch.FloatTensor(2,512))#[out_features, in_features] #一般需要使用正则化的时候都采用自己设置参数的形式
        nn.init.xavier_uniform_(self.weight)


    # def reweight(self,refactor,x,y):#这个结构能否根据refactor来增加比例呢?
    #     matrix = self.re(refactor)
    #     x = x * matrix
    #     y = y * (1-matrix)
    #     return x + y
        
    def forward(self, x, y):
        x1,x2,x3,x4 = x[0],x[1],x[2],x[3] #[B,3,224,224]
        # print("x1:",x1.shape,"x2:",x2.shape,"x3:",x3.shape)#*拼接图象消除了对象的影响,但是后面也没获得什么效果
        
        #resnet
        ##基础测试
        # feat = self.resnet1(x1)
        # output = self.fc(feat)
        # x3 = self.resnet2(x3)
        # x_res = torch.cat((x1, x3), dim=1) #[B,512*2]
        # feat = self.mlp1(x_res) #[B,512]
      
        #vit
        x_llm1 = self.clipvit(x1).pooler_output #[B,1024] #*vit的图象embeding方式开始也是他妈conv?
        feat = self.mlp1(x_llm1)
        output = self.fc1(feat)
        # x_llm2 = self.clipvit(x4).pooler_output #[B,1024] #*vit的图象embeding方式开始也是他妈conv?
        # x_llm2 = self.mlp2(x_llm2)#[B,512]
        # a_x = torch.cat((x_llm1, x_llm2), dim=1)
        # a_x = self.mlp3(a_x)#[B,512]
        
        # output = self.fc(feat)#*分类器优化
        #  # --------------------------- cos(theta) & phi(theta) ---------------------------
        # cosine = F.linear(F.normalize(feat), F.normalize(self.weight))
        # phi = cosine - self.m
        # # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), device='cuda')
        # # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        # one_hot.scatter_(1, y.view(-1, 1).long(), 1)
        # # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # # you can use torch.where if your torch.__version__ is 0.4
        # output *= self.s #获得了每个类别的最终分数?这个还没经过softmax呢??
        #?动态调整m:
        # m_schedule = [1.0, 1.5, 2.0]  # 在不同epoch调整m
        # for epoch in range(num_epochs):
            # criterion.m = m_schedule[min(epoch // 10, len(m_schedule)-1)]


        

        #重加权方法再议,太多都能尝试了.
        # x_reweight1 = self.reweight(img_ori,low_freq,high_freq)#加强语义
        # x_reweight2 = self.reweight(img_patch,low_freq,high_freq)
        # x3 = self.resnet3(x_reweight1)
        # x4 = self.resnet4(x_reweight2)

        # ax = torch.cat((x_res, x_llm), dim=1) #[B,768*2]
        # a_x = self.fc(ax)
        # print(ax.shape)#
        #*直接
        return output, feat



