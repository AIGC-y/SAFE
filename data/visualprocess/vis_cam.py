import torch
from pytorch_grad_cam import GradCAM, \
                            ScoreCAM, \
                            GradCAMPlusPlus, \
                            AblationCAM, \
                            XGradCAM, \
                            EigenCAM, \
                            EigenGradCAM, \
                            LayerCAM, \
                            FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

import cv2
import time


def reshape_transform(tensor, height=14, width=14):
    """在transformer中也使用"""
    # 去掉cls token
    result = tensor[:, 1:, :].reshape(tensor.size(0),
    height, width, tensor.size(2))

    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def visual(model:torch.nn.Module, original:torch.Tensor, Input:torch.Tensor,target_category:torch.Tensor, use_cuda:bool=True):

    # 创建 GradCAM 对象
    cam = GradCAM(model=model,
                target_layers=[model.resnet1.avgpool],
                # 这里的target_layer要看模型情况，
                # 比如还有可能是：target_layers = [model.blocks[-1].ffn.norm]
                use_cuda=use_cuda,
                )#用不用transform看模型也是:reshape_transform=reshape_transform
    
    # 计算 grad-cam
    # target_category = None # 可以指定一个类别，或者使用 None 表示最高概率的类别
    grayscale_cam = cam(input_tensor=Input, targets=target_category)
    grayscale_cam = grayscale_cam[0, :]

    # 将 grad-cam 的输出叠加到原始图像上
    visualization = show_cam_on_image(original, grayscale_cam)

    T = time.time()
    # 保存可视化结果
    cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
    cv2.imwrite(f'results/cam/{T}.jpg', visualization)