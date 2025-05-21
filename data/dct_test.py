
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import functional as F

import cv2
# from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import os


class DCTtrans():
    def __init__(self, image: Image.Image):
        """对于单张图片[C,W,H]\n
        这个和模块不一样,模块初始化都是为了设置结构参数,这里是为了能调用不同方法
        # tesnsor化的数值都是在[0.1]间的!,所以图象回复需要乘255或者其他方式
        """
        # 转换为灰度图像#**只用灰度图象吗?色彩不是也很重要吗?
        self.image = image
        if type(image) == torch.Tensor:
            image_tensor_rgb = image
        else:
            transform = transforms.ToTensor()
            image_tensor_rgb = transform(image)#tensor处理完就变成了[0,1]之间的数值.

        # if len(image_tensor_rgb.shape) ==3:
        #     image_tensor_rgb = image_tensor_rgb
            # .unsqueeze(0) ## 形状: [1, 3, H, W]
        # elif len(image_tensor_rgb.shape) == 4:
        #     pass
        # else: assert False, "image shape error"
        
        # self.B, 
        self.C, self.rows, self.cols = image_tensor_rgb.shape ## 形状: [B, C, H, W]

        # 灰度图象
        weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        self.gray_tensor = (image_tensor_rgb * weights).sum(dim=1, keepdim=True).squeeze(0) # 形状: [B, 1, H, W]
        # rgb各通道分离
        self.r_channel, self.g_channel, self.b_channel = image_tensor_rgb[ 0, :, :].unsqueeze(0), image_tensor_rgb[1, :, :].unsqueeze(0), image_tensor_rgb[2, :, :].unsqueeze(0)

        
    def dct(self,image: torch.Tensor) :#*还真是,必须把self写出来,即便没用
        # 对图像进行 DCT 变换
        f = torch.fft.fft2(image)
        fshift = torch.fft.fftshift(f) #中心化操作?貌似不一定,这里后续需要分离不同频带.
        #复数无法可视化,所以一般可视化需要分解两个
        #分离不同幅度和相位
        magnitude = 20 * torch.log(torch.abs(fshift) + 1e-10)#有增强和避免log(0)
        phase= torch.angle(fshift)
        return fshift, magnitude, phase
    
    def idct(self,fshift: torch.Tensor):
        """逆变换,获得tensor实数--区间(0,1).虚数取消"""
        # 对 DCT 变换后的频谱进行逆变换
        f_ishift = torch.fft.ifftshift(fshift)
        img_back = torch.fft.ifft2(f_ishift)
        img_back = torch.abs(img_back)#这里的非复数是没有意义的,就直接取消了就行
        return img_back
    
    def filter_type(self,fshift: torch.Tensor,type: str = 'circular'):
        """type: 'circular' or 'gaussian'
        只用带宽,相位这里没管
        """
          # 创建低通滤波器（理想圆形滤波器）
        crow, ccol = self.rows // 2, self.cols // 2  # 中心点
        mask_low = torch.zeros((self.rows, self.cols), dtype=torch.float32)
        y, x = torch.meshgrid(torch.arange(self.rows), torch.arange(self.cols), indexing='ij')#用于生成网格序号,x,y都是网格结构,然后数值是利用写行和列的数值(相当于那个序号)

        if type == 'circular': 
            radius = 30  # 低通滤波器半径（可调整）,这是高低通的比例
            dist = torch.sqrt((x - ccol)**2 + (y - crow)**2)#生成点到中心点的距离
            mask_low[dist <= radius] = 1.0  # 中心区域设为1(掩码保留位)
            mask_low = mask_low.unsqueeze(0) # 形状: [1, 1, H, W]
        elif type == 'gaussian':
            # 创建高斯低通滤波器,#?减少振铃效应?
            sigma = 30  # 高斯核标准差（可调整）
            mask_low = torch.exp(-((x - ccol)**2 + (y - crow)**2) / (2 * sigma**2))#(因为接近1而保留,更灵活?但是需要这个吗?)
            mask_low = mask_low / mask_low.max()  # 归一化
            mask_low = mask_low.unsqueeze(0) # 形状: [1, 1, H, W]
        # 创建高通滤波器（1 - 低通滤波器）
        mask_high = 1.0 - mask_low

        # 应用滤波器,乘法有广播broadcast效果#*但如果想要不同滤波器还需要再修改
        fshift_low = fshift * mask_low  # 低通滤波
        fshift_high = fshift * mask_high  # 高通滤波
        # print('fshift_low.shape',fshift_low.shape,'low_freq.shape',fshift_low.shape)
        
        return fshift_low, fshift_high
    

    def save_images(self,low_freq, high_freq, ori_image, output_dir):
        """
        保存低频、高频和灰度图像为图像文件。
        Args:tensor张量要先cpu化
            low_freq: 低频图像张量
            high_freq: 高频图像张量
            gray_image: 灰度图像张量
            output_dir: 保存路径
    """ 
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        def process(x,output_dir,info):
            x = x.squeeze().cpu().numpy()  # 取模保留实数,虚数删去 # 形状: [H, W]
            # x = (x - x.min()) / (x.max() - x.min()) * 255 # 归一化??
            img = Image.fromarray((x * 255).astype('uint8'))
            img.save(os.path.join(output_dir, f"{info}.jpg"))
            # print('save_done',)
            return x

        process(low_freq,output_dir, "low_freq")
        process(high_freq,output_dir, "high_freq")
        process(ori_image,output_dir, "ori_channel")
        print('save_done')

        

    def apply_dct(self,channel: str = 'gray'):
        """
        channel: 'r','g','b','rgb','gray'
        对图像进行 DCT 变换并分离低频和高频信息
        很好,基本维度都是图象的维度,这是最好可视化的.
        其次都需要恢复到原始图象上,或者利用降维看可视化.
        """
        def eachc(C):
            fshift, freq_mag,freq_phase = self.dct(C)
            # self.save_images(fshift, freq_mag,freq_phase, f"results/image0/{channel}")  # 保存图像
            ##分离不同频带
            # todo 1.分离频谱和相位谱;2.分离多种频带并且动态设置差异频谱.
            fshift_low, fshift_high = self.filter_type(fshift,type='gaussian')  # 低通和高通滤波器
            # print('fshift_low.shape',fshift_low.shape,'fshift_high.shape',fshift_high.shape)
            # self.save_images(fshift_low, fshift_high, C, f"results/image3/{channel}")  # 保存图像
            # return fshift_low, fshift_high

            # 计算逆 DCT 变换
            low_freq = self.idct(fshift_low)
            high_freq = self.idct(fshift_high)
            # print('low_freq.shape',low_freq.shape,'high_freq.shape',high_freq.shape)

            # self.save_images(low_freq, high_freq, C, f"results/image6/{channel}")  # 保存图像
    # 
            return low_freq, high_freq
    
        # 计算 DCT 变换
        if channel == 'gray':
            return eachc (self.gray_tensor)
        elif channel == 'r':
            return eachc ( self.r_channel)
        elif channel == 'g':
            return eachc (self.g_channel)
        elif channel == 'b':
            return eachc (self.b_channel)
        elif channel == 'rgb':
            r_low,r_high = eachc (self.r_channel)
            g_low, g_high = eachc (self.g_channel)
            b_low, b_high = eachc (self.b_channel)
            return torch.concat((r_low,g_low,b_low),dim=0), torch.concat((r_high,g_high,b_high),dim=0)
        else:
            raise ValueError("Invalid channel. Choose 'gray', 'rgb', 'r', 'g', or 'b'")
        
      

  
    
    # def __call__(self, channel: str = 'gray'):
       
     



if __name__ == "__main__":
    # 测试代码
    # image_path = "/home/yiruolei/ALLDATASET/AIGCDetect/Chameleon/test/0_real/1c4b521d-428d-4c91-bb17-2c1246ed94af.jpg"  # 真实人脸
    # image_path = '/home/yiruolei/ALLDATASET/AIGCDetect/Chameleon/test/0_real/0a5c98c5-1ecb-45b2-8c4f-4e1478eacfa9.jpg'

    image_path = "/home/yiruolei/ALLDATASET/AIGCDetect/Chameleon/test/1_fake/4db7bcee-07d3-4329-aea3-57d1dbc1c098.jpg"  # 假人脸
    # image_path = "/home/yiruolei/ALLDATASET/AIGCDetect/Chameleon/test/1_fake/0a6c3851-a11c-4d6d-b1f3-59ead005a642.jpg"  # 
    image = Image.open(image_path).convert('RGB')
    dct_transformer = DCTtrans(image)
    dct_transformer.apply_dct(channel='gray')
    dct_transformer.apply_dct(channel='r')
    dct_transformer.apply_dct(channel='g')
    dct_transformer.apply_dct(channel='b')

