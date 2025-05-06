
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
        """只加载图象,具体如何使用不知道\n
        这个和模块不一样,模块初始化都是为了设置结构参数,这里是为了能调用不同方法
        # tesnsor化的数值都是在[0.1]间的!,所以图象回复需要乘255或者其他方式
        """
        # 转换为灰度图像#**只用灰度图象吗?色彩不是也很重要吗?
        self.image = image
        # self.image = image.convert('L')#可以直接加权也可以后面再处理
        transform = transforms.ToTensor()
        image_tensor_rgb = transform(image)#tensor处理完就变成了[0,1]之间的数值.

        if len(image_tensor_rgb.shape) ==3:
            image_tensor_rgb = image_tensor_rgb.unsqueeze(0) ## 形状: [1, 3, H, W]
        elif len(image_tensor_rgb.shape) == 4:
            pass
        else: assert False, "image shape error"
        
        self.B, self.C, self.rows, self.cols = image_tensor_rgb.shape ## 形状: [B, C, H, W]

        # 灰度图象
        weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        self.gray_tensor = (image_tensor_rgb * weights).sum(dim=1, keepdim=True)  # 形状: [B, 1, H, W]
        print(self.gray_tensor)
        # rgb各通道分离
        # self.r_channel, self.g_channel, self.b_channel = image_tensor[:, 0, :, :], image_tensor[:, 1, :, :], image_tensor[:, 2, :, :]

        
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
        """type: 'circular' or 'gaussian'"""
          # 创建低通滤波器（理想圆形滤波器）
        crow, ccol = self.rows // 2, self.cols // 2  # 中心点
        mask_low = torch.zeros((self.rows, self.cols), dtype=torch.float32)
        y, x = torch.meshgrid(torch.arange(self.rows), torch.arange(self.cols), indexing='ij')#用于生成网格序号,这样生成的

        if type == 'circular': 
            radius = 30  # 低通滤波器半径（可调整）,这是高低通的比例
            dist = torch.sqrt((x - ccol)**2 + (y - crow)**2)#生成点到中心点的距离
            mask_low[dist <= radius] = 1.0  # 中心区域设为1
            mask_low = mask_low.unsqueeze(0).unsqueeze(0)  # 形状: [1, 1, H, W]
        elif type == 'gaussian':
            # 创建高斯低通滤波器
            sigma = 30  # 高斯核标准差（可调整）
            mask_low = torch.exp(-((x - ccol)**2 + (y - crow)**2) / (2 * sigma**2))
            mask_low = mask_low / mask_low.max()  # 归一化
            mask_low = mask_low.unsqueeze(0).unsqueeze(0)  # 形状: [1, 1, H, W]
        # 创建高通滤波器（1 - 低通滤波器）
        mask_high = 1.0 - mask_low

        # 应用滤波器,乘法有广播broadcast效果#*但如果想要不同滤波器还需要再修改
        fshift_low = fshift * mask_low  # 低通滤波
        fshift_high = fshift * mask_high  # 高通滤波
        
        return fshift_low, fshift_high
    

    def ishow(spectrum,title='DCT Spectrum'):
        # 转换为可显示的格式
        spectrum = spectrum.squeeze().cpu()

        # 显示原始图片和频谱图
        plt.figure(figsize=(10, 5))

        plt.subplot(121)
        # plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(spectrum, cmap='gray')
        plt.title(' Spectrum')
        plt.axis('off')

        plt.show()
        spectrum = spectrum.numpy()
        cv2.imwrite('spectrum_r.jpg', spectrum)

    def save_images(self,low_freq, high_freq, gray_image, output_dir):
        """
        保存低频、高频和灰度图像为图像文件。
        Args:
            low_freq: 低频图像张量
            high_freq: 高频图像张量
            gray_image: 灰度图像张量
            output_dir: 保存路径
    """ 
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 转换为 PIL.Image 格式并保存
        low_freq_image = Image.fromarray((low_freq.numpy() *255).astype('uint8'))
        high_freq_image = Image.fromarray((high_freq.numpy() *255).astype('uint8'))
        gray_image = Image.fromarray((gray_image.numpy() * 255).astype('uint8'))
        #?另一种处理方式,采用绝对的正则化,还不太一样
        #low_freq = (low_freq - low_freq.min()) / (low_freq.max() - low_freq.min()) * 255
        #high_freq = (high_freq - high_freq.min()) / (high_freq.max() - high_freq.min()) * 255

        low_freq_image.save(os.path.join(output_dir, "low_freq.jpg"))
        high_freq_image.save(os.path.join(output_dir, "high_freq.jpg"))
        gray_image.save(os.path.join(output_dir, "gray_image.jpg"))

    def apply_dct(self):
        """
        image: PIL.image.image
        对图像进行 DCT 变换并分离低频和高频信息
        """
        # 计算 DCT 变换
        fshift, freq_mag,freq_phase = self.dct(self.gray_tensor)
        
        ##分离不同频带
        # todo 1.分离频谱和相位谱;2.分离多种频带并且动态设置差异频谱.
        fshift_low, fshift_high = self.filter_type(fshift,type='gaussian')  # 低通和高通滤波器

        # 计算逆 DCT 变换
        low_freq = self.idct(fshift_low)
        high_freq = self.idct(fshift_high)
       
        #为了可视化处理
        low_freq = low_freq.squeeze().cpu()  # 取模保留实数,虚数删去
        high_freq = high_freq.squeeze().cpu()
        gray_image = self.gray_tensor.squeeze().cpu()

        self.save_images(low_freq, high_freq, gray_image, "results/image3")  # 保存图像
        
    
    def __call__(self):
        self.apply_dct(self)



if __name__ == "__main__":
    # 测试代码
    image_path = "/home/yiruolei/ALLDATASET/AIGCDetect/Chameleon/test/0_real/1c4b521d-428d-4c91-bb17-2c1246ed94af.jpg"  # 替换为你的图像路径
    image = Image.open(image_path).convert('RGB')
    dct_transformer = DCTtrans(image)
    dct_transformer.apply_dct()
