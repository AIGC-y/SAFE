"""transforms 数据增强方式"""
import numpy as np
from PIL import Image
import random
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

import cv2
from scipy.fftpack import dct, idct

import io, os, pdb

##############################transform ###########################################
class RandomJPEG():
    def __init__(self, quality=95, interval=1, p=0.1):
        if isinstance(quality, tuple):
            self.quality = [i for i in range(quality[0], quality[1]) if i % interval == 0]
        else:
            self.quality = quality
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            if isinstance(self.quality, list):
                quality = random.choice(self.quality)
            else:
                quality = self.quality
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img = Image.open(buffer)
        return img

class RandomGaussianBlur():
    def __init__(self, kernel_size, sigma=(0.1, 2.0), p=1.0):
        self.blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return self.blur(img)
        return img

class RandomMask(object):
    def __init__(self, ratio=0.5, patch_size=16, p=0.5):
        """
        Args:
            ratio (float or tuple of float): If float, the ratio of the image to be masked.
                                             If tuple of float, random sample ratio between the two values.
            patch_size (int): the size of the mask (d*d).
        """
        if isinstance(ratio, float):
            self.fixed_ratio = True
            self.ratio = (ratio, ratio)
        elif isinstance(ratio, tuple) and len(ratio) == 2 and all(isinstance(r, float) for r in ratio):
            self.fixed_ratio = False
            self.ratio = ratio
        else:
            raise ValueError("Ratio must be a float or a tuple of two floats.")

        self.patch_size = patch_size
        self.p = p

    def __call__(self, tensor):

        if random.random() > self.p: return tensor

        _, h, w = tensor.shape
        mask = torch.ones((h, w), dtype=torch.float32)

        if self.fixed_ratio:
            ratio = self.ratio[0]
        else:
            ratio = random.uniform(self.ratio[0], self.ratio[1])

        # Calculate the number of masks needed
        num_masks = int((h * w * ratio) / (self.patch_size ** 2))

        # Generate non-overlapping random positions
        selected_positions = set()
        while len(selected_positions) < num_masks:
            top = random.randint(0, (h // self.patch_size) - 1) * self.patch_size
            left = random.randint(0, (w // self.patch_size) - 1) * self.patch_size
            selected_positions.add((top, left))

        for (top, left) in selected_positions:
            mask[top:top+self.patch_size, left:left+self.patch_size] = 0

        return tensor * mask.expand_as(tensor)

class ImageSampler:
    """
    一个用于随机采样图像块并将其拼接成固定大小图像的类
    如果是三通道图象则会显示最后一个的部分
    #!目前很多部分没粘贴图象,不一定好用,试一试.
    """

    def __init__(self, target_size: tuple=(512, 512), min_patch_size: tuple=(32, 32),num_patches: int=40, transforms = None):
        """
        初始化ImageSampler。

        参数：
            target_size (tuple): 目标拼接图像的大小（宽度，高度）。
            min_patch_size (tuple): 每个采样块的最小大小（宽度，高度）。
            max_patch_size (tuple): 每个采样块的最大大小（宽度，高度）。

        """
        # self.image = image#*这个逻辑不对,初始化只初始化参数/
        # self.transforms = transforms
        self.num_patches = num_patches
        self.target_width, self.target_height = target_size
        self.min_patch_width, self.min_patch_height = min_patch_size
        self.patches = []
        self.current_x = 0  # 当前拼接位置的x坐标
        self.current_y = 0  # 当前拼接位置的y坐标

    def sample_patch(self,image: Image.Image):
        """
        从输入图像中随机采样一个图像块，图像块的大小在指定范围内随机。

        返回：
            PIL.Image: 采样得到的图像块。
        """
        img_width, img_height = image.size

        # 随机确定图像块的大小,不手动设置,设置为图的大小为上限
        patch_width = random.randint(self.min_patch_width, int(img_width/2))
        patch_height = random.randint(self.min_patch_height, int(img_height/2))

        max_x = img_width - patch_width
        max_y = img_height - patch_height

        if max_x < 0 or max_y < 0:
            raise ValueError("图像块大小不能大于输入图像大小。")
 
        # 随机选择采样位置
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        # 裁剪出图像块
        patch = image.crop((x, y, x + patch_width, y + patch_height))
        return patch, patch_width, patch_height

    def stitch_patches(self,image: Image.Image ):
        """
        将采样得到的图像块拼接成目标大小的图像。

        参数：
            num_patches (int): 要采样并拼接的图像块数量。

        返回：
            PIL.Image: 拼接完成的图像。还没transform化
        """
        # 创建一个目标大小的空白图像
        stitched_image = Image.new('RGB', (self.target_width, self.target_height))

        for _ in range(self.num_patches):
            patch, patch_width, patch_height = self.sample_patch(image)

            # patch = self.transforms(patch)

            # 计算放置图像块的位置
            if self.current_x + patch_width > self.target_width + 50:
                self.current_x = 0
                self.current_y += patch_height

            if self.current_y + patch_height > self.target_height + 50:
                self.current_y = 0

            # 将图像块粘贴到拼接图像上
            stitched_image.paste(patch, (self.current_x, self.current_y))

            # 更新当前拼接位置
            self.current_x += patch_width

        # 如果拼接图像超出目标大小，则裁剪至目标大小
        stitched_image = stitched_image.crop((0, 0, self.target_width, self.target_height))
        
        return stitched_image
    
    def __call__(self, image: Image.Image):
        """为了能当成transform结构的函数来使用"""
        return self.stitch_patches(image)


def apply_dct(image):
    """
    对图像进行 DCT 变换并分离低频和高频信息
    """
    # 转换为灰度图像#**只用灰度图象吗?色彩不是也很重要吗?
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 对图像进行 DCT 变换
    dct_transformed = dct(dct(gray.T, norm='ortho').T, norm='ortho')
    
    # 分离低频和高频信息
    h, w = dct_transformed.shape
    low_freq = np.zeros_like(dct_transformed)
    high_freq = np.zeros_like(dct_transformed)
    
    # 低频区域 (保留左上角 1/4 的系数)#?具体要多少不一定
    low_freq[:h//2, :w//2] = dct_transformed[:h//2, :w//2]
    # 高频区域 (其余部分)numpy
    high_freq = dct_transformed - low_freq
    
    # 将 numpy 数组转换为 PyTorch 张量
    high_freq = torch.from_numpy(high_freq)
    low_freq = torch.from_numpy(low_freq)
    return low_freq, high_freq

####################################################################################
##################transformcompose##############################
def Get_Transforms(args):
    """把transform这个函数套多层，之后能直接通过使用
        TRANSFORM = Get_Transforms(args)\n
        self.transform = TRANSFORM[0] if is_train else TRANSFORM[1]
    """

    size = args.input_size
    #*
    TRANSFORM_DICT = {
        'resize_BILINEAR': {
            'train': [
                transforms.RandomResizedCrop([size, size], interpolation=InterpolationMode.BILINEAR),
            ],
            'eval': [
                transforms.Resize([size, size], interpolation=InterpolationMode.BILINEAR),
            ],
        },

        'resize_NEAREST': {
            'train': [
                transforms.RandomResizedCrop([size, size], interpolation=InterpolationMode.NEAREST),
            ],
            'eval': [
                transforms.Resize([size, size], interpolation=InterpolationMode.NEAREST),
            ],
        },

        'crop': {
            'train': [
                transforms.RandomCrop([size, size], pad_if_needed=True),
            ],
            'eval': [
                transforms.CenterCrop([size, size]),
            ],
        },

        'source': {
            'train': [
                transforms.RandomCrop([size, size], pad_if_needed=True),
            ],
            'eval': [
            ],
        },
        'ori': {
            'train': [],
            'eval': [],
        },
    }

    # region [Augmentations]
    transform_train, transform_eval = TRANSFORM_DICT[args.transform_mode]['train'], TRANSFORM_DICT[args.transform_mode]['eval']

    #!目前不太理解为什么只有寻来
    transform_train.extend([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        # transforms.ToTensor(),#*因为要先贴图片才可以
        # RandomMask(ratio=(0.00, 0.75), patch_size=16, p=0.5),#*就单纯的随机掩码就行啊
    ])

    # transform_eval.append(transforms.ToTensor())#*也是最后再加这个
    # endregion

    # region [Perturbatiocns in Testing]
    if args.jpeg_factor is not None:
        transform_eval.insert(0, RandomJPEG(quality=args.jpeg_factor, p=1.0))
    if args.blur_sigma is not None:
        transform_eval.insert(0, transforms.GaussianBlur(kernel_size=5, sigma=args.blur_sigma))
    if args.mask_ratio is not None and args.mask_patch_size is not None:
        transform_eval.append(RandomMask(ratio=args.mask_ratio, patch_size=args.mask_patch_size, p=1.0))
    # endregion

    return transforms.Compose(transform_train), transforms.Compose(transform_eval)

################################################################

# 示例用法
if __name__ == "__main__":
    image = Image.open("/home/yiruolei/ALLDATASET/Chameleon/test/0_real/0a4dcb15-6fe3-4a28-9821-ad8da7823f15.jpg").convert('RGB')
    transformtest = transforms.compose([ #*可以建立很多层,只要用[]的列表写上就行,所以可以用字典来设置不同情况
        # RandomJPEG(quality=(50, 95), interval=5, p=0.5),
        # RandomGaussianBlur(kernel_size=3, sigma=(0.1, 2.0), p=0.5),
        # RandomMask(ratio=0.5, patch_size=16, p=0.5),
        ImageSampler(target_size=(512, 512), min_patch_size=(32, 32), num_patches=40),

    ])

    stitched_image = transformtest(image)
    stitched_image.save("output.jpg")

    transform_to_tensor = transforms.ToTensor()
    a = transform_to_tensor(stitched_image)
    # print(a)