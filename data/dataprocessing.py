# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io, os, pdb
import cv2, math, random
import numpy as np

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

from PIL import Image
import random

from data.patchingtest import ImageSampler

ImageFile.LOAD_TRUNCATED_IMAGES = True

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

class ImageSampler:#*确实还是当成一个新操作把，总想着套入transform还是需要多步操作，不如看成分离操作，内部也需要transform
    """
    一个用于随机采样图像块并将其拼接成固定大小图像的类
    如果是三通道图象则会显示最后一个的部分
    #!目前很多部分没粘贴图象,不一定好用,试一试.
    """
    def __init__(self, image, target_size=(512, 512), min_patch_size=(32, 32),transforms = None):
        """
        初始化ImageSampler。
        参数：
            image_path (str): 输入图像的路径。
            target_size (tuple): 目标拼接图像的大小（宽度，高度）。
            min_patch_size (tuple): 每个采样块的最小大小（宽度，高度）。
            max_patch_size (tuple): 每个采样块的最大大小（宽度，高度）。
        """
        self.image = image
        self.transforms = transforms
        self.target_width, self.target_height = target_size
        self.min_patch_width, self.min_patch_height = min_patch_size
        self.patches = []
        self.current_x = 0  # 当前拼接位置的x坐标
        self.current_y = 0  # 当前拼接位置的y坐标

    def sample_patch(self):
        """
        从输入图像中随机采样一个图像块，图像块的大小在指定范围内随机。
        返回：
            PIL.Image: 采样得到的图像块。
        """
        img_width, img_height = self.image.size

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
        patch = self.image.crop((x, y, x + patch_width, y + patch_height))
        return patch, patch_width, patch_height

    def stitch_patches(self, num_patches):
        """
        将采样得到的图像块拼接成目标大小的图像。
        参数：
            num_patches (int): 要采样并拼接的图像块数量。

        返回：
            PIL.Image: 拼接完成的图像。还没transform化
        """
        # 创建一个目标大小的空白图像
        stitched_image = Image.new('RGB', (self.target_width, self.target_height))

        for _ in range(num_patches):
            patch, patch_width, patch_height = self.sample_patch()

            patch_trans = self.transforms(patch)

            # 计算放置图像块的位置
            if self.current_x + patch_width > self.target_width + 50:
                self.current_x = 0
                self.current_y += patch_height

            if self.current_y + patch_height > self.target_height + 50:
                self.current_y = 0

            # 将图像块粘贴到拼接图像上
            stitched_image.paste(patch_trans, (self.current_x, self.current_y))

            # 更新当前拼接位置
            self.current_x += patch_width

        # 如果拼接图像超出目标大小，则裁剪至目标大小
        stitched_image = stitched_image.crop((0, 0, self.target_width, self.target_height))
        
        return stitched_image

    def __call__(self, num_patches):
        """为了能当成transform结构的函数来使用"""
        return self.stitch_patches(num_patches)

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








class TrainDataset(Dataset):

    def __init__(self, is_train, args):#*初始的时候会把所有路径整理出来.而不是图象,这也是一种方法,为了在getitem的时候速度比较快.

        TRANSFORM = Get_Transforms(args)#!甚至这个是创新点
        self.transform = TRANSFORM[0] if is_train else TRANSFORM[1]
        root = args.data_path if is_train else args.eval_data_path #*是大路径的的不同,因为在ai检测中使用不同数据集来使用,而不测试训练数据集的效果

        dataset_list = root.replace(' ', '').split(',')
        print(f"数据集列表: {dataset_list}")
        num_datasets = len(dataset_list)

        if num_datasets == 1:#只有一个列表
            real_list, fake_list = self.get_real_and_fake_lists(dataset_list[0],is_train)
            if is_train and args.num_train is not None:
                self.data_list = real_list[:args.num_train//2] + fake_list[:args.num_train//2]
            else:
                self.data_list = real_list + fake_list #*明显分界
        else:
            assert args.num_train is not None
            self.data_list = []
            for dataset in dataset_list:
                real_list, fake_list = self.get_real_and_fake_lists(dataset,is_train)
                self.data_list.extend(real_list[:args.num_train//(2 * num_datasets)] + fake_list[:args.num_train//(2 * num_datasets)])

    def get_image_paths(self, dir_path):
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
        image_paths = []
        for root, dirs, files in sorted(os.walk(dir_path)):
            for file in sorted(files):
                if file.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def get_real_and_fake_lists(self, folder_path,is_train):
        real_list, fake_list = [], []
        for root, dirs, files in sorted(os.walk(folder_path, followlinks=True)):#*因为最后会回到大文件夹中,这个结构下是不存在两个列表的.
            if is_train and  "train" in root:#训练集
                for dir_name in sorted(dirs):
                    if dir_name == "0_real":
                        real_dir_path = os.path.join(root, dir_name)
                        real_list.extend([{"image_path": image_path, "label" : 0} for image_path in self.get_image_paths(real_dir_path)])
                    elif dir_name == "1_fake":
                        fake_dir_path = os.path.join(root, dir_name)
                        fake_list.extend([{"image_path": image_path, "label" : 1} for image_path in self.get_image_paths(fake_dir_path)])
                continue
            elif not is_train and  "val" in root: #测试集
                for dir_name in sorted(dirs):
                    if dir_name == "0_real":
                        real_dir_path = os.path.join(root, dir_name)
                        real_list.extend([{"image_path": image_path, "label" : 0} for image_path in self.get_image_paths(real_dir_path)])
                    elif dir_name == "1_fake":
                        fake_dir_path = os.path.join(root, dir_name)
                        fake_list.extend([{"image_path": image_path, "label" : 1} for image_path in self.get_image_paths(fake_dir_path)])
                continue
            elif not is_train and  "test" in root:
                for dir_name in sorted(dirs):
                    if dir_name == "0_real":
                        real_dir_path = os.path.join(root, dir_name)
                        real_list.extend([{"image_path": image_path, "label" : 0} for image_path in self.get_image_paths(real_dir_path)])
                    elif dir_name == "1_fake":
                        fake_dir_path = os.path.join(root, dir_name)
                        fake_list.extend([{"image_path": image_path, "label" : 1} for image_path in self.get_image_paths(fake_dir_path)])
                continue
        return real_list, fake_list
    
            

    def __len__(self):

        return len(self.data_list)

    def __getitem__(self, index):
        
        sample = self.data_list[index]
        image_path, targets = sample['image_path'], sample['label']
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print(f'image error: {image_path}')
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        #*重新拼接操作，内部patching使用transform
      
        sampler = ImageSampler(image, target_size=(512, 512), min_patch_size=(8, 8),transforms=self.transform)#*把transform挪到里面去.
        image = sampler.stitch_patches(num_patches=128)#*这个超参可以删除??或者修改为动态的?
        
        if index == 0 :
            image.save("output.jpg")
            print('sampling-jpg_to_test')
        image = transforms.ToTensor()(image)
        
        return image, torch.tensor(int(targets))


if __name__ == "__main__":
    image = Image.open("/home/yiruolei/ALLDATASET/Chameleon/test/0_real/0a4dcb15-6fe3-4a28-9821-ad8da7823f15.jpg").convert('RGB')#*代替dataset结构，直接取一个。（确实无论是不是封装的感觉具体结构都是一样的。）
    TRANSFORM = Get_Transforms(arg)#!甚至这个是创新点（这个不是很要用，因为用的args，所以在测试的时候需要把arg能全部放过来，这也就是为什么用键值对好用的原因，因为在复制的时候很好复制。最好对自己的所有args都有键值对的服用）
    transform = TRANSFORM[0]
    sampler = ImageSampler(image, target_size=(512, 512), min_patch_size=(16, 16),transforms=transform)
    stitched_image = sampler.stitch_patches(num_patches=128)
    stitched_image.save("output.jpg")
    transform_to_tensor = transforms.ToTensor()
    a = transform_to_tensor(stitched_image)
    # print(a)