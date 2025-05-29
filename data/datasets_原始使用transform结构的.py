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
from data.patchtrans import apply_dct
from data.patchtrans import *
from data.patchtrans import process_patches
from data.dct_test import DCTtrans

from PIL import Image
import random
import time as TI

ImageFile.LOAD_TRUNCATED_IMAGES = True


def Get_Transforms(args):
    """
    这里的所有操作都是可以对image也可以对tensor结构的，所有
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
        # 'ori': {
        #     'train': [transforms.ToTensor(),
        #         FastPatchStitchTransform((size, size), (4, 4), 64),
        #               ],
        #     'eval': [transforms.ToTensor(),
        #         FastPatchStitchTransform((size, size), (4, 4), 64),
        #              ],
        # },
         'ori': {
            'train': [ImageSampler((size, size), (16, 16), 200),
                      ],
            'eval': [ImageSampler((size, size), (16, 16), 200),
                # transforms.RandomCrop([size, size], pad_if_needed=True),#没有必要去拼了
                     ],
        },
    }
##transforms.RandomCrop、RandomHorizontalFlip、RandomRotation、ColorJitter 这些操作对 PIL.Image 做通常比对 tensor 做更快、更省内存（因为 PIL 是 C 实现，且只处理 uint8）。
#如果你把 ToTensor() 放前面，后续所有操作都在 float32 的 tensor 上进行，内存占用更大，部分操作速度反而变慢。
    # region [Augmentations]
    transform_train, transform_eval = TRANSFORM_DICT[args.transform_mode]['train'], TRANSFORM_DICT[args.transform_mode]['eval']

    transform_train.extend([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.ToTensor(),#*因为要先贴图片才可以
        # RandomMask(ratio=(0.00, 0.75), patch_size=16, p=0.5),#*就单纯的随机掩码就行啊
    ])

    transform_eval.append(transforms.ToTensor())#*也是最后再加这个
    # endregion

    # region [Perturbatiocns in Testing]
    if args.jpeg_factor is not None:
        transform_eval.insert(0, RandomJPEG(quality=args.jpeg_factor, p=1.0))
    if args.blur_sigma is not None:
        transform_eval.insert(0, transforms.GaussianBlur(kernel_size=5, sigma=args.blur_sigma))
    if args.mask_ratio is not None and args.mask_patch_size is not None:
        transform_eval.append(RandomMask(ratio=args.mask_ratio, patch_size=args.mask_patch_size, p=1.0))
    # endregion

    ### *补充
    transform2 = transforms.Compose([
            transforms.RandomCrop([size, size], pad_if_needed=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.ToTensor(),
            RandomMask(ratio=(0.00, 0.75), patch_size=16, p=0.5),#*就单纯的随机掩码就行啊
    ])
    transform3 = transforms.Compose([
            transforms.RandomCrop([size, size], pad_if_needed=True),
            transforms.ToTensor(),
    ])            


    return transforms.Compose(transform_train), transforms.Compose(transform_eval),transform2, transform3


class TrainDataset(Dataset):

    """文件结构-索引列表化,然后对列表进行图象取出\n
    采用walk结构对所有下层子文件都整理为双列表\n
        但是为什么固态和机械的处理时间相同呢? 
    """
    def __init__(self, is_train, args):#*初始的时候会把所有路径整理出来.而不是图象,这也是一种方法,为了在getitem的时候速度比较快.
        
        self.is_train =is_train#*?只是为了在分块的时候思考用不用，可以不要，然后卸载tranform中但还没想好

        TRANSFORM = Get_Transforms(args)#!甚至这个是创新点
        self.transform1 = TRANSFORM[0] if is_train else TRANSFORM[1]
        self.transform2 = TRANSFORM[2] if is_train else TRANSFORM[3]
        root = args.data_path if is_train else args.eval_data_path #*是大路径的的不同,因为在ai检测中使用不同数据集来使用,而不测试训练数据集的效果

        dataset_list = root.replace(' ', '').split(',')
        # print(f"数据集列表: {dataset_list}")
        num_datasets = len(dataset_list)

        if num_datasets == 1:#只有一个列表
            real_list, fake_list = self.get_real_and_fake_lists(dataset_list[0],is_train)
            # print(f"真实列表: {real_list}, 伪影列表: {fake_list}") 
            # print(f"真实列表: {len(real_list)}, 伪影列表: {len(fake_list)}")
            if args.ratio_train < 1:
                real_index = int(args.ratio_train*len(real_list))
                fake_index = int(args.ratio_train*len(fake_list))
                # print(f"real_index: {real_index}, fake_index: {fake_index}")
                if is_train:
                    self.data_list = real_list[:real_index-1] + fake_list[:fake_index-1]
                else:
                    self.data_list = real_list[real_index:] + fake_list[fake_index:]#*这个是测试的时候使用的_list):]
                # print(f"数据长度: {len(self.data_list)}")
            else:
                # if args.eval:
                #      self.data_list = fake_list #*现在测试就用假的
                # else:
                #     self.data_list = real_list + fake_list
               self.data_list = real_list + fake_list
            #    print(f"数据长度: {len(self.data_list)}")
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
            if is_train :#训练集,就是给什么就用什么,子目录全用了.
                for dir_name in sorted(dirs):
                    if dir_name == "0_real" or dir_name =='nature':
                        real_dir_path = os.path.join(root, dir_name)
                        real_list.extend([{"image_path": image_path, "label" : 0} for image_path in self.get_image_paths(real_dir_path)])
                    elif dir_name == "1_fake" or dir_name == 'ai':
                        fake_dir_path = os.path.join(root, dir_name)
                        fake_list.extend([{"image_path": image_path, "label" : 1} for image_path in self.get_image_paths(fake_dir_path)])
                continue
            elif not is_train and  "val" in root: #测试集
                for dir_name in sorted(dirs):
                    if dir_name == "0_real" or dir_name ==  'nature':
                        real_dir_path = os.path.join(root, dir_name)
                        real_list.extend([{"image_path": image_path, "label" : 0} for image_path in self.get_image_paths(real_dir_path)])
                    elif dir_name == "1_fake" or dir_name == 'ai':
                        fake_dir_path = os.path.join(root, dir_name)
                        fake_list.extend([{"image_path": image_path, "label" : 1} for image_path in self.get_image_paths(fake_dir_path)])
                continue
            elif not is_train and  "test" in root:
                for dir_name in sorted(dirs):
                    if dir_name == "0_real" or dir_name == 'nature':
                        real_dir_path = os.path.join(root, dir_name)
                        # print("real_dir_path",real_dir_path)
                        real_list.extend([{"image_path": image_path, "label" : 0} for image_path in self.get_image_paths(real_dir_path)])
                    elif dir_name == "1_fake" or dir_name == 'ai':
                        fake_dir_path = os.path.join(root, dir_name)
                        # print("fake_dir_path",fake_dir_path)
                        fake_list.extend([{"image_path": image_path, "label" : 1} for image_path in self.get_image_paths(fake_dir_path)])
                continue
        return real_list, fake_list
            

    def __len__(self):

        return len(self.data_list)

    def __getitem__(self, index):
        
        sample = self.data_list[index]
        image_path, targets = sample['image_path'], sample['label']
        
        #####训测都变化######
        # add = self.data_list[random.randint(0, len(self.data_list) - 1)]
        # addimg_path, addtargets = add['image_path'], add['label']
        
        # replaceratio=random.random()
        # targets = (1-replaceratio) * targets + replaceratio * addtargets
        # image = process_patches(image_path, addimg_path,replace_ratio= replaceratio)
        #######
        if self.is_train:##使用标签融合不行
            add = self.data_list[random.randint(0, len(self.data_list) - 1)]
            addimg_path, addtargets = add['image_path'], add['label']
            
            replaceratio=random.random()
            targets = (1-replaceratio) * targets + replaceratio * addtargets
            image = process_patches(image_path, addimg_path,replace_ratio= replaceratio)
        else:
            try:
                image = Image.open(image_path).convert('RGB')
                # print(f'open image: {image_path}',image)
            except:
                print(f'image error: {image_path}')
                return self.__getitem__(random.randint(0, len(self.data_list) - 1))
        try:
                image = Image.open(image_path).convert('RGB')
                # print(f'open image: {image_path}',image)
        except:
                print(f'image error: {image_path}')
                return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        image_patch = self.transform1(image)#输出的大小要是固定大小才可以，如果上面的处理删除了，在transform中尺寸久不对了 #[C,H,W]
        image_ori = self.transform2(image)

        #todo *对特征进行频谱还是图象频谱,反正得对图象patch然后在分类不同特征.
        #* 潜在DF模型的思路有借鉴意义吗??这个是生成图象,痕迹被消失了.感觉其实一般了这样.?
        ###* DCT是可逆变换.是不是平移不变变换呢????可以不可以换层还不知道,学习一下别人怎么写的这个也可以产生一个大点...也是拼接原理::这个结构本身是如何.而任务需要这样的吗???
        # print('image.shape',image)
        # dct_transformer = DCTtrans(image_ori)
        # lowfreq, highfreq= dct_transformer.apply_dct(channel='rgb')
        # print('type',type(image_ori),type(highfreq))
        # a = image_ori - lowfreq
        # a = lowfreq
        # a = highfreq
        a= 0
        lowfreq2, highfreq2= DCTtrans(image_patch).apply_dct(channel='rgb') #[C,H,W]都是图片结构的
        # # print('image_patch',image_patch.shape,"highfreq:",highfreq2.shape,"highfreq:",highfreq.shape,)
        # # b = image_patch - lowfreq2
        # # b = highfreq2
        b = lowfreq2
        # b=0

        #*先尝试只用这三个数据看看
        return (image_ori,image_patch, a, b,), torch.tensor(int(targets))
        # return image_patch, torch.tensor(int(targets))


def save_feature(feature, output_dir):
    """
    保存一个数据为图像文件。batch数据就需要遍历batch
    Args:
        feature:tensor张量要先cpu化
        output_dir: 保存路径
""" 
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    time = TI.time()
    x = feature.squeeze().cpu()  # 形状: [H, W]

    # 转换为 PIL.Image 格式并保存
    image = Image.fromarray((x.numpy() *255).astype('uint8'))
    
    #?另一种处理方式,采用绝对的正则化,还不太一样
    #image = (x - x.min()) / (x.max() - x.min()) * 255

    image.save(os.path.join(output_dir, f"{time}.jpg"))


# if __name__ == "__main__":
#     for i in 
# #这里测试不同的代码
 
