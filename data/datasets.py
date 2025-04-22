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
from data.patchtrans import *

from PIL import Image
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True


def Get_Transforms(args):

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
            'train': [ImageSampler((512, 512), (8, 8), 128),],
            'eval': [],
        },
    }

    # region [Augmentations]
    transform_train, transform_eval = TRANSFORM_DICT[args.transform_mode]['train'], TRANSFORM_DICT[args.transform_mode]['eval']

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
        
        self.is_train =is_train#*?只是为了在分块的时候思考用不用，可以不要，然后卸载tranform中但还没想好

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

        #*先重新拼接一下(其实最好是放在transform中,但是目前没放进去)
        #*这里设置的大小也比原本的大,反正会裁
        
        image = self.transform(image)#输出的大小要是固定大小才可以，如果上面的处理删除了，在transform中尺寸久不对了
        # if index == 0 :
        #     image.save("output.jpg")
        #     print('sampling-jpg_to_test')
        image = transforms.ToTensor()(image)
        
        return image, torch.tensor(int(targets))


if __name__ == "__main__":
#这里测试不同的代码
    # print(a)