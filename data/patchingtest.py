import numpy as np
from PIL import Image
import random
from torchvision import transforms

class ImageSampler:
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
        # self.transforms = transforms
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

# 示例用法
if __name__ == "__main__":
    image = Image.open("/home/yiruolei/ALLDATASET/Chameleon/test/0_real/0a4dcb15-6fe3-4a28-9821-ad8da7823f15.jpg").convert('RGB')
    sampler = ImageSampler(image, target_size=(512, 512), min_patch_size=(16, 16))
    stitched_image = sampler.stitch_patches(num_patches=128)
    stitched_image.save("output.jpg")
    transform_to_tensor = transforms.ToTensor()
    a = transform_to_tensor(stitched_image)
    # print(a)