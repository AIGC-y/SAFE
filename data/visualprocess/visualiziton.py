import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn import decomposition

from sklearn.datasets import load_iris,load_digits

# 1. 

###不同可视化方式######
def visualize(x1,y1,info=None,size = 20):
    '嵌入空间可视化'''
    def aa(x):
        x_min, x_max = x.min(0), x.max(0)
        X_norm = (x - x_min) / (x_max - x_min)  # 归一化
        return X_norm
    
    x1 = aa(x1)
    # x2 = aa(x2)

    plt.figure(figsize=(size, size))
    #*设置点为label
    # for i in range(X_norm.shape[0]):#?这个用法貌似不好看,不如直接用点
    #     plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
    #             fontdict={'weight': 'bold', 'size': 9})
        # plt.scatter(X_norm[i, 0], X_norm[i, 1],  c=str(y[i]), #color=plt.cm.Set1(y[i]), 
                # )
    #*设置点为颜色点
    COLOR='tab10'
    plt.scatter(x1[:, 0], x1[:, 1], c=y1+10, cmap=COLOR)#*没有append直接用切片也可以
    # plt.scatter(x2[:, 0], x2[:, 1], c=y2+2, cmap='Oranges')

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.legend(frameon=False)
    plt.savefig("results/visual/images/%s_%s.jpg"%(info,sys._getframe().f_back.f_code.co_name))
    plt.show()
    print('figure saved')

def tSNE(x1,y1,info):
    """x,y,为numpy,
    n_components:是可视化的维度,不是原始维度"""
    x1_tsne = manifold.TSNE(n_components=2, init='pca', random_state=501, n_iter=1000, verbose=1).fit_transform(x1)
    # x2_tsne = manifold.TSNE(n_components=2, init='pca', random_state=501, n_iter=1000, verbose=1).fit_transform(x2)
    print("Org data dimension is {}. Embedded data dimension is {}".format(x1.shape[-1], x1_tsne.shape[-1]))
    visualize(x1_tsne, y1,info)

def isomap(x,y,info):
    X_isomap = Isomap(n_components=2).fit_transform(x)
    visualize(X_isomap, y,info)

def LLE(x,y,info):
    X_LLE = LocallyLinearEmbedding(n_components=2).fit_transform(x)
    visualize(X_LLE, y, info)

def PCA(x,y,info):
    X_PCA = decomposition.PCA(n_components=2).fit_transform(x)
    visualize(X_PCA, y, info)


##############################
#####数据载入########
# def loadmnist():
#     '读取mnist数据集'''
#     mnist = input_data.read_data_sets('./MNIST', one_hot=True)
#     X = mnist.validation.images
#     labels = mnist.validation.labels
#     y = np.argmax(labels, axis=1)
#     return X, y


if __name__ == "__main__":
    # x,y = loadmnist()
    # digits = load_digits()
    # x= digits.data
    # y= digits.target

    #**数据的获取就是重写一个test过程，只要数据输出，其他计算全删除就行了。这是最简单的方法，其他方法都需要改好多模块。这个就是需要什么什
    ##**注意别爆内存或者显存，数据一般的就直接保存。不行的需要用txt文件来追加。
    info = 'vitsafe-sdv4'
    y1 =  np.load('/home/yiruolei/project/AIGCdetector/SAFE/results/visual/datasave/label_vitsafe_sdv4.npy')
    x1 = np.load('/home/yiruolei/project/AIGCdetector/SAFE/results/visual/datasave/B_vitsafe_sdv4.npy')
    tSNE(x1,y1,info)

    # y2 = np.load('/home/yiruolei/project/AIGCdetector/SAFE/visual/datasave/b1.npy')
    # x2 = np.load('/home/yiruolei/project/AIGCdetector/SAFE/visual/datasave/d1.npy')
    # x3 = np.load('/home/yiruolei/project/AIGCdetector/SAFE/visual/datasave/e1.npy')
    
    # # x2 = np.load('/home/yiruolei/project/AIGCdetector/SAFE/results/datasave/c.npy')
    # # print(x==x2)
    # info1 = 'data4'
    # tSNE(x3,y2,info1)
    # info2='data5'
    # tSNE(x2,y2,info2)

   
    # isomap(x,y,info)
    # LLE(x,y,info)
    # PCA(x,y,info)