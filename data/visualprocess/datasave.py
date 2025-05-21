
#import pickle
import pandas as pd
import h5py

save_path="images/a"
##整体写入

#with open(save_path,'w'),貌似可以控制在with结构下的模块打开,不会一直往里面写.



# with open('/home/yiruolei/project/FND/M3FEND-main/data/en/test.pkl', "rb") as f:

# print(type(t))
    # number=5535
# print(t['content'][number])
# print(t['comments'][number])

#*方法1 !直接把整体的dataframe输入是缩略图
# with open('./test.txt','w') as f:#
#     f.write(str(t))
#*方法2 直接把整体写入,txt或者csv(?csv速度快很多)json很多乱ma
    #?一般都是一行行遍历,所以如果只要两列也要遍历所有数据,否则两个模型
# t['content'].to_csv('./data/test.csv',s)
# t['comments'].to_csv('./data/test.csv')
# t['comments'].to_json('./data/test.json')
#!这里只能用整体,如果是t['content']:AttributeError: 'Series' object has no attribute 'to_html'
# t.to_html('./data/entest.html')

# a = pd.read_html('./data/entest.html')
# print(a)
# #*|content|comments|category|label|content_emotion|comments_emotion|emotion_gap|style_feature
# #*数据后面的 content_emotion|comments_emotion|emotion_gap|style_feature是处理好的矩阵形式！


#能用显存还是他妈用显存,为啥内存更容易包我真不理解,一直上涨,但是显存cat没有

# 1. 写入数据到HDF5文件
def save_features_to_hdf5(batch_data,file_path = 'visual/datasave/features.h5'):
    """x:特征数据，形状为 (num_samples, feature_dim)
    # batch_data: 张量数据，形状为 (batch_size, height, width, channels)
    z
    """
    with h5py.File(file_path, 'a') as f:#*以字典这个结构保存的数据集?卧槽?里面格式这么多吗?
        if 'features' not in f:
            # 创建一个可调整大小的数据集，初始形状为 (0, *batch_data.shape[1:])
            dset = f.create_dataset('features', 
                                    (0, *batch_data.shape[1:]), #初始为0行,后面维度不变
                                    maxshape=(None, *batch_data.shape[1:]),#第一维可无限扩展
                                    chunks=True,#分块储存
                                    dtype='float32')#存储的数据类型
        else:
            dset = f['features']
        
        current_size = dset.shape
        # 扩展数据集大小并写入新数据
        dset.resize(current_size + batch_data.shape[0], axis=0)
        dset[current_size:current_size + batch_data.shape[0], ...] = batch_data
    # print("特征已保存到 features.h5")

# 2. 从HDF5文件读取数据,需要明确取什么块
def load_features_from_hdf5(file_path='features.h5', start_idx=0, num_samples=None):
    """
    从HDF5文件中读取特征数据。
    参数:
        file_path: HDF5文件路径，默认为 'features.h5'
        start_idx: 起始索引，默认为 0
        num_samples: 要读取的样本数，默认为 None（读取从start_idx到末尾的所有数据）
    返回:
        读取的特征数据
    """
    with h5py.File(file_path, 'r') as f:
        dset = f['features']
        if num_samples is None:
            return dset[start_idx:]
        else:
            return dset[start_idx:start_idx + num_samples]
        
