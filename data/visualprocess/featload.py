import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold,datasets
import torch
import tqdm
from models.m3fend import M3FENDModel
from models.multiexpert_orig import C2Expert

from utils.dataloader import bert_data
from utils.utils import data2gpu, Averager, metrics, Recorder, batch_mini2sub

# from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

def getFileLogger(self, log_file):
        logger = logging.getLogger()
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

def TSNE(x,y,info):
    '''X是特征，不包含target;X_tsne是已经降维之后的特征
        y可以是任何标签:领域或者真假或者分类
    '''

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(x)
    print("Org data dimension is {}. Embedded data dimension is {}".format(x.shape[-1], X_tsne.shape[-1]))
        
    '''嵌入空间可视化'''

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    size=12
    plt.figure(figsize=(size, size),frameon=False)
   
   #*设置点为label
    # for i in range(X_norm.shape[0]):#?这个用法貌似不好看,不如直接用点
    #     plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
    #             fontdict={'weight': 'bold', 'size': 9})
        # plt.scatter(X_norm[i, 0], X_norm[i, 1],  c=str(y[i]), #color=plt.cm.Set1(y[i]), 
                # )
    #*设置点为颜色点
    COLOR='tab10'
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y+10, cmap=COLOR)#*没有append直接用切片也可以
    #marker=X_norm[:, 1]

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.legend(frameon=False)
    plt.savefig('images/'+info+COLOR+'_'+str(size)+'test.pdf', dpi=120)
    plt.show()
    print('figure saved')


    # X_tsne = manifold.TSNE(n_components=2,random_state=33).fit_transform(x)
    # # X_pca = PCA(n_components=2).fit_transform(digits.data)

    # ckpt_dir="images"
    # if not os.path.exists(ckpt_dir):
    #     os.makedirs(ckpt_dir)

    # plt.figure(figsize=(10, 5))
    # plt.subplot(121)
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y,label="t-SNE")
    # plt.legend()
    # plt.subplot(122)
    # # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y,label="PCA")
    # plt.legend()
    # plt.savefig('images/digits_tsne-pca.png', dpi=120)
    # plt.show()

# def allimage(x,y1,y2,info):
#     tSNE(x,y1,info)
#     tSNE(x,y2,info)
#     isomap(x,y1,info)
#     isomap(x,y2,info)
#     LLE(x,y1,info)
#     LLE(x,y2,info)
#     PCA(x,y1,info)
#     PCA(x,y2,info)


def c2experttest(model,data_iter):#?是否也可以不用subbatch来测试,应该模型是不记录batchsize数据的?
    category=[] 
    label = []
    feat=[]
    for step_n, batch in enumerate(data_iter):
        with torch.no_grad():
            subbatch= batch_mini2sub(batch,domain_num)
            for i in range(domain_num):
                if subbatch[i][0].shape != torch.Size([0]):#*要么用大小对应来看,要么是否也能用item这个还没有测试,但是本质上应该更简单
                    subbatch_data = data2gpu(subbatch[i], True)
                    #没有进行标注的数据,因为在dataloder中不好用吗?在batch中重新用元组表示,
                    sublabel = subbatch_data['label']
                    subcategory = subbatch_data['category']
                    label.extend(sublabel.detach().cpu().numpy().tolist())
                    category.extend(subcategory.detach().cpu().numpy().tolist())

                    # batch_feat = model(subbatch_data)['expert_feat']
                    cat_feat = model(subbatch_data)['cat_feat']
                    # print(batch_feat.shape)
                    feat.extend(cat_feat.detach().cpu().numpy().tolist())
                    # feat.extend(batch_feat.detach().cpu().numpy().tolist())
    return label, category, feat

def m3test(model, data_iter):
    category=[] 
    label = []
    feat=[]
    print('开始处理')
    for step_n, batch in enumerate(data_iter):
        with torch.no_grad():
             with torch.no_grad():
                batch_data = data2gpu(batch, True)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                # _,batch_feat = model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                # feat.extend(batch_feat.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
    
    with open('/home/yiruolei/project/FND/TaMvMe/store.pkl', "rb") as f:
        feat  = pickle.load(f)

    return label, category, feat              

def MTMCASE(model,data_iter):
    # logger = getFileLogger('./','casestuydy.txt')
    category_dict = {
            "科技": 0,
            "军事": 1,
            "教育考试": 2,
            "灾难事故": 3,
            "政治": 4,
            "医药健康": 5,
            "财经商业": 6,
            "文体娱乐": 7,
            "社会生活": 8,
            }
    sig=torch.nn.Sigmoid()
    pred = []
    label = []
    category = []
    pred=[[],[],[],[],[],[],[],[],[]]
    for step_n, batch in enumerate(data_iter):
        with torch.no_grad():
            batch_data = data2gpu(batch, True)
            #没有进行标注的数据,因为在dataloder中不好用吗?在batch中重新用元组表示,
            batch_label = batch_data['label']
            batch_category = batch_data ['category']
            label.extend(batch_label.detach().cpu().numpy().tolist())
            category.extend(batch_category.detach().cpu().numpy().tolist())
            for i in range(9):
                pred[i].extend(sig(model(batch_data)['each_pre'][i]).detach().cpu().numpy().tolist())
    # print(label.size(),pred.size())            
    for i in range(9):
        result=metrics(label, pred[i], category, category_dict)
        # logger.info()
        print(result)


if __name__ == '__main__':
    # with open ('a.json','w') as f:
    #     json.dump(torch.load('/home/yiruolei/project/FND/log/firsttask/logs/param_model_ch_3/m3fend/parameter_m3fend.pkl')numpy(),f)
    # with open ('b.json','w') as f:
    #     json.dump(torch.load('/home/yiruolei/project/FND/log/firsttask/logs/param_model/m3fend/parameter_m3fend.pkl').numpy(),f)
    # print(torch.load('/home/yiruolei/project/FND/log/firsttask/logs/param_model_ch_3/m3fend/parameter_m3fend.pkl'))
    

    # #*visualization test
    # digits = load_digits()
    # # print(digits.data.shape,digits.target.shape)
    # TSNE(digits.data,digits.target,'aa')
    
    dataset='ch'
    domain_num =9
    # model_name = 'm3FENModel'
    model_name = 'c2expert'
    INFO='mlp融合-768'
    # INFO='cat_feat'
    #*load data
    if dataset == 'en':
        root_path = './data/en/'
        category_dict = {
        "gossipcop": 0,
        "politifact": 1,
        "COVID": 2,
        }
        para_pa = 'multiexpert_256en_3_'+INFO+'.pkl'
    elif dataset == 'ch':
        root_path = './data/ch/'
        if domain_num == 9:
            category_dict = {
            "科技": 0,
            "军事": 1,
            "教育考试": 2,
            "灾难事故": 3,
            "政治": 4,
            "医药健康": 5,
            "财经商业": 6,
            "文体娱乐": 7,
            "社会生活": 8,
            }
            para_pa = 'multiexpert_256ch_9_'+INFO+'.pkl'
        elif domain_num == 6:
            category_dict = {
            "教育考试": 0,
            "灾难事故": 1,
            "医药健康": 2,
            "财经商业": 3,
            "文体娱乐": 4,
            "社会生活": 5,
            }
            para_pa = 'multiexpert_256ch_6_'+INFO+'.pkl'
        elif domain_num == 3:
            category_dict = {
            "政治": 0,  #852
            "医药健康": 1,  #1000
            "文体娱乐": 2,  #1440
            }
            para_pa = 'multiexpert_256ch_3_'+INFO+'.pkl'
    test_path=root_path+ 'test.pkl'
    loader = bert_data(max_len = 170, batch_size = 256,
                        category_dict = category_dict, num_workers=4, dataset_name = dataset)
    test_loader= loader.load_data(test_path, False)
    print('data loaded')
    
    #*load model
    if model_name == 'c2expert':
        param_root = '/home/yiruolei/project/FND/log/firsttask/logs/param_model/longtail/'
        model_saved_path = param_root + para_pa 
        model = C2Expert(dataset=dataset,
                        dropout=None, 
                        reduce_dimension=True, 
                        use_norm=True, 
                        num_experts=domain_num)
       
    elif model_name == 'm3FENModel':
        model_saved_path='/home/yiruolei/project/FND/log/firsttask/logs/param_model_'+dataset+'_'+str(domain_num)+'/m3fend/parameter_m3fend.pkl'
        model = M3FENDModel(768, [384], 0.2, 
                            7, 7, 2, 
                            50, domain_num,dataset=dataset)
    model =model.cuda()
    # model_saved_path2='/home/yiruolei/project/FND/log/firsttask/logs/param_model/m3fend/parameter_m3fend.pkl'
  
    # print(torch.load(model_saved_path),torch.load(model_saved_path2))
    model.load_state_dict(torch.load(model_saved_path))
    print('model loaded')
    
    
    
    model.eval()
    # print(model)#?为什么这里的domainmemory内部都是空的?
    data_iter = tqdm.tqdm(test_loader) 
   
    #*casestudy
    # MTMCASE(model,data_iter)

    #*可视化
    if model_name == 'c2expert':
        label, category, feat = c2experttest(model,data_iter)
       
    elif model_name == 'm3FENModel':
        label, category, feat = m3test(model,data_iter)
        
                    
    label = np.array(label)   
    category = np.array(category)
    feat = np.array(feat)   
    # print(feat.shape)
    # print(feat)
    TSNE(feat,label,str(model_name)+'_'+str(dataset)+'_'+str(domain_num)+'_label_')
    TSNE(feat,category,str(model_name)+'_'+str(dataset)+'_'+str(domain_num)+'_category_')
    #######*######
    
  


    # digits = load_digits()
    # X_tsne = manifold.TSNE(n_components=2,random_state=33).fit_transform(digits.data)
    # X_pca = PCA(n_components=2).fit_transform(digits.data)

    # ckpt_dir="images"
    # if not os.path.exists(ckpt_dir):
    #     os.makedirs(ckpt_dir)

    # plt.figure(figsize=(10, 5))
    # plt.subplot(121)
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=digits.target,label="t-SNE")
    # plt.legend()
    # plt.subplot(122)
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target,label="PCA")
    # plt.legend()
    # plt.savefig('images/digits_tsne-pca.png', dpi=120)
    # plt.show()