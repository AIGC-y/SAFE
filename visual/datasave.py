
#import pickle
import pandas as pd

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