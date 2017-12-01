#coding:utf-8
import numpy as np

class Config:

    BASE_FOLDER = u'/media/yijie/文档/dataset/kaggle_Iceberg'
    batch_size = 128
    global_epoches = 55
    validationRatio = 0.11
    num_workers = 4
    LR = 0.0005
    MOMENTUM = 0.95
    global_seed=999
    env = 'scene'  # visdom env
    plot_every = 10 # 每10步可视化一次
    workers = 4 # CPU多线程加载数据
    model = "SimpleNet"
    result_path='submit.csv' #提交文件保存路径

def parse(self,kwargs,print_=True):
        '''
        根据字典kwargs 更新 config参数
        '''
        for k,v in kwargs.iteritems():
            if not hasattr(self,k):
                raise Exception("opt has not attribute <%s>" %k)
            setattr(self,k,v) 
        if print_:
            print('user config:')
            print('#################################')
            for k in dir(self):
                if not k.startswith('_') and k!='parse' and k!='state_dict':
                    print k,getattr(self,k)
            print('#################################')
        return self


def state_dict(self):
    return  {k:getattr(self,k) for k in dir(self) if not k.startswith('_') and k!='parse' and k!='state_dict' }

Config.parse = parse
Config.state_dict = state_dict
opt = Config()