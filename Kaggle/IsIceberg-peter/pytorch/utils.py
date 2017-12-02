#coding:utf-8
import numpy as np
import visdom


class Visualizer():
    '''
    对可视化工具visdom的封装
    '''
    def __init__(self, env, **kwargs):
        import visdom
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}

    def plot_many(self, d):
        for k, v in d.iteritems():
            self.plot(k, v)

    def plot(self, name, y):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=unicode(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append'
                      )
        self.index[name] = x + 1


class Config:

    BASE_FOLDER = u'/media/yijie/文档/dataset/kaggle_Iceberg'
    batch_size = 128
    global_epoches = 55
    validationRatio = 0.11
    period = 20
    num_workers = 4
    LR = 0.0005
    MOMENTUM = 0.95
    global_seed=999
    env = 'Iceberg'  # visdom env
    workers = 4 # 多线程加载数据
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