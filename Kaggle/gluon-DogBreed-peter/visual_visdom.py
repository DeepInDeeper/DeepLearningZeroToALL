#coding:utf8
import visdom
import numpy as np


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
