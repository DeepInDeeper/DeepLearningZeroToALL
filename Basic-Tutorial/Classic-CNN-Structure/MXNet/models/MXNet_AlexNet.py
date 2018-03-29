#-*-coding:utf-8-*-
from mxnet.gluon import nn

class MXNet_AlexNet(nn.HybridBlock):
    def __init__(self,num_classes,verbose=False,**kwargs):
        super(MXNet_AlexNet,self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            # 第一阶段
            net.add(nn.Conv2D(channels=96,kernel_size=11,strides=4,activation="relu"))
            net.add(nn.MaxPool2D(pool_size=3,strides=2))
            # 第二阶段
            net.add(nn.Conv2D(channels=256,kernel_size=5,padding=2,activation="relu"))
            net.add(nn.MaxPool2D(pool_size=3,strides=2))
            # 第三阶段
            net.add(nn.Conv2D(channels=384,kernel_size=3,padding=1,activation="relu"))
            net.add(nn.Conv2D(channels=384,kernel_size=3,padding=1,activation="relu"))
            net.add(nn.Conv2D(channels=256,kernel_size=3,padding=1,activation="relu"))
            net.add(nn.MaxPool2D(pool_size=3,strides=2))
            # 第四阶段
            net.add(nn.Flatten())
            net.add(nn.Dense(4096,activation="relu"))
            net.add(nn.Dropout(.5))
            # 第五阶段
            net.add(nn.Dense(4096,activation="relu"))
            net.add(nn.Dropout(.5))
            # 第六阶段
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x, ):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s' % (i + 1, out.shape))
        return out


