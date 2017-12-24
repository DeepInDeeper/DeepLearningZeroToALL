#coding:utf-8

from mxnet.gluon import nn
from mxnet import nd
from mxnet import init

## 使用ResNet框架
class Residual(nn.HybridBlock):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1,
                                  strides=strides)
            self.bn1 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm()
            if not same_shape:
                self.conv3 = nn.Conv2D(channels, kernel_size=1,
                                      strides=strides)

    def hybrid_forward(self, F, x):
        out = F.tanh(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return F.tanh(out + x)

    
class ResNet(nn.HybridBlock):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            # 模块1
            net.add(nn.Conv2D(channels=32, kernel_size=3, strides=1, 
                              padding=1))
            net.add(nn.BatchNorm())
            net.add(nn.Activation(activation='tanh'))
            # 模块2
            for _ in range(3):
                net.add(Residual(channels=32))
            # 模块3
            net.add(Residual(channels=64, same_shape=False))
            for _ in range(2):
                net.add(Residual(channels=64))
            # 模块4
            net.add(Residual(channels=128, same_shape=False))
            for _ in range(2):
                net.add(Residual(channels=128))
            # 模块5
            net.add(nn.GlobalAvgPool2D())
            net.add(nn.Flatten())
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        return out




## mynet
class ConvRes(nn.HybridBlock):
    """docstring for ConvRes"""
    def __init__(self, channels,same_shape=True):
        super(ConvRes, self).__init__()
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.bn = nn.BatchNorm()
            self.conv = nn.Conv2D(channels,kernel_size=3,padding=1,strides=strides)
            self.act = nn.LeakyReLU(alpha=1)
            if not same_shape:
                self.conv2 = nn.Conv2D(channels,kernel_size=1,strides=strides)
    
    def hybrid_forward(self,F,x):
        out = self.bn(x)
        out = self.conv(x)
        out = self.act(x)
        if not self.same_shape:
            out = self.conv2(out)
        return out
        

class ConvCNN(nn.HybridBlock):
    """docstring for ConvCNN"""
    def __init__(self, channels,avg=True): 
        super(ConvCNN, self).__init__()
        self.avg= avg
        self.avgpool = nn.AvgPool2D(pool_size=2)
        with self.name_scope():
            self.conv = nn.Conv2D(channels=32, kernel_size=3, strides=1, padding=1)
            self.bn = nn.BatchNorm()
            self.act = nn.LeakyReLU(alpha=1)
            self.maxpool = nn.AvgPool2D(pool_size=2, strides=2)


    def hybrid_forward(self,F,x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.maxpool(out)
        out = self.act(out)
        if self.avg:
            out = self.avgpool(out)
        return out


class SimpleNet(nn.HybridBlock):
    """docstring for SimpleNet"""
    def __init__(self,num_classes,verbose=False):
        super(SimpleNet, self).__init__()
        self.verbose = verbose
        with self.name_scope():
            net = self.net = nn.HybridSequential()

            net.add(nn.AvgPool2D(pool_size=1))

            for _ in range(3):
                net.add(ConvCNN(channels=32))

            #net.add(ConvRes(channels=64,same_shape=False))
            net.add(nn.Dropout(.7))
            net.add(nn.Flatten())
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self,F,x):
        out = x
        for i,b in enumerate(self.net):
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
            out = b(out)
        return out


def get_simple_net(ctx,net):
    num_outputs = 2
    if net == "mynet":
        net = SimpleNet(num_outputs)
    elif net == "resnet":
        net = ResNet(num_outputs)
    net.initialize(ctx=ctx, init=init.Xavier())
    return net






