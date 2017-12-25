#coding:utf-8

from mxnet.gluon import nn
from mxnet import nd
from mxnet import init

## model
class Conv(nn.HybridBlock):
    """docstring for Conv"""
    def __init__(self, channels,kernel_size,pool_size,strides,dropout):
        super(Conv, self).__init__()
        with self.name_scope():
            self.conv = nn.Conv2D(channels,kernel_size=kernel_size,activation="relu")
            self.bn = nn.BatchNorm()
            self.pool = nn.MaxPool2D(pool_size=pool_size,strides=strides)
            self.drop = nn.Dropout(dropout)

    def hybrid_forward(self,F,x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.pool(out)
        out = self.drop(out)
        return out

class MyModel(nn.HybridBlock):
    """docstring for MyModel"""
    def __init__(self,num_outputs,verbose=False):
        super(MyModel, self).__init__()
        with self.name_scope():
            self.verbose = verbose
            net = self.net = nn.HybridSequential()

            net.add(Conv(channels=8,kernel_size=3,pool_size=3,strides=2,dropout=0.2))
            net.add(Conv(channels=16,kernel_size=3,pool_size=2,strides=2,dropout=0.2))
            net.add(Conv(channels=32,kernel_size=3,pool_size=2,strides=2,dropout=0.3))
            net.add(Conv(channels=64,kernel_size=3,pool_size=2,strides=2,dropout=0.3))

            net.add(nn.Flatten())
            net.add(nn.Dense(512))
            net.add(nn.Dropout(0.2))
            net.add(nn.Dense(256))
            net.add(nn.Dropout(0.5))
            net.add(nn.Dense(num_outputs))
    def hybrid_forward(self,F,x):
        out = x
        for i,b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
            
        return out




## mynet
class ConvRes(nn.HybridBlock):
    """docstring for ConvRes"""
    def __init__(self, channels,same_shape):
        super(ConvRes, self).__init__()
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if same_shape else 2
          
            self.bn1 = nn.BatchNorm()
            self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1)
            #self.act1 = nn.LeakyReLU(alpha=1)
            if not same_shape:
                self.conv3 = nn.Conv2D(channels, kernel_size=1,
                                      strides=strides)          
    def hybrid_forward(self,F,x):
        out = self.bn1(x)
        out = self.conv1(out)
        out = F.tanh(out)
        if not self.same_shape:
            x = self.conv3(x)
        return F.tanh(out+x)


        

class ConvCNN(nn.HybridBlock):
    """docstring for ConvCNN"""
    def __init__(self, channels,kernel_size,pool_size,avg=True): 
        super(ConvCNN, self).__init__()
        self.avg= avg
        self.avgpool = nn.AvgPool2D(pool_size=pool_size)
        with self.name_scope():
            self.conv = nn.Conv2D(channels=32, kernel_size=kernel_size, padding=2,strides=1)
            self.bn = nn.BatchNorm()
            self.act = nn.LeakyReLU(alpha=1)
            self.maxpool = nn.MaxPool2D(pool_size=pool_size, strides=pool_size)


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
            net.add(ConvCNN(channels=32,kernel_size=7,pool_size=4,avg=False))
            net.add(nn.Dropout(.7))
            net.add(ConvCNN(channels=32,kernel_size=5,pool_size=2,avg=True))
            net.add(ConvCNN(channels=32,kernel_size=5,pool_size=2,avg=True))

            #net.add(ConvRes(channels=64,same_shape=False))
            
            net.add(nn.Flatten())
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self,F,x):
        out = x
        for i,b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
            
        return out


def get_simple_net(ctx,net):
    num_outputs = 2
    if net == "mynet":
        net = SimpleNet(num_outputs)
    elif net == "resnet":
        net = ResNet(num_outputs)
    elif net == "mymode":
        net = MyModel(num_outputs)

    net.initialize(ctx=ctx, init=init.Xavier())
    return net






