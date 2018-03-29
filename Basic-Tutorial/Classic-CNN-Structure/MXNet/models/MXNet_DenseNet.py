#-*-coding:utf-8-*-

from mxnet.gluon import nn
from mxnet import nd

class DenseBlock(nn.HybridBlock):
    # layer 表示层级，growth_rate 表示每个卷积输出的通道数
    def __init__(self,layers,growth_rate,**kwargs):
        super(DenseBlock,self).__init__(**kwargs)
        self.net = net = nn.HybridSequential()
        for i in range(layers):
            net.add(self.conv_block(channels=growth_rate))

    def hybrid_forward(self, F, x, *args, **kwargs):
        for layer in self.net:
            out = layer(x)
            x = nd.concat(x,out,dim=1)
        return x

    def conv_block(self,channels):
        out = nn.HybridSequential()
        out.add(
            nn.BatchNorm(),
            nn.Conv2D(channels=channels,kernel_size=3,padding=1,activation="relu")
        )
        return out


class MXNet_DenseNet(nn.HybridBlock):
    def __init__(self,num_class,verbose=False,init_channels=64,**kwargs):
        super(MXNet_DenseNet,self).__init__(**kwargs)
        self.verbose = verbose
        block_layers = [6,12,24,16]
        growth_rate = 64
        with self.name_scope():
            # block 1
            b1 = nn.HybridSequential()
            b1.add(
                nn.Conv2D(init_channels,kernel_size=7,strides=2,padding=3),
                nn.BatchNorm(),
                nn.Activation("relu"),
                nn.MaxPool2D(pool_size=3,strides=2,padding=1)
            )
            # block 2 :DenseBlock
            b2 = nn.HybridSequential()
            channels = init_channels
            for i,layers in enumerate(block_layers):
                b2.add(DenseBlock(layers=layers,growth_rate=growth_rate))
                channels += layers*growth_rate
                if i != (len(block_layers)-1):
                    b2.add(self.transition_block(channels//2))
            # block 3
            b3 = nn.HybridSequential()
            b3.add(
                nn.BatchNorm(),
                nn.Activation("relu"),
                nn.AvgPool2D(pool_size=1),
                nn.Flatten(),
                nn.Dense(num_class)
            )
            self.net = net = nn.HybridSequential()
            net.add(b1,b2,b3)

    def hybrid_forward(self, F, x, *args, **kwargs):
        out = x
        for i,b in enumerate(self.net):
            out = b(x)
            if self.verbose:
                print("Block %d output %s" % (i+1,out.shape))
        return out

    # 过渡块，主要是用来 对通道数激增进行过渡，将长宽减半，同时改变通道数目
    def transition_block(self,channels):
        out = nn.HybridSequential()
        out.add(
            nn.BatchNorm(),
            nn.Activation("relu"),
            nn.Conv2D(channels=channels,kernel_size=1),
            nn.AvgPool2D(pool_size=2,strides=2)
        )
        return out
