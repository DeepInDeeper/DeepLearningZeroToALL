#-*-coding:utf-8-*-
# ResNet-18

from mxnet.gluon import nn
from mxnet import nd

class Residual(nn.HybridBlock):
    def __init__(self,channels,same_shape=True,**kwargs):
        super(Residual,self).__init__(**kwargs)
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2D(channels=channels,kernel_size=3,padding=1,strides=strides)
        self.bn1 = nn.BatchNorm()

        self.conv2 = nn.Conv2D(channels=channels,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm()

        if not same_shape:
            self.conv3 = nn.Conv2D(channels,kernel_size=1,strides=strides)

    def hybrid_forward(self, F, x, *args, **kwargs):
        out = nd.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return nd.relu(x+out)

class MXNet_ResNet(nn.HybridBlock):
    def __init__(self,num_class,verbose=False,**kwargs):
        super(MXNet_ResNet,self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            # block 1
            b1 = nn.Conv2D(channels=64,kernel_size=7,strides=2)
            # block 2
            b2 = nn.HybridSequential()
            b2.add(
                nn.MaxPool2D(pool_size=3,strides=2),
                Residual(64),
                Residual(64)
            )
            #block 3
            b3 = nn.HybridSequential()
            b3.add(
                Residual(128,same_shape=False),
                Residual(128)
            )
            # block 4
            b4 = nn.HybridSequential()
            b4.add(
                Residual(256,same_shape=False),
                Residual(256)
            )
            # block 5
            b5 = nn.HybridSequential()
            b5.add(
                Residual(512,same_shape=False),
                Residual(512)
            )
            # block 6
            b6 = nn.HybridSequential()
            b6.add(
                nn.AvgPool2D(pool_size=3),
                nn.Dense(num_class)
            )

            # all block together
            self.net = net = nn.HybridSequential()
            net.add(b1,b2,b3,b4,b5,b6)

    def hybrid_forward(self, F, x, *args, **kwargs):
        out = x
        for i,b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print("Block %d output %s" % (i+1,out.shape))
        return out

