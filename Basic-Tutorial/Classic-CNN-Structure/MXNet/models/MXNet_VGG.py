#-*-coding:utf-8-*-
from mxnet.gluon import nn

class MXNet_VGG(nn.HybridBlock):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(MXNet_VGG, self).__init__(**kwargs)
        self.verbose = verbose
        architecture = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            net.add(self.vgg_stack(architecture))
            net.add(nn.Flatten())
            net.add(nn.Dense(4096,activation="relu"))
            net.add(nn.Dropout(0.5))
            net.add(nn.Dense(4096,activation="relu"))
            net.add(nn.Dropout(0.5))
            net.add(nn.Dense(num_classes))

    @staticmethod
    def vgg_block(num_convs,channels):
        out = nn.HybridSequential()
        for _ in range(num_convs):
            out.add(nn.Conv2D(channels=channels,kernel_size=3,padding=1,activation="relu"))
        out.add(nn.MaxPool2D(pool_size=2,strides=2))
        return out

    def vgg_stack(self,architecture):
        out = nn.HybridSequential()
        for (num_convs,channels) in architecture:
            out.add(self.vgg_block(num_convs,channels))
        return out

    def hybrid_forward(self, F, x,):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s' % (i+1, out.shape))
        return out

