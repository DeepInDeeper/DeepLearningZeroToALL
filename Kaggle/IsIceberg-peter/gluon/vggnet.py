#coding:utf-8
from mxnet.gluon import nn
from mxnet import nd
from mxnet import init

## VGG 的模块
##kernel(3×3) 的卷积层然后接上一个池化层，之后再将这个模块重复多次

class vgg_block(nn.HybridBlock):
	"""docstring for vgg_block"""
	def __init__(self, num_convs,channels):
		super(vgg_block, self).__init__()
		self.num_convs = num_convs

		with self.name_scope():
			self.conv1 = nn.Conv2D(channels=channels,kernel_size=3,padding=1,activation="relu")
			if num_convs == 2:
				self.conv2 = nn.Conv2D(channels=channels,kernel_size=3,padding=1,activation="relu")
			#self.pool = nn.MaxPool2D(pool_size=2,strides=2)
			self.maxpool = nn.MaxPool2D(pool_size=2, strides=2)

	def hybrid_forward(self,F,x):
		out = self.conv1(x)
		if self.num_convs == 2:
			out = self.conv2(out)
		out = self.maxpool(out)
		return out


class VggNet(nn.HybridBlock):
	"""docstring for VggNet"""
	def __init__(self, num_outputs,verbose=False):
		super(VggNet, self).__init__()
		self.verbose = verbose
		architecture = ((1,64), (1,128), (2,256), (2,512), (2,512))
		with self.name_scope():
			net = self.net = nn.HybridSequential()

			for (num_convs, channels) in architecture:
				net.add(vgg_block(num_convs, channels))
				#net.add(nn.MaxPool2D(pool_size=2, strides=2))

			net.add(nn.Flatten())
			net.add(nn.Dense(1024,activation="relu"))
			net.add(nn.Dropout(.8))
			net.add(nn.Dense(1024,activation="relu"))
			net.add(nn.Dropout(.8))
			net.add(nn.Dense(num_outputs))

	def hybrid_forward(self,F,x):
		out = x
		for i,b in enumerate(self.net):
			out = b(out)
			if self.verbose:
				print('Block %d output: %s'%(i+1, out.shape))		
		return out

def vgg_net(ctx):
	num_outputs = 2
	net = VggNet(num_outputs)
	net.initialize(ctx=ctx, init=init.Xavier())
	return net
		

		